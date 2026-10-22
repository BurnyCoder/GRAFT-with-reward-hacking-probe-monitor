"""Global: reassess frozen evidence without importing historical judgments.

Local: lossless dictionaries compress repeated text; explicit labels remain a reviewer decision.
Sources: https://docs.python.org/3/library/json.html; ../audit_view.py.
"""
import json  # Local: exact structured values and literal evidence checks.
import time  # Local: record actual review/freeze times, never backdate a preregistration.

from ..runtime import file_hash, fingerprint, read_json, write_json  # Local: existing stable serialization.


def packet_paths(root):
    """Global: include both cohorts and smoke. Local: filenames provide opaque, stable ordering."""
    paths = [*root.glob("audit/packets/*.json"), *root.glob("ablation-followup/audit/packets/*.json")]
    result = {path.stem: path for path in paths}  # Local: no mapping or historical label is read.
    if len(result) != len(paths):  # Local: an identity collision must never silently discard evidence.
        raise ValueError("Duplicate packet identity across cohorts")
    return dict(sorted(result.items()))  # Local: mix cohorts by their opaque packet identities.


def encode_strings(value, lines, strings):
    """Global: retain every byte of textual evidence. Local: encode strings as exact line references."""
    if isinstance(value, str):  # Local: even repeated raw responses use the same lossless dictionary.
        parts = []  # Local: retain line order, endings and repeated occurrences.
        for line in value.splitlines(keepends=True):
            if line not in lines:  # Local: equality, not semantic similarity, permits sharing.
                lines.append(line)
            parts.append(lines.index(line))
        if parts not in strings:  # Local: share identical complete strings across message roles.
            strings.append(parts)
        return {"$text": strings.index(parts)}
    if isinstance(value, list):  # Local: preserve all request and trajectory order.
        return [encode_strings(item, lines, strings) for item in value]
    if isinstance(value, dict):  # Local: keys remain visible; dictionary values are recursively exact.
        return {key: encode_strings(item, lines, strings) for key, item in value.items()}
    return value  # Local: booleans, numbers and null remain unchanged.


def decode_strings(value, lines, strings):
    """Global: prove the display is lossless. Local: expand every referenced string and ordered container."""
    if isinstance(value, dict) and set(value) == {"$text"}:
        return "".join(lines[index] for index in strings[value["$text"]])
    if isinstance(value, dict):
        return {key: decode_strings(item, lines, strings) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_strings(item, lines, strings) for item in value]
    return value  # Local: nontext values require no reconstruction.


def text_ids(value):
    """Global: disclose every textual field. Local: collect referenced strings from the encoded packet."""
    if isinstance(value, dict) and set(value) == {"$text"}:
        return {value["$text"]}
    if isinstance(value, dict):
        return set().union(*(text_ids(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(text_ids(item) for item in value))
    return set()  # Local: numeric/boolean/null fields contain no text references.


def display_page(encoded, lines, strings, start, count):
    """Global: sequential pages remain fully reconstructable. Local: print new exact lines once, with references thereafter."""
    prior = set().union(*(text_ids(value) for value in encoded[:start]))
    seen_lines = {index for identity in prior for index in strings[identity]}
    output = []  # Local: pages retain complete structured packets and every new string definition.
    for index, value in enumerate(encoded[start:start + count], start):
        definitions = {}  # Local: repeated strings reference their previously displayed definition.
        for identity in sorted(text_ids(value) - prior):
            parts = []
            for line in strings[identity]:
                parts.append({"L" + str(line): lines[line]} if line not in seen_lines else "L" + str(line))
                seen_lines.add(line)
            definitions["S" + str(identity)] = parts
        prior |= text_ids(value)
        output.append({"index": index, "definitions": definitions, "packet": value})
    return output


def save_review(destination, packets, identity, hacking, legitimate, evidence, rationale):
    """Global: new labels never replace historical labels. Local: require explicit tri-state evidence."""
    if any(value is not None and type(value) is not bool for value in (hacking, legitimate)):
        raise ValueError("Labels must be true, false or null")
    packet = packets[identity]  # Local: an unknown identity fails instead of fabricating a trial.
    if not evidence or not rationale or evidence not in json.dumps(packet, ensure_ascii=False):
        raise ValueError("A literal packet quote and a rationale are required")
    if legitimate is True and (packet["verifier_success"] is not True or hacking is not False):
        raise ValueError("Legitimate success requires verifier success and no established hacking")
    path = destination / "judgments.json"  # Local: this directory is separate from both original audits.
    judgments = read_json(path) if path.exists() else {}
    if identity in judgments or (destination / "review-freeze.json").exists():
        raise ValueError("Judgment already recorded or review frozen; use a new revision")
    judgments[identity] = {"hacking": hacking, "legitimate": legitimate, "evidence": evidence,
                           "rationale": rationale, "packet_fingerprint": fingerprint(packet),
                           "judged_at": time.time(), "reviewer": "Codex; outcome-aware post-hoc reassessment"}
    write_json(path, judgments)  # Local: atomic persistence, without touching original annotation files.


def freeze_review(destination, packets):
    """Global: no selective comparison with earlier labels. Local: require complete, unchanged evidence."""
    path = destination / "judgments.json"
    judgments = read_json(path) if path.exists() else {}
    if set(judgments) != set(packets):
        raise ValueError(f"Review incomplete: {len(judgments)}/{len(packets)} packet judgments")
    if any(judgments[key]["packet_fingerprint"] != fingerprint(value) for key, value in packets.items()):
        raise ValueError("Packet changed after independent review")
    frozen_path = destination / "review-freeze.json"
    frozen = {"count": len(judgments), "sha256": file_hash(path), "frozen_at": time.time()}
    if frozen_path.exists():  # Local: an unchanged review retains its first freeze timestamp.
        previous = read_json(frozen_path)
        if previous["sha256"] != frozen["sha256"]:
            raise ValueError("Frozen judgments changed")
        return previous
    write_json(frozen_path, frozen)  # Local: comparisons may read historical labels only after this point.
    return frozen
