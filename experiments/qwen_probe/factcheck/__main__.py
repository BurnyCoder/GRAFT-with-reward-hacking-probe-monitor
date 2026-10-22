"""Global: explicit, modular saved-evidence review phases; never run a model or finalize an old audit.

Local: all new writes are confined to the chosen review destination and its private display logs.
Source: https://docs.python.org/3/library/argparse.html.
"""
import argparse  # Local: separate viewing, recording and freezing actions.

from ..audit_view import compact_packet  # Local: reuse the tested lossless message-reference display.
from ..runtime import ROOT, event, read_json, setup, write_json  # Local: timestamped full terminal/file logs.
from .evidence import decode_strings, display_page, encode_strings, freeze_review, packet_paths, save_review
from .snapshot import snapshot, verify_preservation


def main():
    """Global: one readable wrapper. Local: dispatch each operation to a focused implementation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("snapshot", "view", "judge", "freeze", "preserve"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--input", help="Reviewer-authored JSON list; no automatic labels are generated")
    args = parser.parse_args()
    private = setup("factcheck-2026-10-22")  # Local: new logs never enter the historical log directory.
    root, destination = private.parent, ROOT / "reports/qwen_probe/factcheck-2026-10-22"
    if args.phase == "snapshot":
        result = snapshot(root, destination)
        event("factcheck_snapshot", tracked=len(result["tracked_files"]), evidence=len(result["evidence"]))
        return
    if args.phase == "preserve":
        event("factcheck_preserved", **verify_preservation(destination))
        return
    paths = packet_paths(root)  # Local: this phase still cannot read historical per-trial labels.
    packets = {key: read_json(path) for key, path in paths.items()}
    if args.phase == "view":
        lines, strings, encoded = [], [], []  # Local: deterministic dictionaries allow exact references across packets.
        for packet in packets.values():
            value = compact_packet(packet)
            encoded.append(encode_strings(value, lines, strings))
            assert decode_strings(encoded[-1], lines, strings) == value  # Local: every displayed structure must round-trip.
        write_json(private / "display-dictionary.json", {"lines": lines, "strings": strings})
        for page in display_page(encoded, lines, strings, args.start, args.count):
            event("factcheck_view", **page)  # Local: log all exact new text and ordered references, never snippets.
        return
    if args.phase == "judge":
        if not args.input:
            parser.error("judge requires --input with explicit reviewer-authored judgments")
        for row in read_json(ROOT / args.input):
            save_review(destination, packets, row["packet_id"], row["hacking"], row["legitimate"], row["evidence"], row["rationale"])
            event("factcheck_judgment", **row)
        return
    event("factcheck_freeze", **freeze_review(destination, packets))  # Local: all packets must be reviewed before comparison.


if __name__ == "__main__":
    main()  # Local: importing helpers never starts a phase.
