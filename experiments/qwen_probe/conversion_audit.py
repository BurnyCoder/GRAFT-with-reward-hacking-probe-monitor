"""Global: reproduce the dated correction from saved evidence without running models.

Local: read historical pools, source trajectories, loss logs and response packets; write a new check file only.
Sources: conversion.py's pinned parser; https://docs.python.org/3/library/hashlib.html.
"""
import argparse  # Local: a separate command cannot accidentally invoke the model pipeline.
import json  # Local: read saved JSONL targets and losses.
from collections import Counter  # Local: aggregate exact observed counts.
from pathlib import Path  # Local: explicit input and new output locations.

from .conversion import PARSER_PATH, conversion_identity, target_parser, upstream_conversion, validate_targets
from .runtime import ROOT, event, file_hash, fingerprint, read_json, setup, write_json


def check_saved_targets(root):
    """Global: validate every locked training record. Local: compare historical targets with corrected in-memory copies."""
    lock_path, pool_path = root / "data/lock.json", root / "data/no_intervention/gr_train_retain.jsonl"
    ids = read_json(lock_path)["train"]  # Local: preserve the historical sample rather than selecting a favorable subset.
    records = [json.loads(line) for line in pool_path.read_text().splitlines() if line]  # Local: read, never rewrite, the original pool.
    by_id = {r["record_id"]: r for r in records}  # Local: associate saved targets with pinned source paths.
    if len(ids) != len(set(ids)) or len(records) != len(by_id) or set(ids) != set(by_id):
        raise ValueError("Saved pool must contain every locked training record exactly once")
    converter = upstream_conversion(root)  # Local: only the pure conversion function will run.
    sources = {fingerprint(str(path.relative_to(ROOT))): path for path, *_ in converter.iter_trajectories()
               if fingerprint(str(path.relative_to(ROOT))) in by_id}  # Local: reuse the released source enumeration.
    if set(sources) != set(by_id):  # Local: a missing source invalidates completeness, not the denominator.
        raise ValueError("Missing pinned source trajectories for the locked training pool")
    totals, details = Counter(), []  # Local: publish aggregate checks plus reproducible per-record hashes.
    for identity in ids:  # Local: retain the frozen study order.
        source = read_json(sources[identity])  # Local: metadata is evidence, never a label to be edited.
        saved = by_id[identity]["messages"]  # Local: historical training target text.
        corrected = converter.atif_to_messages(source)  # Local: no tokenizer, model or shell command is executed.
        counts = validate_targets(source, corrected)  # Local: fail if any corrected turn cannot parse losslessly.
        old_targets = [m["content"] for m in saved if m["role"] == "assistant"]  # Local: inspect all historical assistant turns.
        old_parsed = [target_parser().parse_response(text) for text in old_targets]  # Local: actual pinned parser acceptance.
        if len(old_targets) != counts["assistant_turns"]:  # Local: conversion must not change trajectory length.
            raise ValueError("Historical and corrected assistant-turn counts differ")
        rejected = sum(bool(p.error) for p in old_parsed)  # Local: warnings alone are not counted as historical rejection.
        totals.update(counts)  # Local: source-preserving corrected counts cover every training trajectory.
        totals.update({"trajectories": 1, "historical_rejected_turns": rejected,
                       "historical_affected_trajectories": int(rejected > 0),
                       "historical_completion_turns": sum(bool(p.is_task_complete) for p in old_parsed)})
        details.append({"id": identity, "source": str(sources[identity].relative_to(ROOT)),
                        "source_sha256": file_hash(sources[identity]), "saved_messages_hash": fingerprint(saved),
                        "corrected_messages_hash": fingerprint(corrected), "historical_rejected_turns": rejected, **counts})
    return {"totals": dict(totals), "corrected_rejected_turns": 0,
            "lock_sha256": file_hash(lock_path), "historical_pool_sha256": file_hash(pool_path), "records": details}


def check_degraded_arms(root):
    """Global: disclose failure without inferring its cause. Local: reparse saved responses and summarize raw training losses."""
    audited_path = root / "audit/unblinded.json"  # Local: use all closed-pilot records, excluding the engineering smoke.
    audited, result = read_json(audited_path), {}  # Local: original labels remain untouched.
    for arm in ("qwen_judge_graft", "original_graft"):  # Local: the two degraded arms named in the correction.
        counts, packets = Counter(), {}  # Local: retain denominators and exact packet identities.
        for row in audited:  # Local: include both prompt conditions for each named deployment.
            if row["smoke"] or row["arm"] != arm:
                continue  # Local: exclude smoke and unrelated treatments without selecting on outcomes.
            path = root / "audit/packets" / f"{row['packet_id']}.json"  # Local: frozen evidence read by the intent auditor.
            packet = read_json(path)  # Local: raw responses include invalid outputs omitted from parsed trajectories.
            counts["episodes"] += 1  # Local: count every audited episode, including agent failures.
            packets[row["packet_id"]] = file_hash(path)  # Local: make response evidence traceable without republishing raw logs.
            for call in packet["raw_model_calls"]:  # Local: no regeneration or reclassification is performed.
                parsed = target_parser().parse_response(call["response"])  # Local: evaluate the original interface contract.
                counts["responses"] += 1  # Local: denominator includes every saved response.
                counts["rejected_responses"] += bool(parsed.error)  # Local: errors are distinct from warnings or poor actions.
        path = root / "checkpoints" / arm / "metrics.jsonl"  # Local: raw optimizer logs, not a prior audit's conclusion.
        curve = [json.loads(line) for line in path.read_text().splitlines() if line]  # Local: do not load checkpoint tensors.
        values = [{"step": r["step"], "loss": r["train/unclassified/loss_mean_local"]}
                  for r in curve if "train/unclassified/loss_mean_local" in r]  # Local: batch means, not fixed-set validation loss.
        result[arm] = {**dict(counts), "unclassified_loss": values, "metrics_sha256": file_hash(path), "packet_sha256": packets}
    return {"unblinded_sha256": file_hash(audited_path), "arms": result}  # Local: annotations are inputs, never revised outputs.


def main():
    """Global: independently reproducible, model-free check. Local: write supplemental evidence to a new path."""
    parser = argparse.ArgumentParser(description=__doc__)  # Local: use a dedicated audit entry point.
    parser.add_argument("--output", type=Path, required=True, help="New JSON path; existing files are never overwritten")
    args = parser.parse_args()  # Local: artifact input root comes from the existing .env GRAFT_ARTIFACTS setting.
    destination = args.output.resolve()  # Local: resolve before creating any new audit output.
    if not destination.is_relative_to(ROOT) or destination.exists():  # Local: never overwrite a historical artifact.
        parser.error("Output must be a new file inside the current workspace")
    root = setup()  # Local: reuse UTC terminal/file logging and .env paths; this does not start the experiment clock.
    event("conversion_audit_start", analysis_status="post-hoc engineering diagnostic", root=str(root))
    targets = check_saved_targets(root)  # Local: validate source semantics for the complete historical training set.
    behavior = check_degraded_arms(root)  # Local: read loss logs and saved responses only.
    result = {"analysis_status": "post-hoc engineering diagnostic; no model execution",
              "audit_implementation_sha256": file_hash(__file__),  # Local: identify the check code independently of historical analyses.
              "sft_conversion": conversion_identity(root), "parser_sha256": file_hash(PARSER_PATH),
              "targets": targets, "degraded_arms": behavior}  # Local: preserve descriptive provenance alongside counts.
    write_json(destination, result)  # Local: save supplemental results separately from all historical files.
    event("conversion_audit_complete", output=str(destination), totals=targets["totals"],
          corrected_rejected_turns=targets["corrected_rejected_turns"])


if __name__ == "__main__":  # Local: importing helpers in regression tests cannot run the audit.
    main()  # Local: orchestrate read-only phases; no training or evaluation entry point is reachable here.
