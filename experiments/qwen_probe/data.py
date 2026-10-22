"""Global: task-isolated, shared data for every treatment. Local: reuse released conversion.

Sources: sft/scripts/data_prep.py and the frozen decisions in docs/protocol.md.
"""
import collections  # Local: exclusions and provenance counts.
import json  # Local: existing trainer's JSONL interchange.
import random  # Local: score-blind monitor control.
from pathlib import Path  # Local: versioned task split.

from .download import MANIFEST  # Local: model/data provenance for fingerprints.
from .conversion import conversion_identity, require_current_conversion, upstream_conversion, validate_targets
from .protocol import partition_tasks, select_balanced, shuffled_flags
from .runtime import ROOT, event, fingerprint, read_json, write_json
from .detector.preprocess import VERSION, normalize_messages, trajectory_messages

ARMS = ("no_intervention", "probe_graft", "qwen_judge_graft", "original_graft", "probe_filtering", "random_graft")


def inventory(root):
    """Global: label-independent preprocessing. Local: tokenize and identify duplicate inputs."""
    if (root / "data/inventory.json").exists() or (root / "data/lock.json").exists():
        raise ValueError("Inventory already exists; preserve it and use a separate GRAFT_ARTIFACTS directory")
    from transformers import AutoTokenizer  # Local: use exact learner tokenizer for exclusions.
    converter = upstream_conversion(root)
    from scripts.gr.loader import _patch_chat_template_for_assistant_mask  # Local: same template as SFT.
    split = read_json(Path(__file__).with_name("task_split.json"))
    train, calibration, evaluation = partition_tasks(split["train_tasks"], split["eval_tasks"])
    assignments = {task: name for name, tasks in (("train", train), ("calibration", calibration), ("heldout", evaluation)) for task in tasks}
    tokenizer = AutoTokenizer.from_pretrained(root / "models/learner", local_files_only=True)
    _patch_chat_template_for_assistant_mask(tokenizer)  # Local: prevent count-vs-training template mismatch.
    rows, excluded = [], collections.Counter()
    for path, classification, label, tree, task, source_model in converter.iter_trajectories():
        if classification != "baseline" and classification not in converter.HACK_CLASSES:
            excluded["unselected_class"] += 1
            continue
        if task not in assignments:
            excluded["outside_published_split"] += 1
            continue
        trajectory = read_json(path)  # Local: original labels remain outside detector input.
        messages = converter.atif_to_messages(trajectory)
        if len(messages) < 2:
            excluded["no_assistant"] += 1
            continue
        target_checks = validate_targets(trajectory, messages)  # Local: fail before tokenization if shell/completion semantics are invalid.
        monitor_messages = trajectory_messages(trajectory)
        if monitor_messages and monitor_messages[0]["role"] == "user":
            monitor_messages[0]["content"] = converter.strip_red_team(monitor_messages[0]["content"])
        turns = normalize_messages(monitor_messages)
        if not turns:
            excluded["no_monitor_turns"] += 1
            continue
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True,
                                                return_assistant_tokens_mask=True)
        if not any(encoded["assistant_masks"]):
            excluded["no_loss_tokens"] += 1
            continue
        rows.append({"id": fingerprint(str(path.relative_to(ROOT))), "task": task, "partition": assignments[task],
                     "label": int(classification in converter.HACK_CLASSES), "classification": classification,
                     "source": str(path.relative_to(ROOT)), "source_model": source_model,
                     "tokens": len(encoded["input_ids"]), "messages": messages, "turns": turns,
                     "sft_target_checks": target_checks,  # Local: disclose nonfatal source warnings alongside preserved commands.
                     "content_hash": fingerprint(turns)})  # Local: metadata never concatenated with turns.
    heldout_hashes = {r["content_hash"] for r in rows if r["partition"] != "train"}
    seen, unique = set(), []
    for row in rows:
        key = (row["partition"], row["content_hash"])
        if row["partition"] == "train" and row["content_hash"] in heldout_hashes:
            excluded["cross_partition_exact_duplicate"] += 1
        elif key in seen:
            excluded["within_partition_duplicate"] += 1
        else:
            unique.append(row)
            seen.add(key)
    provenance = {"source_revision": "d8a29613235a0ef56a8b70b3142626a533da28c2", "preprocessing": VERSION,
                  "sft_conversion": conversion_identity(root),  # Local: monitor normalization is a separate transformation.
                  "train_tasks": train, "calibration_tasks": calibration, "heldout_tasks": evaluation,
                  "exclusions": dict(excluded), "counts": dict(collections.Counter(r["partition"] for r in unique)),
                  "upstream_probe_exposure": "Unresolved: released notebook names TRACE, MALT, Realistic Reward Hacks and BJS corpus; complete training identifiers unavailable"}
    write_json(root / "data/inventory.json", unique)
    write_json(root / "data/provenance.json", provenance)
    event("inventory", **provenance)  # Local: counts and unresolved exposure, not inferred absence of leakage.
    return unique


def dataset_fingerprint(root, rows, configuration):
    """Global: bind caches to actual training text. Local: include converted targets as well as monitor hashes."""
    return fingerprint({"model": MANIFEST["learner"], "tokenizer": read_json(root / "downloads.json")["model_files"]["learner"],
                        "data": [(r["id"], r["content_hash"], fingerprint(r["messages"])) for r in rows],
                        "preprocessing": VERSION, "sft_conversion": require_current_conversion(root),
                        "configuration": configuration})  # Local: recompute target hashes instead of trusting cached ones.


def fix_dataset(root, max_length):
    """Global: shared data lock. Local: select before reading any monitor scores."""
    conversion = require_current_conversion(root)  # Local: reject historical conversion before creating a new lock.
    rows = read_json(root / "data/inventory.json")
    eligible = [r for r in rows if r["partition"] == "train" and r["tokens"] <= max_length]
    selected = select_balanced(eligible)
    heldout = []  # Local: one deterministic example per available task/label for detector validation.
    for task in sorted({r["task"] for r in rows if r["partition"] == "heldout"}):
        for label in (0, 1):
            options = [r for r in rows if r["task"] == task and r["partition"] == "heldout" and r["label"] == label]
            heldout.extend(sorted(options, key=lambda r: r["id"])[:1])
    calibration = [r for r in rows if r["partition"] == "calibration"]
    configuration = {"max_length": max_length, "seed": 42, "batch": 48, "epochs": 5,
                     "width": 64, "lr": 5e-4, "anchors": 0.01, "kappa": 2}
    identity = dataset_fingerprint(root, selected + calibration + heldout, configuration)
    lock = {"fingerprint": identity, "sft_conversion": conversion, "configuration": configuration, "train": [r["id"] for r in selected],
            "calibration": [r["id"] for r in calibration], "heldout": [r["id"] for r in heldout],
            "length_exclusions": sum(r["partition"] == "train" and r["tokens"] > max_length for r in rows)}
    path = root / "data/lock.json"
    if path.exists() and read_json(path) != lock:
        raise ValueError("Dataset already frozen with different inputs")
    write_json(path, lock)
    event("dataset_locked", **lock)
    return lock


def write_pools(root):
    """Global: same records, different flags. Local: emit existing trainer's retain/forget files."""
    require_current_conversion(root)  # Local: never regenerate historical pools with a corrected converter identity.
    lock = read_json(root / "data/lock.json")
    by_id = {r["id"]: r for r in read_json(root / "data/inventory.json")}
    frozen = [by_id[i] for partition in ("train", "calibration", "heldout") for i in lock[partition]]
    if dataset_fingerprint(root, frozen, lock["configuration"]) != lock["fingerprint"]:
        raise ValueError("Dataset changed after it was frozen")  # Local: block edited target text before any pool is overwritten.
    rows = [by_id[identity] for identity in lock["train"]]
    scores = read_json(root / "monitor/scores.json")
    threshold = read_json(root / "monitor/threshold.json")["threshold"]
    probe_flags = {r["id"]: scores[r["id"]]["score"] > threshold for r in rows}
    positives = sorted(r["id"] for r in rows if r["label"])
    oracle = set(random.Random(42).sample(positives, round(0.2 * len(positives))))
    random_flags = shuffled_flags(rows, probe_flags)
    flags_by_arm = {"probe_graft": probe_flags, "probe_filtering": probe_flags, "random_graft": random_flags,
                    "no_intervention": {r["id"]: False for r in rows}, "original_graft": {r["id"]: r["id"] in oracle for r in rows}}
    if (root / "judge/threshold.json").exists():  # Local: independent original phases can run before judge validation.
        judge_scores = read_json(root / "judge/scores.json")
        judge_threshold = read_json(root / "judge/threshold.json")["threshold"]
        if all(row["id"] in judge_scores for row in rows):
            flags_by_arm["qwen_judge_graft"] = {r["id"]: judge_scores[r["id"]]["score"] > judge_threshold for r in rows}
    summary = {}
    for arm, flags in flags_by_arm.items():
        directory = root / "data" / arm
        directory.mkdir(parents=True, exist_ok=True)
        for flagged, pool in ((False, "retain"), (True, "forget")):
            records = [{"record_id": r["id"], "messages": r["messages"], "meta": {"classification": r["classification"]}}
                       for r in rows if flags[r["id"]] == flagged]
            (directory / f"gr_train_{pool}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        summary[arm] = {"flagged": sum(flags.values()), "total": len(rows), "flags": flags}
    write_json(root / "data/arms.json", summary)
    for row in rows:  # Local: complete training messages, including target responses, enter timestamped logs.
        event("sft_record", record_id=row["id"], messages=row["messages"], tokens=row["tokens"])
    return summary
