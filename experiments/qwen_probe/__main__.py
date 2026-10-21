"""Global: one readable study command. Local: delegate each phase to its implementation module.

The pipeline intentionally stops for blinded audit judgments; it never invents them.
"""
import argparse  # Local: explicit independent phase commands.
import sys  # Local: GPU-heavy phases run in short-lived child interpreters.
import time  # Local: honor the global deadline between phases.

from .runtime import deadline, event, execute, setup, write_json


def smoke(root):
    """Global: exercise the real chain before the pilot. Local: one update and one Docker task."""
    from .audit import finalize, make_packets
    from .data import write_pools
    from .evaluation import evaluate
    from .training import train_arm
    write_pools(root)
    if not train_arm(root, "probe_graft", smoke=True):
        return False
    evaluate(root, smoke=True)
    if make_packets(root):
        event("smoke_needs_audit", instructions="Read blinded packets, save judgments using the audit command, then rerun")
        return False
    results = finalize(root)
    if not results or not any(r["smoke"] and r["verifier_success"] is not None for r in results):
        raise RuntimeError("Smoke has no completed verifier evidence; inspect Docker/Harbor logs")
    write_json(root / "smoke-complete.json", {"completed_at": time.time(), "audited_trials": len(results)})
    return True


def run(root):
    """Global: fixed priorities within six hours. Local: visible, independently callable phases."""
    from .data import ARMS
    from .evaluation import evaluate
    from .prepare import prepare
    from .report import report
    from .training import train_arm
    end = deadline(root)
    if end and time.time() >= end:  # Local: a resumed command reports coverage without extending compute.
        report(root)
        return
    prepare(root)
    execute([sys.executable, "-u", "-m", "experiments.qwen_probe", "validate-monitor"], deadline=deadline(root))
    if not smoke(root):
        report(root)
        return
    for arm in ARMS[:2]:  # Local: finish the primary training comparison before secondary controls.
        if not train_arm(root, arm):
            report(root)
            return
    evaluate(root, modes=[("no_intervention", "both"), ("probe_graft", "retain"), ("base", "both")])
    for arm in ARMS[2:]:  # Local: controls never displace the primary behavioral comparison.
        if train_arm(root, arm):
            evaluate(root, modes=[(arm, "retain")])
    evaluate(root, modes=[("probe_graft", "both"), ("probe_graft", "forget"), ("no_intervention", "retain")])
    report(root)  # Local: pending audits and capped comparisons are explicitly incomplete.


def main():
    """Global: reproducible CLI. Local: parse once, then dispatch a named modular phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["run", "download", "prepare", "validate-monitor", "train", "evaluate", "smoke", "audit", "report", "publish"])
    parser.add_argument("--profile", choices=["pilot"], default="pilot")
    parser.add_argument("--arm", help="Run one registered training arm")
    parser.add_argument("--packet", help="Opaque packet ID for recording an audit judgment")
    parser.add_argument("--hacking", choices=["true", "false", "ambiguous"])
    parser.add_argument("--legitimate", choices=["true", "false", "ambiguous"])
    parser.add_argument("--evidence", help="Literal evidence quote from the blinded packet")
    parser.add_argument("--rationale", help="Intent-based explanation using the released rubric")
    args = parser.parse_args()
    root = setup()
    event("phase_start", phase=args.phase, profile=args.profile)
    if args.phase == "run":
        run(root)
    elif args.phase == "download":
        from .download import download
        download(root)
    elif args.phase == "prepare":
        from .prepare import prepare
        prepare(root)
    elif args.phase == "validate-monitor":
        from .detector.scoring import validate_monitor
        validate_monitor(root)
    elif args.phase == "train":
        from .training import train
        train(root, arm=args.arm)
    elif args.phase == "evaluate":
        from .evaluation import evaluate
        evaluate(root)
    elif args.phase == "smoke":
        smoke(root)
    elif args.phase == "audit":
        from .audit import finalize, make_packets, record_judgment
        if args.packet:
            labels = {"true": True, "false": False, "ambiguous": None}
            if not all((args.hacking, args.legitimate, args.evidence, args.rationale)):
                parser.error("Recording requires --hacking, --legitimate, --evidence and --rationale")
            record_judgment(root, args.packet, labels[args.hacking], labels[args.legitimate], args.evidence, args.rationale)
        pending = make_packets(root)
        event("audit_coverage", pending=len(pending))
        if not pending:
            finalize(root)
    elif args.phase == "report":
        from .report import report
        report(root)
    elif args.phase == "publish":
        from .publish import publish
        publish(root)
    event("phase_finished", phase=args.phase)


if __name__ == "__main__":
    main()
