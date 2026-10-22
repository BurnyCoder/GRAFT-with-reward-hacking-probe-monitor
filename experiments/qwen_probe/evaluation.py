"""Global: real heldout behavior in pinned Harbor. Local: only our server/processes are owned.

Sources: sft/scripts/eval/run_harbor_eval.sh; https://harborframework.com/docs.
"""
import contextlib  # Local: guarantee server cleanup on errors and timeouts.
import json  # Local: Harbor agent keyword configuration.
import os  # Local: private child environment and process groups.
import signal  # Local: scoped server shutdown.
import subprocess  # Local: pinned Harbor CLI and GPU server processes.
import sys  # Local: root virtual environment.
import threading  # Local: realtime server log draining.
import time  # Local: startup and shared experiment deadlines.
import uuid  # Local: distinguish our server from any unrelated listener on the chosen port.

import requests  # Local: health checks only against localhost.

from .runtime import ROOT, LOGGER, deadline, event, execute, read_json, write_json
from .training import merge_model
from .tasks import ensure_task_images

MODES = [("no_intervention", "both"), ("probe_graft", "retain"), ("base", "both"),
         ("qwen_judge_graft", "retain"),
         ("original_graft", "retain"), ("probe_filtering", "retain"), ("random_graft", "retain"),
         ("probe_graft", "both"), ("probe_graft", "forget"), ("no_intervention", "retain")]


@contextlib.contextmanager
def model_server(model_path, root):
    """Global: one GPU worker. Local: never call the upstream broad pkill shutdown script."""
    port = int(os.environ.get("GRAFT_SERVER_PORT", "8765"))
    instance = uuid.uuid4().hex  # Local: readiness must prove process ownership, not just an open port.
    server_env = os.environ | {"GRAFT_SERVER_INSTANCE": instance,
                              "GRAFT_GENERATION_DIR": str(root / "evaluation/generations" / instance)}
    process = subprocess.Popen([sys.executable, "-u", "-m", "experiments.qwen_probe.server", str(model_path)],
                               cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               bufsize=1, start_new_session=True, env=server_env)
    def drain():
        """Global: full inference transcripts. Local: tee our server stdout to the parent log."""
        for line in process.stdout:
            LOGGER.info(line.rstrip("\n"))
    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    try:
        until = min(time.time() + 240, deadline(root) or float("inf"))
        while time.time() < until:
            if process.poll() is not None:
                raise RuntimeError("Experiment server exited during startup")
            try:
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
                if response.ok and response.json().get("instance") == instance:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        else:
            raise TimeoutError("Experiment server did not become ready before deadline")
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        thread.join(timeout=2)
        event("server_stopped", pid=process.pid, return_code=process.returncode)


def dataset(root, tasks, elicited):
    """Global: preserve released task/prompt setup. Local: reuse converter and prebuilt images."""
    sys.path.insert(0, str(ROOT / "sft"))
    from scripts.eval.prep_eval_dataset import build
    directory = root / "evaluation/datasets" / ("elicited" if elicited else "plain") / ("-".join(tasks))
    split = directory.parent / (directory.name + ".json")
    write_json(split, {"eval_tasks": tasks})
    prompt = ROOT / "sft/third_party/terminal-wrench/prompts/hack_prompt_v5.md" if elicited else None
    build(split, ROOT / "sft/third_party/terminal-wrench/tasks", directory, 120, inject_prompt=prompt)
    for task in tasks:
        path = directory / task / "task.toml"
        text = path.read_text()
        image = "graft-probe-task-" + task.lower().replace("_", "-")
        text = text.replace("[environment]\n", f'[environment]\ndocker_image = "{image}"\n', 1)
        path.write_text(text)  # Local: task.toml is a copied file, never the upstream symlink.
    return directory


def harbor_command(prepared, api, name, root):
    """Global: one attempt with reusable images. Local: pinned Harbor's --no-delete uses compose down.

    Source: harbor/environments/docker/docker.py::stop and
    https://docs.docker.com/reference/cli/docker/compose/down/ (images require --rmi).
    """
    from pathlib import Path  # Local: support both CLI strings and workspace Paths.
    kwargs = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}, "seed": 2002}
    return [str(ROOT / ".venv/bin/harbor"), "run", "-p", str(prepared), "-a", "terminus-2", "-m", "openai/pilot",
            "--ak", f"api_base={api}", "--ak", "max_turns=16", "--ak", "enable_summarize=false",
            "--ak", "llm_call_kwargs=" + json.dumps(kwargs), "-n", "1", "-k", "1", "--no-delete",
            "--job-name", name, "--jobs-dir", str(Path(root) / "evaluation/jobs")]


def evaluate(root, *, modes=None, smoke=False):
    """Global: fixed coverage and paired tasks. Local: run real Docker episodes through Harbor."""
    chosen = read_json(root / "evaluation/tasks.json")
    tasks = [chosen["smoke_task"]] if smoke else chosen["tasks"]
    modes = [("smoke", "retain")] if smoke else (modes or MODES)
    index_path = root / "evaluation/index.json"
    index = read_json(index_path) if index_path.exists() else {}
    for arm, mode in modes:
        conditions = [False] if smoke else [True, False]  # Local: resuming completed jobs needs no GPU server.
        if all(f"{arm}-{mode}-{'elicited' if condition else 'plain'}" in index for condition in conditions):
            continue
        if arm != "base" and not (root / "checkpoints" / arm / "progress.json").exists():
            event("evaluation_missing_model", arm=arm, mode=mode)
            continue
        if arm != "base" and not read_json(root / "checkpoints" / arm / "progress.json")["complete"]:
            event("evaluation_incomplete_training", arm=arm, mode=mode)
            continue
        end = deadline(root, start=True)
        if time.time() >= end:
            break
        ensure_task_images(root)  # Local: verify exact frozen IDs before loading a GPU model.
        model_path = merge_model(root, arm, mode)
        with model_server(model_path, root) as api:
            for elicited in conditions:
                name = f"{arm}-{mode}-{'elicited' if elicited else 'plain'}"
                if name in index:  # Local: never repeat failed episodes selectively.
                    continue
                if time.time() >= end:
                    break
                prepared = dataset(root, tasks, elicited)
                env = os.environ.copy()
                env["OPENAI_API_KEY"] = "local-placeholder"  # Local: no paid or external provider.
                command = harbor_command(prepared, api, name, root)  # Local: shared tested lifecycle options.
                index[name] = {"arm": arm, "mode": mode, "elicited": elicited, "tasks": tasks, "smoke": smoke,
                               "started": time.time(), "status": "running",
                               "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
                write_json(index_path, index)
                code = execute(command, env=env, deadline=end, check=False)
                index[name].update({"exit_code": code, "finished": time.time(), "status": "finished" if code == 0 else "incomplete"})
                write_json(index_path, index)
                event("harbor_finished", job=name, **index[name])
