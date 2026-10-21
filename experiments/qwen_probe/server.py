"""Global: one local GPU worker for Harbor. Local: a small text-only OpenAI protocol bridge.

Generation uses Transformers directly; no external API or model code is executed.
Sources: https://huggingface.co/docs/transformers/main_classes/text_generation
and https://fastapi.tiangolo.com/async/ . The bridge adds full research logging and cancellation.
"""
import asyncio  # Local: serialize GPU requests while keeping health/cancellation responsive.
import os  # Local: workspace port configuration.
import sys  # Local: model directory argument.
import threading  # Local: thread-safe cancellation between HTTP and generation.
import time  # Local: timestamps in the OpenAI-compatible response.
import uuid  # Local: request identifiers without experiment-arm metadata.
from pathlib import Path  # Local: per-request journals are workspace files, not external telemetry.

import torch  # Local: single-device generation.
import uvicorn  # Local: one ASGI worker bound to localhost only.
from fastapi import FastAPI, Request  # Local: Harbor-compatible HTTP endpoint.
from fastapi.responses import JSONResponse  # Local: standard API errors.
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, TextStreamer

from .runtime import event, setup, write_json


class Cancelled(StoppingCriteria):
    """Global: bound orphan requests. Local: stop generating when the caller disconnects."""

    def __init__(self, stopped):
        self.stopped = stopped  # Local: threading.Event shared with the async monitor.

    def __call__(self, input_ids, scores, **kwargs):
        return self.stopped.is_set()  # Local: checked after each generation step.


def serve(model_path, context_limit=16384):
    """Global: reproducible local inference. Local: locked generation, full text logs, no secrets."""
    setup()
    torch.set_num_threads(4)  # Local: avoid CPU oversubscription alongside Docker.
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                attn_implementation="sdpa", local_files_only=True).cuda().eval()
    app, lock = FastAPI(), asyncio.Lock()  # Local: exactly one in-flight GPU generation.

    @app.get("/health")
    async def health():
        """Global: avoid evaluating before startup. Local: expose readiness only after model load."""
        return {"ready": True, "instance": os.environ.get("GRAFT_SERVER_INSTANCE")}

    @app.get("/v1/models")
    async def models():
        """Global: local API discovery. Local: expose a neutral model name for blinded logs."""
        return {"object": "list", "data": [{"id": "pilot", "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def completion(request: Request):
        """Global: Harbor receives full raw model responses. Local: support its nonstreaming text calls."""
        body = await request.json()
        if body.get("stream"):
            return JSONResponse({"error": {"message": "This bounded research bridge uses nonstreaming API responses"}}, status_code=400)
        messages = body["messages"]  # Local: task/system/agent history from Harbor.
        for message in messages:
            if isinstance(message.get("content"), list):
                message["content"] = "".join(part.get("text", "") for part in message["content"])
        identity = uuid.uuid4().hex
        async with lock:  # Local: no generation RNG interleaving across episodes.
            journal = Path(os.environ["GRAFT_GENERATION_DIR"]) / f"{identity}.json"  # Local: owned by this server instance.
            record = {"started_at": time.time(), "messages": messages}  # Local: no request headers or credentials.
            write_json(journal, record)  # Global: preserve evidence even if the client times out before receiving it.
            stopped = threading.Event()
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", return_dict=True,
                                                    add_generation_prompt=True, enable_thinking=False).to("cuda")
            prompt_length = inputs["input_ids"].shape[1]
            event("generation_input", request_id=identity, messages=messages,
                  rendered_prompt=tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False), seed=2002)
            if prompt_length >= context_limit:
                return JSONResponse({"error": {"message": "maximum context length exceeded", "code": "context_length_exceeded", "type": "invalid_request_error"}}, status_code=400)
            async def watch_disconnect():
                """Global: agent timeout ends its GPU work. Local: poll without blocking the server."""
                while not stopped.is_set():
                    if await request.is_disconnected():
                        stopped.set()
                    await asyncio.sleep(0.2)
            def generate():
                """Global: fixed non-thinking sampling. Local: stream complete tokens to timestamped parent logs."""
                torch.manual_seed(2002)
                temperature = float(body.get("temperature", 0.7))
                with torch.inference_mode():
                    return model.generate(**inputs, max_new_tokens=context_limit - prompt_length,
                                          do_sample=temperature > 0, temperature=max(temperature, 0.001),
                                          top_p=0.8, top_k=20, max_time=110,
                                          stopping_criteria=[Cancelled(stopped)],
                                          streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False))
            watcher = asyncio.create_task(watch_disconnect())
            try:
                generated = await asyncio.to_thread(generate)
            finally:
                stopped.set()
                watcher.cancel()
            tokens = generated[0, prompt_length:]
            content = tokenizer.decode(tokens, skip_special_tokens=True)
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            finish = "stop" if int(tokens[-1]) in eos else "length"
            write_json(journal, record | {"text": content, "finish_reason": finish, "completed_at": time.time()})
            event("generation_output", request_id=identity, text=content, token_ids=tokens.tolist(), finish_reason=finish)
            return {"id": "chatcmpl-" + identity, "object": "chat.completion", "created": int(time.time()), "model": "pilot",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
                    "usage": {"prompt_tokens": prompt_length, "completion_tokens": len(tokens), "total_tokens": prompt_length + len(tokens)}}
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("GRAFT_SERVER_PORT", "8765")), workers=1)


if __name__ == "__main__":
    serve(sys.argv[1])  # Local: process exit fully unloads this deployment before the next.
