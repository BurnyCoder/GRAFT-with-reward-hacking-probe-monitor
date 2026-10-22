"""Global: frozen trajectory monitoring before SFT. Local: exact encoder/head and content cache.

Sources: pinned notebook cells 219/221/231; https://huggingface.co/docs/peft/package_reference/peft_model
"""
import sqlite3  # Local: reuse identical turn embeddings; global: avoid repeated GPU work.
import time  # Local: shared experiment deadline.

import numpy as np  # Local: lossless float32 cache blobs.
import torch  # Local: inference-only detector execution.
from peft import PeftModel  # Local: official frozen LoRA loading.
from transformers import AutoModel, AutoTokenizer  # Local: matching embedding model/tokenizer.

from ..download import MANIFEST
from ..monitoring import validate_scores
from ..runtime import deadline, event, file_hash, fingerprint, read_json
from .model import TurnTransformerProbe, last_token_pool
from .preprocess import VERSION


class FrozenDetector:
    """Global: no detector fitting. Local: batch turn encoding and released classifier inference."""

    def __init__(self, root):
        """Global: reject incompatible artifacts. Local: freeze and strictly load all parameters."""
        self.root = root
        self.identity = fingerprint({"model": MANIFEST["encoder"], "adapter": MANIFEST["files"],
                                     "tokenizer": read_json(root / "downloads.json")["model_files"]["encoder"],
                                     "preprocessing": VERSION, "configuration": {"tokens": 4096, "turns": 248, "dtype": "bf16"}})
        self.tokenizer = AutoTokenizer.from_pretrained(root / "models/encoder", local_files_only=True)
        self.tokenizer.padding_side = "left"  # Local: notebook's pooling assumes final token is real.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModel.from_pretrained(root / "models/encoder", dtype=torch.bfloat16,
                                         attn_implementation="sdpa", local_files_only=True)
        self.encoder = PeftModel.from_pretrained(base, root / "detector/qwen3_lora_turn_source_invariant", is_trainable=False)
        self.encoder = self.encoder.cuda().eval().requires_grad_(False)
        head_path = root / "detector/turn_transformer_probe_ft.pth"
        if file_hash(head_path) != MANIFEST["files"][0]["sha256"]:
            raise ValueError("Probe checksum changed")
        self.head = TurnTransformerProbe().eval().requires_grad_(False)
        self.head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True), strict=True)
        self.head = self.head.cuda()  # Local: classifier retains released float32 precision.
        self.database = sqlite3.connect(root / "monitor/embeddings.sqlite")
        self.database.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL, tokens INTEGER NOT NULL)")
        event("detector_loaded", fingerprint=self.identity, trainable_parameters=sum(p.numel() for p in self.encoder.parameters() if p.requires_grad))

    @torch.inference_mode()
    def score(self, turns):
        """Global: measured coverage. Local: first 248 turns, first 4096 tokens per turn."""
        selected = turns[:248]
        vectors, lengths, missing = {}, {}, []
        for index, text in enumerate(selected):
            key = fingerprint({"detector": self.identity, "data": text})  # Local: data-dependent cache identity.
            cached = self.database.execute("SELECT vector, tokens FROM embeddings WHERE key=?", (key,)).fetchone()
            if cached:
                vectors[index], lengths[index] = np.frombuffer(cached[0], dtype=np.float32).copy(), cached[1]
            else:
                missing.append((index, text, key))
        for start in range(0, len(missing), 4):  # Local: four frozen-encoder turns fit the 8 GB device.
            end = deadline(self.root)
            if end and time.time() >= end:
                raise TimeoutError("Registered experiment deadline reached while scoring")
            batch = missing[start:start + 4]
            texts = [item[1] for item in batch]
            tokenized = self.tokenizer(texts, padding=True, truncation=True, max_length=4096, return_tensors="pt").to("cuda")
            output = self.encoder(**tokenized)
            embeddings = torch.nn.functional.normalize(last_token_pool(output.last_hidden_state, tokenized["attention_mask"]).float(), dim=-1)
            for (index, text, key), embedding in zip(batch, embeddings.cpu().numpy()):
                tokens = len(self.tokenizer(text, truncation=False)["input_ids"])
                vectors[index], lengths[index] = embedding, tokens
                self.database.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)", (key, embedding.tobytes(), tokens))
                event("monitor_turn_input", cache_key=key, text=text, tokens=tokens, retained_tokens=min(tokens, 4096))
            self.database.commit()  # Local: persist completed turns even if the trajectory later times out.
        embeddings = torch.from_numpy(np.stack([vectors[i] for i in range(len(selected))])).unsqueeze(0).cuda()
        logit = self.head(embeddings, torch.zeros(1, len(selected), dtype=torch.bool, device="cuda"))
        return {"score": torch.sigmoid(logit).item(), "logit": logit.item(), "turns": len(turns),
                "retained_turns": len(selected), "truncated_turns": sum(x > 4096 for x in lengths.values()),
                "tokens_before_turn_truncation": sum(lengths.values()),
                "retained_tokens": sum(min(x, 4096) for x in lengths.values()), "input_hash": fingerprint(turns)}

    def close(self):
        """Local: persist and release the embedding database before the phase process exits."""
        self.database.close()


def validate_monitor(root):
    """Global: calibration precedes heldout access. Local: cache flags before unloading the detector."""
    lock = read_json(root / "data/lock.json")
    manifest = {"dataset": lock["fingerprint"], "detector": MANIFEST, "preprocessing": VERSION}
    validate_scores(root, "monitor", manifest, FrozenDetector)  # Local: same existing manifest and threshold bytes.
