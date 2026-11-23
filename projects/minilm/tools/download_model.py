#!/usr/bin/env python3
"""download_model.py - fetch the original MiniLM-L6 files from Hugging Face (about 90 MB)."""
from pathlib import Path

from huggingface_hub import snapshot_download

dest = Path(__file__).resolve().parent.parent / "model_build" / "hf"
snapshot_download(
    "sentence-transformers/all-MiniLM-L6-v2",
    allow_patterns=["model.safetensors", "vocab.txt", "config.json"],
    local_dir=str(dest),
)
print("downloaded to", dest)
