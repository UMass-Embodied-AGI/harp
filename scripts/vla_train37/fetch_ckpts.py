#!/usr/bin/env python3
"""Download zimplex/harp-vla-train37-* checkpoints, verify SHA256SUMS, and
build an eval view whose training-machine paths are remapped to HF ids.

Run it with an interpreter that has ``huggingface_hub`` (the LeRobot env):
``"$LEROBOT_PY" scripts/vla_train37/fetch_ckpts.py [--root DIR] [--models act,pi0]``.
The repos are public; no login is needed.

  checkpoints/train37/<model>/       pristine HF snapshot
  checkpoints/train37_eval/<model>/  symlinks + patched JSON (what servers load)
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = ["act", "diffusion", "pi0", "pi05", "smolvla"]
SMOLVLM = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
# Some checkpoint JSON configs still name absolute paths from the machine the
# models were trained on (under /mnt/lustre). Those paths do not exist
# elsewhere, so the eval view rewrites each one whose basename is a known
# public model to that model's Hugging Face id; other values are unchanged.
REMAP = {"smolvlm2_500m_video_instruct": SMOLVLM}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def remap(obj, hits):
    if isinstance(obj, dict):
        return {k: remap(v, hits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [remap(v, hits) for v in obj]
    if isinstance(obj, str) and obj.startswith("/mnt/lustre"):
        base = obj.rstrip("/").rsplit("/", 1)[-1]
        if base in REMAP:
            hits.append((obj, REMAP[base]))
            return REMAP[base]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="checkpoints")
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    root = Path(args.root)
    for m in args.models.split(","):
        src = root / "train37" / m
        snapshot_download(f"zimplex/harp-vla-train37-{m}", local_dir=src)
        bad = []
        for line in (src / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            name = name.lstrip("*")
            if sha256(src / name) != digest:
                bad.append(name)
        if bad:
            raise SystemExit(f"{m}: SHA256 mismatch {bad}")
        view = root / "train37_eval" / m
        view.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.name.startswith("."):
                continue
            dst = view / f.name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            if f.suffix == ".json":
                hits = []
                data = remap(json.loads(f.read_text()), hits)
                dst.write_text(json.dumps(data, indent=4))
                for old, new in hits:
                    print(f"[fetch] {m}/{f.name}: {old} -> {new}")
            else:
                os.symlink(f.resolve(), dst)
        print(f"[fetch] {m}: verified + view at {view}", flush=True)


if __name__ == "__main__":
    main()
