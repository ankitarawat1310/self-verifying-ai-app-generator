"""Register svaga-qwen30b-ft on the local Ollama server.

Ollama 0.34.1 and later reject a Modelfile ADAPTER line
("LoRA adapters are no longer supported"). The server still applies an adapter
that is already stored as a model layer: it passes the GGUF to llama-server
with --lora. This copies svaga-qwen30b-lora.gguf into Ollama's blob store and
writes a manifest that is the base model plus that layer.

Requires svaga-qwen30b-base (ollama create -f Modelfile.base).
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ADAPTER = HERE / "svaga-qwen30b-lora.gguf"
BASE_NAME = "svaga-qwen30b-base"
FT_NAME = "svaga-qwen30b-ft"
ADAPTER_MEDIA = "application/vnd.ollama.image.adapter"


def models_dir() -> Path:
    env = os.environ.get("OLLAMA_MODELS", "").strip()
    if env:
        return Path(env)
    return Path.home() / ".ollama" / "models"


def find_manifest(root: Path, name: str) -> Path | None:
    hits = []
    for tree in ("manifests-v2", "manifests"):
        base = root / tree
        if not base.is_dir():
            continue
        hits.extend(p for p in base.rglob(name) if p.is_dir())
    for folder in hits:
        latest = folder / "latest"
        if latest.is_file():
            return latest
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not ADAPTER.is_file():
        print(f"missing adapter: {ADAPTER}", file=sys.stderr)
        return 1
    root = models_dir()
    base_manifest = find_manifest(root, BASE_NAME)
    if base_manifest is None:
        print(
            f"{BASE_NAME} is not installed under {root}.\n"
            "Run: ollama create svaga-qwen30b-base -f Modelfile.base",
            file=sys.stderr,
        )
        return 1

    digest = sha256_file(ADAPTER)
    size = ADAPTER.stat().st_size
    blob = root / "blobs" / f"sha256-{digest}"
    blob.parent.mkdir(parents=True, exist_ok=True)
    if not blob.is_file() or blob.stat().st_size != size or sha256_file(blob) != digest:
        shutil.copyfile(ADAPTER, blob)

    mf = json.loads(base_manifest.read_text(encoding="utf-8"))
    layers = [layer for layer in mf.get("layers", []) if layer.get("mediaType") != ADAPTER_MEDIA]
    layers.append({"mediaType": ADAPTER_MEDIA, "digest": f"sha256:{digest}", "size": size})
    mf["layers"] = layers

    out = base_manifest.parent.parent / FT_NAME / "latest"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mf, separators=(",", ":")), encoding="utf-8")
    print(f"registered {FT_NAME}")
    print(f"adapter sha256:{digest} ({size} bytes)")
    print(f"manifest {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
