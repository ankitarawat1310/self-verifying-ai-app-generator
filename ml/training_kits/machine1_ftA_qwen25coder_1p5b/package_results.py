"""Zip everything needed to plug the model back into SVAGA and to report results. Send the zip back to the team.
Usage: python package_results.py --config config.yaml   ->  RETURN_<run_name>.zip next to this kit"""
import argparse, hashlib, json, zipfile
from pathlib import Path
import yaml
ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); a = ap.parse_args()
cfg = yaml.safe_load(open(a.config)); kit = Path(a.config).resolve().parent; out = kit / "outputs"
need = [out / "final_adapter" / "adapter_model.safetensors", out / "final_adapter" / "adapter_config.json", out / "metrics" / "train_summary.json"]
missing = [str(p) for p in need if not p.exists()]
if missing: raise SystemExit(f"missing: {missing}")
files = [p for d in ("final_adapter", "tokenizer", "metrics", "samples") for p in (out / d).rglob("*") if p.is_file()]
files += [p for p in (out / "preflight.json", out / "environment.txt", kit / "config.yaml", kit / "data" / "manifest.json") if p.exists()]
man = {"run_name": cfg["run_name"], "model_id": cfg["model_id"], "revision": cfg.get("revision"),
       "files": {str(p.relative_to(kit)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
zp = kit.parent / f"RETURN_{cfg['run_name']}.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files: z.write(p, f"{cfg['run_name']}/{p.relative_to(kit)}")
    z.writestr(f"{cfg['run_name']}/RETURN_MANIFEST.json", json.dumps(man, indent=2))
print(f"wrote {zp}  ({zp.stat().st_size/2**20:.1f} MB). Checkpoints are not included; keep outputs/checkpoints until results are accepted.")
