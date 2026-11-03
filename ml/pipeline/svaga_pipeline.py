"""SVAGA fine-tuning dataset pipeline: bronze (raw) -> silver (cleaned) -> gold (balanced, split) -> ml_ready exports.

Warehouse: one DuckDB file (warehouse/svaga.duckdb) with schemas bronze, silver, gold, ops.
Every step is a plain function taking the project root, so the same code runs from Airflow
(dags/svaga_dataset_dag.py) or from run_pipeline.py.

Sources (bronze):
  generate  CodeFeedback-Filtered-Instruction  (KS package, filtered from 156,526 raw rows)
  edit      CommitPackFT, Python               (KS package, filtered from 56,025 raw rows)
  repair    SWE-smith-py, rev 77cab905         (AR frozen package, 10,000 + 1,000 rows)
Held-out (never trained on): HumanEval, HumanEvalFix, MBPP+, and the 20 SVAGA benchmark prompts.
"""
from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import duckdb

SEED = 2026
MAX_EST_TOKENS = 4096          # exact Qwen token count where the source records it, else a conservative chars/3.2 estimate; exact per-model check runs in each training kit's preflight
VAL_PER_TYPE = 190
TEST_PER_TYPE = 190
SHINGLE = 13                   # words; a shared 13-word run with a held-out prompt counts as contamination
SYSTEM_PROMPT = {
    "generate": "You are SVAGA's code model. Task: GENERATE. Write correct, secure Python that satisfies the request.",
    "edit": "You are SVAGA's code model. Task: EDIT. Change the given code as requested and reply with a unified diff only.",
    "repair": "You are SVAGA's code model. Task: REPAIR. Fix the bug in the given code and reply with a unified diff only.",
}
SOURCES = {
    # name: (task_type, files by original split, license rule)
    "codefeedback": ("generate", {"train": "bronze/ks/codefeedback_train.jsonl", "validation": "bronze/ks/codefeedback_validation.jsonl"}),
    "commitpackft": ("edit", {"train": "bronze/ks/commitpackft_train.jsonl", "validation": "bronze/ks/commitpackft_validation.jsonl"}),
    "swesmith_py": ("repair", {"train": "bronze/ar/train.jsonl", "validation": "bronze/ar/validation.jsonl"}),
}
# Dataset-level licenses where rows carry none. Verify against each dataset card before submission.
DATASET_LICENSE = {"codefeedback": ("apache-2.0", "dataset_card_to_verify"), "swesmith_py": ("mit", "dataset_card_to_verify")}


def warehouse_path(root: Path) -> Path:
    """SVAGA_WAREHOUSE overrides the location (needed where the data folder forbids deleting DuckDB's WAL file)."""
    import os
    return Path(os.getenv("SVAGA_WAREHOUSE") or root / "warehouse" / "svaga.duckdb")


def con(root: Path) -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(str(warehouse_path(root)))
    for s in ("bronze", "silver", "gold", "ops"):
        c.execute(f"CREATE SCHEMA IF NOT EXISTS {s}")
    return c


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _turn(messages, role):
    return "\n\n".join(m.get("content", "") for m in messages if m.get("role") == role)


# ---------------------------------------------------------------- bronze
def ingest(root: Path) -> dict:
    """Load every source row unchanged (raw JSON kept), plus the held-out prompt list."""
    c = con(root)
    c.execute("CREATE OR REPLACE TABLE bronze.rows (source VARCHAR, task_type VARCHAR, orig_split VARCHAR, line_no INTEGER, raw_json VARCHAR, file_sha256 VARCHAR)")
    counts = {}
    for source, (task, files) in SOURCES.items():
        for split, rel in files.items():
            path = root / rel
            data = path.read_bytes()
            fsha = hashlib.sha256(data).hexdigest()
            rows = [(source, task, split, i, line, fsha) for i, line in enumerate(data.decode("utf-8").splitlines()) if line.strip()]
            c.executemany("INSERT INTO bronze.rows VALUES (?,?,?,?,?,?)", rows)
            counts[f"{source}/{split}"] = len(rows)
    held = [json.loads(l) for l in (root / "bronze/heldout/heldout_all.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    c.execute("CREATE OR REPLACE TABLE bronze.heldout (task_id VARCHAR, split VARCHAR, prompt VARCHAR)")
    c.executemany("INSERT INTO bronze.heldout VALUES (?,?,?)", [(h["task_id"], h.get("split"), h["prompt"]) for h in held])
    # upstream raw funnels recorded by the KS build (raw download -> KS package)
    c.execute("CREATE OR REPLACE TABLE bronze.upstream_funnel (source VARCHAR, step VARCHAR, rows BIGINT)")
    for source in ("codefeedback", "commitpackft"):
        rep = json.loads((root / f"bronze/ks/{source}_filtering_report.json").read_text())
        s = rep["sources"][source]
        c.execute("INSERT INTO bronze.upstream_funnel VALUES (?,?,?)", (source, "raw_rows_read", s["rows_read"]))
        for k, v in s.items():
            if k.startswith("dropped_"):
                c.execute("INSERT INTO bronze.upstream_funnel VALUES (?,?,?)", (source, k, -int(v)))
        c.execute("INSERT INTO bronze.upstream_funnel VALUES (?,?,?)", (source, "ks_package_rows", rep["final_rows"]))
    ar = json.loads((root / "bronze/ar/manifest.json").read_text())
    c.execute("INSERT INTO bronze.upstream_funnel VALUES ('swesmith_py','raw_rows_read',50908)")
    c.execute("INSERT INTO bronze.upstream_funnel VALUES ('swesmith_py','ar_package_rows',?)", (ar["counts"]["train"] + ar["counts"]["validation"],))
    counts["heldout"] = len(held)
    c.close()
    return counts


# ---------------------------------------------------------------- silver
def clean(root: Path) -> dict:
    """Normalize to one schema and apply quality filters. Every dropped row is logged in ops.filter_log."""
    c = con(root)
    rows = c.execute("SELECT source, task_type, orig_split, line_no, raw_json FROM bronze.rows ORDER BY source, orig_split, line_no").fetchall()
    held_shingles = set()
    for (p,) in c.execute("SELECT prompt FROM bronze.heldout").fetchall():
        w = _norm(p).split()
        held_shingles.update(" ".join(w[i:i + SHINGLE]) for i in range(max(1, len(w) - SHINGLE + 1)))
    lic_by_source = {}
    kept, log, seen = [], [], set()
    for source, task, split, line_no, raw in rows:
        rid = f"{source}:{split}:{line_no}"
        try:
            r = json.loads(raw)
            msgs = r["messages"]
            user, asst = _turn(msgs, "user").strip(), _turn(msgs, "assistant").strip()
        except Exception:
            log.append((rid, source, "schema", "unparsable_json_or_no_messages")); continue
        if not user or not asst:
            log.append((rid, source, "schema", "empty_user_or_assistant")); continue
        # task-specific validity
        if task == "generate":
            blocks = re.findall(r"```([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", asst, flags=re.S)
            py = [b for lang, b in blocks if lang.lower() in ("python", "py", "python3")]
            if not py:
                py = [b for lang, b in blocks if not lang] if blocks else [asst]
            try:
                for b in py:
                    ast.parse(b)
            except SyntaxError:
                log.append((rid, source, "validity", "python_does_not_parse")); continue
        else:
            if not (("---" in asst and "+++" in asst) or asst.startswith("diff --git")) or "@@" not in asst:
                log.append((rid, source, "validity", "target_not_unified_diff")); continue
        # license
        lic = r.get("license")
        basis = "row"
        if not lic or lic == "unknown":
            lic, basis = DATASET_LICENSE.get(source, ("unknown", "missing"))
        if lic == "unknown":
            log.append((rid, source, "license", "license_unknown")); continue
        # length
        exact = (r.get("tokens") or {}).get("total")   # AR's rows carry exact Qwen2.5-Coder token counts
        est = exact if exact else (len(user) + len(asst) + 200) / 3.2
        if est > MAX_EST_TOKENS:
            log.append((rid, source, "length", "over_4096_estimated_tokens")); continue
        # contamination against held-out benchmarks and the 20 SVAGA tasks
        w = _norm(user).split()
        if any(" ".join(w[i:i + SHINGLE]) in held_shingles for i in range(max(1, len(w) - SHINGLE + 1))):
            log.append((rid, source, "decontaminate", "shares_13_words_with_heldout")); continue
        # exact duplicate (same instruction and answer, any source)
        key = _sha(_norm(user) + "\x00" + _norm(asst))
        if key in seen:
            log.append((rid, source, "dedupe", "exact_duplicate")); continue
        seen.add(key)
        if source == "swesmith_py":
            src_id, group = r["instance_id"], r["repo"]
            extra = {k: r.get(k) for k in ("source_revision", "image_name", "locally_replayed", "independently_verified", "training_approved", "context_semantically_complete")}
        else:
            src_id, group = r.get("source_id"), f"{source}:domain{r.get('domain_level')}"
            extra = {"domain_level": r.get("domain_level"), "ks_sample_id": r.get("sample_id")}
        kept.append((rid, source, task, split, src_id, group, lic, basis, user, asst, int(est), key, json.dumps(extra)))
    c.execute("""CREATE OR REPLACE TABLE silver.examples (row_id VARCHAR, source_dataset VARCHAR, task_type VARCHAR, orig_split VARCHAR,
                 source_id VARCHAR, group_id VARCHAR, license VARCHAR, license_basis VARCHAR, instruction VARCHAR, target VARCHAR,
                 est_tokens INTEGER, content_sha256 VARCHAR, provenance_json VARCHAR)""")
    c.executemany("INSERT INTO silver.examples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", kept)
    c.execute("CREATE OR REPLACE TABLE ops.filter_log (row_id VARCHAR, source VARCHAR, step VARCHAR, reason VARCHAR)")
    if log:
        c.executemany("INSERT INTO ops.filter_log VALUES (?,?,?,?)", log)
    out = {"kept": len(kept), "dropped": len(log), "dropped_by_reason": dict(Counter(f"{s}:{x}" for _, s, _, x in log))}
    c.close()
    return out


# ---------------------------------------------------------------- gold
def build_gold(root: Path) -> dict:
    """Balanced, leak-free train/val/test. Rows not selected stay in silver, marked in gold.unused."""
    c = con(root)
    ex = c.execute("SELECT row_id, source_dataset, task_type, orig_split, group_id FROM silver.examples ORDER BY row_id").fetchall()
    rng = random.Random(SEED)
    by = defaultdict(list)
    for row_id, source, task, split, group in ex:
        by[(task, split)].append((row_id, group))
    assign = {}

    def take(pool, n):
        pool = pool[:]
        rng.shuffle(pool)
        return pool[:n], pool[n:]

    def round_robin(pool, n):
        groups = defaultdict(list)
        for item in pool:
            groups[item[1]].append(item)
        for g in groups.values():
            rng.shuffle(g)
        order = sorted(groups)
        rng.shuffle(order)
        picked = []
        while len(picked) < n and any(groups.values()):
            for g in order:
                if groups[g] and len(picked) < n:
                    picked.append(groups[g].pop())
        rest = [i for g in groups.values() for i in g]
        return picked, rest

    pools = {}
    # generate: test carved from the KS train pool; val from the KS validation file
    test_g, train_g = take(by[("generate", "train")], TEST_PER_TYPE)
    val_g, _ = take(by[("generate", "validation")], VAL_PER_TYPE)
    pools["generate"] = (train_g, val_g, test_g)
    # edit: val and test both from the KS validation file (disjoint rows)
    val_e, rest_e = take(by[("edit", "validation")], VAL_PER_TYPE)
    test_e, _ = take(rest_e, TEST_PER_TYPE)
    pools["edit"] = (by[("edit", "train")], val_e, test_e)
    # repair: AR validation repos are disjoint from train repos; split those repos into val repos and test repos
    rv = by[("repair", "validation")]
    repos = sorted({g for _, g in rv})
    rng.shuffle(repos)
    val_repos, test_repos, nv, nt = set(), set(), 0, 0
    cnt = Counter(g for _, g in rv)
    for g in repos:
        if nv <= nt:
            val_repos.add(g); nv += cnt[g]
        else:
            test_repos.add(g); nt += cnt[g]
    val_r, _ = round_robin([i for i in rv if i[1] in val_repos], VAL_PER_TYPE)
    test_r, _ = round_robin([i for i in rv if i[1] in test_repos], TEST_PER_TYPE)
    pools["repair"] = (by[("repair", "train")], val_r, test_r)
    n_train = min(len(p[0]) for p in pools.values())
    for task, (train_pool, val, test) in pools.items():
        picked, _ = round_robin(train_pool, n_train)
        for row_id, _ in picked:
            assign[row_id] = "train"
        for row_id, _ in val:
            assign[row_id] = "val"
        for row_id, _ in test:
            assign[row_id] = "test"
    c.execute("CREATE OR REPLACE TABLE gold.split_assignment (row_id VARCHAR, split VARCHAR)")
    c.executemany("INSERT INTO gold.split_assignment VALUES (?,?)", list(assign.items()))
    c.execute("""CREATE OR REPLACE TABLE gold.examples AS
        SELECT printf('svaga_%06d', row_number() OVER (ORDER BY a.split, s.task_type, s.row_id)) AS sample_id,
               s.task_type, s.source_dataset, s.source_id, s.group_id, s.license, s.license_basis, a.split,
               s.instruction, s.target, s.est_tokens, s.content_sha256, s.provenance_json, s.row_id
        FROM silver.examples s JOIN gold.split_assignment a USING (row_id)""")
    c.execute("""CREATE OR REPLACE TABLE gold.unused AS SELECT s.row_id, s.source_dataset, s.task_type, 'unused_balance' AS reason
                 FROM silver.examples s ANTI JOIN gold.split_assignment a USING (row_id)""")
    out = {"train_per_type": n_train, "split_counts": c.execute("SELECT split, task_type, count(*) FROM gold.examples GROUP BY ALL ORDER BY 1,2").fetchall()}
    c.close()
    return out


# ---------------------------------------------------------------- quality gates
def quality_checks(root: Path) -> dict:
    c = con(root)
    q = {}
    q["no_row_in_two_splits"] = c.execute("SELECT count(*) FROM (SELECT row_id FROM gold.split_assignment GROUP BY row_id HAVING count(*)>1)").fetchone()[0] == 0
    q["no_content_in_two_splits"] = c.execute("SELECT count(*) FROM (SELECT content_sha256 FROM gold.examples GROUP BY content_sha256 HAVING count(DISTINCT split)>1)").fetchone()[0] == 0
    q["repair_repos_disjoint_across_splits"] = c.execute("""SELECT count(*) FROM (SELECT group_id FROM gold.examples WHERE task_type='repair'
                                                            GROUP BY group_id HAVING count(DISTINCT split)>1)""").fetchone()[0] == 0
    q["no_unknown_license"] = c.execute("SELECT count(*) FROM gold.examples WHERE license='unknown'").fetchone()[0] == 0
    q["no_empty_target"] = c.execute("SELECT count(*) FROM gold.examples WHERE length(trim(target))=0").fetchone()[0] == 0
    bal = c.execute("SELECT count(DISTINCT n) FROM (SELECT task_type, count(*) n FROM gold.examples WHERE split='train' GROUP BY task_type)").fetchone()[0]
    q["train_balanced_by_rows"] = bal == 1
    c.close()
    failed = [k for k, v in q.items() if not v]
    if failed:
        raise AssertionError(f"quality checks failed: {failed}")
    return q


# ---------------------------------------------------------------- ml_ready exports
def export(root: Path) -> dict:
    c = con(root)
    out_dir = root / "gold"
    out_dir.mkdir(exist_ok=True)
    manifest = {"seed": SEED, "files": {}, "schema": ["sample_id", "task_type", "source_dataset", "source_id", "license", "split", "messages"]}
    for split in ("train", "val", "test"):
        recs = c.execute("""SELECT sample_id, task_type, source_dataset, source_id, license, split, instruction, target
                            FROM gold.examples WHERE split=? ORDER BY sample_id""", [split]).fetchall()
        rng = random.Random(f"{SEED}-{split}")
        rng.shuffle(recs)  # interleave task types so every training batch sees all three jobs
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for sid, task, src, srcid, lic, sp, user, asst in recs:
                f.write(json.dumps({"sample_id": sid, "task_type": task, "source_dataset": src, "source_id": srcid, "license": lic, "split": sp,
                                    "messages": [{"role": "system", "content": SYSTEM_PROMPT[task]},
                                                 {"role": "user", "content": user},
                                                 {"role": "assistant", "content": asst}]}, ensure_ascii=False) + "\n")
        manifest["files"][f"{split}.jsonl"] = {"rows": len(recs), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    meta = c.execute("SELECT sample_id, task_type, source_dataset, source_id, group_id, license, license_basis, split, est_tokens, provenance_json FROM gold.examples ORDER BY sample_id").fetchall()
    with (out_dir / "gold_metadata.csv").open("w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["sample_id", "task_type", "source_dataset", "source_id", "group_id", "license", "license_basis", "split", "est_tokens", "provenance_json"])
        w.writerows(meta)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    c.close()
    return manifest


# ---------------------------------------------------------------- statistics and charts
def report(root: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = con(root)
    rep = root / "reports"
    ch = rep / "charts"
    ch.mkdir(parents=True, exist_ok=True)
    colors = {"generate": "#2a78d6", "edit": "#e08a1e", "repair": "#2e9e6a"}
    names = {"codefeedback": "generate", "commitpackft": "edit", "swesmith_py": "repair"}
    stats = {}
    # 1. funnel per source: raw -> upstream package -> silver -> gold train/val/test
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, source in zip(axes, ("codefeedback", "commitpackft", "swesmith_py")):
        raw = c.execute("SELECT rows FROM bronze.upstream_funnel WHERE source=? AND step='raw_rows_read'", [source]).fetchone()[0]
        pkg = c.execute("SELECT count(*) FROM bronze.rows WHERE source=?", [source]).fetchone()[0]
        sil = c.execute("SELECT count(*) FROM silver.examples WHERE source_dataset=?", [source]).fetchone()[0]
        gold = c.execute("SELECT count(*) FROM gold.examples WHERE source_dataset=?", [source]).fetchone()[0]
        steps = [("raw", raw), ("filtered package", pkg), ("silver (cleaned)", sil), ("gold (balanced)", gold)]
        stats[source] = dict(steps)
        ax.barh([s for s, _ in steps][::-1], [v for _, v in steps][::-1], color=colors[names[source]])
        for i, (_, v) in enumerate(steps[::-1]):
            ax.text(v, i, f" {v:,}", va="center", fontsize=9)
        ax.set_title(f"{source} ({names[source]})", fontsize=11)
        ax.set_xscale("log"); ax.set_xlim(right=raw * 6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Rows kept at each pipeline stage (log scale)", fontsize=12)
    fig.tight_layout(); fig.savefig(ch / "1_funnel.png", dpi=150); plt.close(fig)
    # 2. silver drop reasons
    drops = c.execute("SELECT source, step || ': ' || reason AS r, count(*) n FROM ops.filter_log GROUP BY ALL ORDER BY n DESC").fetchall()
    stats["silver_drops"] = [list(d) for d in drops]
    if drops:
        fig, ax = plt.subplots(figsize=(9, 0.5 + 0.45 * len(drops)))
        ax.barh([f"{s} | {r}" for s, r, _ in drops][::-1], [n for _, _, n in drops][::-1], color="#8a94a6")
        for i, (_, _, n) in enumerate(drops[::-1]):
            ax.text(n, i, f" {n:,}", va="center", fontsize=9)
        ax.set_title("Rows removed in cleaning, by reason"); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(ch / "2_drop_reasons.png", dpi=150); plt.close(fig)
    # 3. gold rows and token share per split and task
    tab = c.execute("SELECT split, task_type, count(*) n, sum(est_tokens) t FROM gold.examples GROUP BY ALL").fetchall()
    stats["gold"] = [list(x) for x in tab]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    splits = ["train", "val", "test"]
    for ax, idx, title in ((axes[0], 2, "Rows per split"), (axes[1], 3, "Estimated tokens per split")):
        bottom = [0, 0, 0]
        for task in ("generate", "edit", "repair"):
            vals = [next((x[idx] for x in tab if x[0] == s and x[1] == task), 0) for s in splits]
            ax.bar(splits, vals, bottom=bottom, color=colors[task], label=task)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_title(title); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(ch / "3_gold_balance.png", dpi=150); plt.close(fig)
    # 4. length distribution per task (train)
    fig, ax = plt.subplots(figsize=(8, 4))
    for task in ("generate", "edit", "repair"):
        v = [x[0] for x in c.execute("SELECT est_tokens FROM gold.examples WHERE split='train' AND task_type=?", [task]).fetchall()]
        ax.hist(v, bins=40, alpha=0.6, color=colors[task], label=f"{task} (median {sorted(v)[len(v)//2]:,})")
    ax.set_xlabel("estimated tokens per example"); ax.set_ylabel("examples"); ax.legend(frameon=False)
    ax.set_title("Example length, training split"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(ch / "4_length_distribution.png", dpi=150); plt.close(fig)
    # 5. repository / domain diversity
    div = c.execute("SELECT task_type, count(DISTINCT group_id) FROM gold.examples WHERE split='train' GROUP BY 1").fetchall()
    stats["train_group_diversity"] = dict(div)
    (rep / "dataset_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    c.close()
    return stats
