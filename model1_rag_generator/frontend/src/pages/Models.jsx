import { useEffect, useState } from "react";
import { fetchModelReport, modelChartUrl } from "../api.js";

const num = (v, d = 2) => (v === undefined || v === null || v === "" || Number.isNaN(Number(v)) ? "-" : Number(v).toFixed(d));
const pct = (v) => (v === undefined || v === null || v === "" ? "-" : `${Math.round(Number(v) * 100)}%`);

function Card({ title, children, note }) {
  return (
    <section className="space-y-2">
      <h3 className="font-semibold">{title}</h3>
      {note && <p className="text-xs text-slate-400">{note}</p>}
      {children}
    </section>
  );
}

function Charts({ section, files }) {
  if (!files?.length) return null;
  return (
    <div className="grid md:grid-cols-2 gap-3">
      {files.map((f) => (
        <a key={f} href={modelChartUrl(section, f)} target="_blank" rel="noreferrer" title="Open full size">
          <img src={modelChartUrl(section, f)} alt={f} className="w-full rounded border border-slate-800 bg-white" />
        </a>
      ))}
    </div>
  );
}

export default function Models() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    fetchModelReport().then(setData).catch((e) => setError(String(e.message || e)));
  }, []);
  if (error) return <p className="p-4 text-red-400">{error}</p>;
  if (!data) return <p className="p-4 text-slate-400">Loading model results…</p>;

  const rows = data.comparison || [];
  const gate = data.gate;
  const ds = data.dataset_manifest?.files;
  const gateRows = gate
    ? [
        ["Rule gate (all 6 checks pass)", gate.rule_gate, null],
        ...["logistic_regression", "random_forest", "gradient_boosting"].flatMap((m) =>
          gate[m]
            ? [
                [`${m.replace(/_/g, " ")}, same # shipped`, gate[m].same_accept_count_as_rule, gate[m].same_accept_count_as_rule.roc_auc],
                [`${m.replace(/_/g, " ")}, threshold 0.5`, gate[m]["threshold_0.5"], gate[m]["threshold_0.5"].roc_auc],
              ]
            : []
        ),
      ]
    : [];

  return (
    <div className="max-w-6xl mx-auto p-4 space-y-8">
      <div>
        <h2 className="text-xl font-bold">Models</h2>
        <p className="text-sm text-slate-400">
          Four open code models fine-tuned on the SVAGA dataset (generate, edit, repair) with identical QLoRA settings, plus a learned
          release gate trained on the Demo 1 runs. All numbers are on data the models never trained on.
        </p>
      </div>

      <Card
        title="Training data"
        note="Bronze → silver → gold pipeline (DuckDB warehouse, Airflow DAG): balanced by task, decontaminated against HumanEval, MBPP+, HumanEvalFix and the 20 SVAGA tasks."
      >
        {ds && (
          <p className="text-sm">
            Train {Number(ds["train.jsonl"]?.rows).toLocaleString()} · validation {ds["val.jsonl"]?.rows} · untouched test {ds["test.jsonl"]?.rows} rows
          </p>
        )}
        <Charts section="dataset" files={data.charts?.dataset} />
      </Card>

      <Card
        title="Fine-tuned models: base vs fine-tuned on the untouched test split"
        note="Loss is only comparable within one model (each uses its own tokenizer): compare the reduction. Generation metrics use 30 test examples per task."
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm border border-slate-800">
            <thead>
              <tr className="bg-slate-900 text-left">
                <th className="p-2">Model</th>
                <th className="p-2">GPU</th>
                <th className="p-2 text-right">Train rows</th>
                <th className="p-2 text-right">Minutes</th>
                <th className="p-2 text-right">Test loss</th>
                <th className="p-2 text-right">Reduction</th>
                <th className="p-2 text-right">Repair F1</th>
                <th className="p-2 text-right">Repair exact</th>
                <th className="p-2 text-right">Edit valid diff</th>
                <th className="p-2 text-right">Generate parses</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.run} className="border-t border-slate-800">
                  <td className="p-2">
                    {r.model}
                    <div className="text-xs text-slate-500">{r.run}</div>
                  </td>
                  <td className="p-2 text-xs">{String(r.gpu).replace("NVIDIA ", "")}</td>
                  <td className="p-2 text-right tabular-nums">{Number(r.train_rows).toLocaleString()}</td>
                  <td className="p-2 text-right tabular-nums">{num(r.train_minutes, 0)}</td>
                  <td className="p-2 text-right tabular-nums">
                    {num(r.test_loss_base, 3)} → {num(r.test_loss_finetuned, 3)}
                  </td>
                  <td className="p-2 text-right tabular-nums font-semibold">−{num(r.test_loss_reduction_pct, 0)}%</td>
                  <td className="p-2 text-right tabular-nums">
                    {num(r.repair_changed_line_f1_base)} → {num(r.repair_changed_line_f1_fine_tuned)}
                  </td>
                  <td className="p-2 text-right tabular-nums">
                    {pct(r.repair_exact_match_base)} → {pct(r.repair_exact_match_fine_tuned)}
                  </td>
                  <td className="p-2 text-right tabular-nums">
                    {pct(r.edit_valid_diff_base)} → {pct(r.edit_valid_diff_fine_tuned)}
                  </td>
                  <td className="p-2 text-right tabular-nums">
                    {pct(r.generate_python_parses_base)} → {pct(r.generate_python_parses_fine_tuned)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-slate-500">{rows.length} of 4 models have complete results.</p>
        <Charts section="models" files={data.charts?.models} />
      </Card>

      {data.public_benchmarks?.length > 0 && (
        <Card
          title="Public benchmarks (pass@1, executed tests)"
          note="General Python coding, not SVAGA apps. Five models on one shared dataset. Base is the better of the base model's prompts; McNemar compares the same problems, highlighted when p < 0.05."
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm border border-slate-800">
              <thead>
                <tr className="bg-slate-900 text-left">
                  <th className="p-2">Model</th>
                  <th className="p-2">Benchmark</th>
                  <th className="p-2 text-right">Problems</th>
                  <th className="p-2 text-right">Base</th>
                  <th className="p-2 text-right">Fine-tuned</th>
                  <th className="p-2 text-right">Change (pts)</th>
                  <th className="p-2 text-right">McNemar p</th>
                </tr>
              </thead>
              <tbody>
                {data.public_benchmarks.map((r) => (
                  <tr key={r.run + r.benchmark} className="border-t border-slate-800">
                    <td className="p-2">{r.model}</td>
                    <td className="p-2">{r.benchmark}</td>
                    <td className="p-2 text-right tabular-nums">{r.n}</td>
                    <td className="p-2 text-right tabular-nums">
                      {r.best_base_pct ?? r.base_svaga_prompt_pct}%{r.best_base === 'base_plain' ? ' (plain prompt)' : ''}
                    </td>
                    <td className="p-2 text-right tabular-nums font-semibold">{r.fine_tuned_pct}%</td>
                    <td className="p-2 text-right tabular-nums">{r.change_pts ?? ''}</td>
                    <td className={`p-2 text-right tabular-nums ${String(r.significant) === 'True' ? 'text-amber-400 font-semibold' : ''}`}>
                      {Number(r.mcnemar_p) < 0.0001 ? '<0.0001' : Number(r.mcnemar_p).toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {data.m4_repair?.length > 0 && (
        <Card
          title="M4: fine-tuned models as SVAGA's repairer (28 apps M1 rejected)"
          note="Negative result at 2B and 30B. Fine-tuning taught the diff format but not where to fix: no model fixed a broken app, and the fine-tuned models' scattered edits lowered hidden-check scores more than the bases'. The gate rejected every repaired candidate, so no damaged app was shipped."
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm border border-slate-800">
              <thead>
                <tr className="bg-slate-900 text-left">
                  <th className="p-2">Repair model</th>
                  <th className="p-2 text-right">Valid diff</th>
                  <th className="p-2 text-right">Applies fully</th>
                  <th className="p-2 text-right">Gate accepts</th>
                  <th className="p-2 text-right">Broken apps fixed</th>
                  <th className="p-2 text-right">Hidden checks, before → after</th>
                  <th className="p-2 text-right">Damaged apps shipped</th>
                </tr>
              </thead>
              <tbody>
                {data.m4_repair.map((r) => (
                  <tr key={r.condition} className="border-t border-slate-800">
                    <td className="p-2">{r.repair_model}</td>
                    <td className="p-2 text-right tabular-nums">{r.valid_diff_pct}%</td>
                    <td className="p-2 text-right tabular-nums">{r.diff_applied_fully_pct}%</td>
                    <td className="p-2 text-right tabular-nums">{r.gate_accept_after_pct}%</td>
                    <td className="p-2 text-right tabular-nums">{r.broken_apps_fixed}</td>
                    <td className="p-2 text-right tabular-nums">{r.hidden_checks_before_pct}% → {r.hidden_checks_candidate_after_pct}%</td>
                    <td className="p-2 text-right tabular-nums font-semibold">{r.shipped_false_assurance}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {gate && (
        <Card
          title="Learned release gate (model 5)"
          note={`Predicts whether an app is actually correct from signals available at decision time. ${gate.n_runs} Demo 1 runs, ${gate.n_correct} correct; leave-one-task-out cross-validation. Best model: ${gate.best_model?.replace(/_/g, " ")}, ROC-AUC 95% interval ${gate.best_auc_95ci_task_bootstrap?.join("–")}.`}
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm border border-slate-800">
              <thead>
                <tr className="bg-slate-900 text-left">
                  <th className="p-2">Gate</th>
                  <th className="p-2 text-right">Shipped</th>
                  <th className="p-2 text-right">Correct shipped</th>
                  <th className="p-2 text-right">Broken shipped</th>
                  <th className="p-2 text-right">False assurance</th>
                  <th className="p-2 text-right">Precision</th>
                  <th className="p-2 text-right">Recall</th>
                  <th className="p-2 text-right">F1</th>
                  <th className="p-2 text-right">ROC-AUC</th>
                </tr>
              </thead>
              <tbody>
                {gateRows.map(([name, s, auc]) => (
                  <tr key={name} className="border-t border-slate-800">
                    <td className="p-2">{name}</td>
                    <td className="p-2 text-right tabular-nums">{s.accepted}</td>
                    <td className="p-2 text-right tabular-nums">{s.tp_correct_shipped}</td>
                    <td className="p-2 text-right tabular-nums">{s.fp_broken_shipped}</td>
                    <td className="p-2 text-right tabular-nums">{pct(s.false_assurance_of_accepted)}</td>
                    <td className="p-2 text-right tabular-nums">{num(s.precision)}</td>
                    <td className="p-2 text-right tabular-nums">{num(s.recall)}</td>
                    <td className="p-2 text-right tabular-nums">{num(s.f1)}</td>
                    <td className="p-2 text-right tabular-nums">{auc == null ? "-" : num(auc)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Charts section="gate" files={data.charts?.gate} />
        </Card>
      )}
    </div>
  );
}
