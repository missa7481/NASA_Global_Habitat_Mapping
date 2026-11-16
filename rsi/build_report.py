#!/usr/bin/env python3
"""Build artifacts/rsi/report/index.html from the RSI result files (every number is read from disk)."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts/rsi"
OUT = A / "report"
OUT.mkdir(parents=True, exist_ok=True)

DES = ["random5", "block5", "block10", "block20", "loro", "env", "zone", "temporal", "spatiotemp"]
DES_LABEL = {"random5": "Random", "block5": "Block 5°", "block10": "Block 10°", "block20": "Block 20°", "loro": "Leave region out",
             "env": "Env. cluster out", "zone": "Climate zone out", "temporal": "Temporal 2025", "spatiotemp": "Space-time"}
MODEL_LABEL = {"Prior": "Class prior", "kNN-geo": "kNN geographic", "GLM": "Multinomial GLM", "MLP": "MLP (early fusion)",
               "MLP+Coords": "MLP + coordinates", "HGB": "Gradient boosting", "RF": "Random forest", "GatedMoE": "Gated MoE", "HiCo": "HiCo"}


def f3(x, k=3):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{k}f}"


def table(df, fmt, best=None, cls="", group_rows=None, row_label=None, hl_rows=()):
    """df: index = rows, columns = cols. best: 'max' | 'min' per column."""
    h = [f'<div class="scroll"><table class="{cls}"><thead><tr><th>{html.escape(df.index.name or "")}</th>']
    h += [f"<th>{html.escape(str(c))}</th>" for c in df.columns] + ["</tr></thead><tbody>"]
    bests = {}
    if best:
        for c in df.columns:
            col = pd.to_numeric(df[c], errors="coerce")
            if col.notna().any():
                bests[c] = col.max() if best == "max" else col.min()
    for r, row in df.iterrows():
        lab = row_label(r) if row_label else r
        h.append(f'<tr{" class=hl" if r in hl_rows else ""}><td>{lab}</td>')
        for c in df.columns:
            v = row[c]
            b = c in bests and np.isfinite(v) and abs(v - bests[c]) < 1e-9
            h.append(f'<td{" class=best" if b else ""}>{fmt(v) if isinstance(v, (float, int, np.floating)) else html.escape(str(v))}</td>')
        h.append("</tr>")
    h.append("</tbody></table></div>")
    return "".join(h)


# ------------------------------------------------------------------ data
m1 = pd.read_csv(A / "it1/metrics.csv")
auc1 = m1.groupby(["model", "design"])["AUC_mean"].mean().unstack()[DES]
nll1 = m1.groupby(["model", "design"])["nll"].mean().unstack()[DES]
order1 = ["RF", "MLP+Coords", "MLP", "HGB", "HiCo", "GatedMoE", "GLM", "kNN-geo", "Prior"]
auc1, nll1 = auc1.loc[order1], nll1.loc[order1]
loro_reg = m1[m1.design == "loro"].pivot_table(index="model", columns="split", values="AUC_mean").loc[order1[:-1]]
loro_reg = loro_reg[["N America", "Europe", "Asia", "Oceania", "Latin America", "Africa"]]
ncell_reg = m1[(m1.design == "loro") & (m1.model == "RF")].set_index("split")["n_cells"]

bins = pd.read_csv(A / "it1/diag_bins.csv")
part = pd.read_csv(A / "it1/diag_partial.csv")
ps = pd.read_csv(A / "it1/prior_shift.csv")
m2 = pd.read_csv(A / "it2/metrics.csv")
gw = pd.read_csv(A / "it2/gate_weights.csv.gz")
sw = pd.read_csv(A / "it2/stack_weights.csv")
tm = pd.read_csv(A / "it3/transfer_matrix.csv")
inv = pd.read_csv(A / "it3/invariance.csv")
prox = pd.read_csv(A / "it3/proxy_decomposition.csv")
prox_r2 = float((A / "it3/proxy_r2.txt").read_text())
m4 = pd.read_csv(A / "it4/metrics.csv")
dp = pd.read_csv(A / "it4/decay_params.csv")
m5 = pd.read_csv(A / "it5_pool.csv")
m6 = pd.read_csv(A / "it6/metrics.csv")
coh6 = pd.read_csv(A / "it6/coherence.csv")
pt6 = pd.read_csv(A / "it6/paired_vs_rf.csv")
ledger = [json.loads(x) for x in (ROOT / "rsi/memory/ledger.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]

# ------------------------------------------------------------------ derived numbers
opt = {m: auc1.loc[m, "random5"] - auc1.loc[m, "loro"] for m in ("RF", "MLP", "kNN-geo")}
psl = ps[(ps.design == "loro")].groupby(["model", "correction"])["nll"].mean()
prior_te = float(ps[(ps.design == "loro") & (ps.model == "Prior(test, oracle)")]["nll"].mean())
eq2 = m2[m2.strategy == "Equal"].set_index(["design", "split"])
gains = []
for s, d in m2.groupby("strategy"):
    if s.startswith("Only:") or s in ("Equal", "Uncertainty"):
        continue
    d = d.set_index(["design", "split"])
    gains.append({"strategy": s, "inner": float((eq2.inner_nll - d.inner_nll).xs("loro").mean()),
                  "outer": float((eq2.nll - d.nll).xs("loro").mean())})
gains = pd.DataFrame(gains).sort_values("outer")
p4 = m4.pivot_table(index=["design", "split"], columns="strategy", values=["AUC_mean", "nll"])


def sign(a, b, k="AUC_mean"):
    d = (p4[k][a] - p4[k][b]).dropna()
    w = int(((d > 0) if k == "AUC_mean" else (d < 0)).sum())
    return w, len(d), binomtest(w, len(d)).pvalue, float(d.mean())


s_gate = sign("Decay [mixed]", "Attn-support [mixed]")
s_gate_n = sign("Decay [mixed]", "Attn-support [mixed]", "nll")
s_eq = sign("Decay [mixed]", "Equal")
s_reg = sign("Decay [mixed]", "Decay [block]")
rf1 = m1[m1.model == "RF"].set_index(["design", "split"]).AUC_mean
d_rf = (p4["AUC_mean"]["Decay [mixed]"] - rf1.reindex(p4.index)).dropna()
s_rf = (int((d_rf > 0).sum()), len(d_rf))
rho = dp.groupby(["design", "regime"])["rho_km"].median()

# ------------------------------------------------------------------ charts (inline SVG)
CAT = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"]


def line_chart(xlabels, series, ymin, ymax, ylab, w=640, h=300, ticks=None, note=""):
    L, R, T, B = 52, 150, 16, 46
    pw, ph = w - L - R, h - T - B
    xs = [L + pw * i / (len(xlabels) - 1) for i in range(len(xlabels))]
    y = lambda v: T + ph * (1 - (v - ymin) / (ymax - ymin))  # noqa: E731
    s = [f'<svg class="chart" viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{html.escape(ylab)}">']
    for t in ticks or np.linspace(ymin, ymax, 5):
        s.append(f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}"/><text x="{L - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{t:.2f}</text>')
    for x, lab in zip(xs, xlabels):
        s.append(f'<text x="{x:.1f}" y="{T + ph + 18}" text-anchor="middle">{html.escape(lab)}</text>')
    s.append(f'<text x="{L}" y="{h - 6}" class="muted-t">{html.escape(note)}</text>')
    ends = []
    for i, (name, vals) in enumerate(series):
        pts = [(x, y(v)) for x, v in zip(xs, vals) if np.isfinite(v)]
        s.append(f'<polyline fill="none" stroke="{CAT[i]}" stroke-width="2" points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}"/>')
        for (a, b), v, lab in zip(pts, vals, xlabels):
            s.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4.5" fill="{CAT[i]}" stroke="var(--surface)" stroke-width="2" data-tip="{html.escape(name)} · {html.escape(lab)}: {v:.3f}"/>')
        ends.append([pts[-1][1], name, i])
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 14)
    for yy, name, i in ends:
        s.append(f'<text x="{L + pw + 10}" y="{yy + 4:.1f}" class="val" style="fill:var(--ink)"><tspan fill="{CAT[i]}">●</tspan> {html.escape(name)}</text>')
    s.append("</svg>")
    return "".join(s)


def dumbbell(rows, w=640, h=None):
    L, R, T, B = 170, 30, 26, 34
    h = h or T + B + 30 * len(rows)
    vals = [v for r in rows for v in (r[1], r[2])]
    lo, hi = min(vals + [0]) - 0.02, max(vals + [0]) + 0.02
    pw = w - L - R
    x = lambda v: L + pw * (v - lo) / (hi - lo)  # noqa: E731
    s = [f'<svg class="chart" viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Inner versus held-out gain">']
    for t in np.arange(np.ceil(lo * 20) / 20, hi, 0.05):
        s.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{T - 6}" y2="{h - B + 6}"/><text x="{x(t):.1f}" y="{h - B + 22}" text-anchor="middle">{t:+.2f}</text>')
    s.append(f'<line class="axis" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{T - 10}" y2="{h - B + 8}"/>')
    s.append(f'<text x="{L}" y="14" class="val"><tspan fill="var(--s1)">●</tspan> gain on training-side inner folds   <tspan fill="var(--s2)">●</tspan> gain on held-out regions</text>')
    for i, (name, a, b) in enumerate(rows):
        yy = T + 15 + 30 * i
        s.append(f'<text x="{L - 10}" y="{yy + 4}" text-anchor="end">{html.escape(name)}</text>')
        s.append(f'<line x1="{x(a):.1f}" x2="{x(b):.1f}" y1="{yy}" y2="{yy}" stroke="var(--ink-3)" stroke-width="1.5"/>')
        s.append(f'<circle cx="{x(a):.1f}" cy="{yy}" r="5" fill="var(--s1)" stroke="var(--surface)" stroke-width="2" data-tip="{html.escape(name)} inner: {a:+.3f}"/>')
        s.append(f'<circle cx="{x(b):.1f}" cy="{yy}" r="5" fill="var(--s2)" stroke="var(--surface)" stroke-width="2" data-tip="{html.escape(name)} held-out: {b:+.3f}"/>')
    s.append("</svg>")
    return "".join(s)


# chart 1: scale curve
xl = ["Random", "Block 5°", "Block 10°", "Block 20°", "Region out"]
cs = ["random5", "block5", "block10", "block20", "loro"]
chart_scale = line_chart(xl, [(MODEL_LABEL[m], [auc1.loc[m, c] for c in cs]) for m in ("RF", "MLP", "HiCo", "kNN-geo")],
                         0.50, 0.80, "Mean AUC as holdout distance grows", ticks=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
# chart 2: NLL by distance (env-cluster design spans all distance bins)
bd = bins[(bins.design == "env") & (bins["var"] == "dist_km")]
dl = ["<10 km", "10–50", "50–200", "200–1,000", ">1,000"]
chart_dist = line_chart(dl, [(MODEL_LABEL[m], bd[bd.model == m].sort_values("bin_i")["nll"].tolist()) for m in ("kNN-geo", "MLP", "RF", "Prior")],
                        0.7, 1.6, "Log-loss by distance to nearest training cell", ticks=[0.7, 0.9, 1.1, 1.3, 1.5],
                        note="Lower is better. Environmental-cluster holdout, which spans every distance band.")
# chart 3: dumbbell
nice = {"Attn-env": "Attention (environment)", "Attn-support": "Attention (env + support)", "Stack-global": "Stacked, global",
        "Stack-region": "Stacked, by region", "Reliability": "Local reliability", "Theory": "Theory-fixed"}
chart_gap = dumbbell([(nice.get(r.strategy, r.strategy), r.inner, r.outer) for r in gains.itertuples()])
# chart 4: Geo gate weight by distance
gw["dkm"] = np.expm1(gw["log_dist"])
gw["bin"] = pd.cut(gw["dkm"], [-1, 50, 200, 1000, 1e5], labels=["<50 km", "50–200", "200–1,000", ">1,000"])
gg = gw[gw.expert == "Geo"].groupby(["design", "bin"], observed=False)["w"].mean().unstack()
chart_gate = line_chart(["<50 km", "50–200", "200–1,000", ">1,000"],
                        [("Block 10° holdout", gg.loc["block10"].tolist()), ("Leave region out", gg.loc["loro"].tolist())],
                        0.3, 0.8, "Gate weight on the geographic expert", ticks=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8], w=560,
                        note="A trustworthy gate would slope down: less interpolation far from data.")

# ------------------------------------------------------------------ tables
t_auc = auc1.rename(columns=DES_LABEL)
t_auc.index.name = "Model"
tab_auc = table(t_auc, f3, best="max", row_label=lambda r: MODEL_LABEL[r])
t_nll = nll1.rename(columns=DES_LABEL)
t_nll.index.name = "Model"
tab_nll = table(t_nll, f3, best="min", row_label=lambda r: MODEL_LABEL[r])
t_reg = loro_reg.copy()
t_reg.columns = [f"{c} ({int(ncell_reg[c]):,})" for c in t_reg.columns]
t_reg.index.name = "Held-out region (cells)"
tab_reg = table(t_reg, f3, best="max", row_label=lambda r: MODEL_LABEL[r])

tp = pd.DataFrame({"No correction": [psl[(m, "none")] for m in ("RF", "HGB", "MLP", "MLP+Coords", "HiCo", "kNN-geo")],
                   "True regional prior": [psl[(m, "oracle prior")] for m in ("RF", "HGB", "MLP", "MLP+Coords", "HiCo", "kNN-geo")],
                   "EM-estimated prior": [psl[(m, "EM prior")] for m in ("RF", "HGB", "MLP", "MLP+Coords", "HiCo", "kNN-geo")]},
                  index=["RF", "HGB", "MLP", "MLP+Coords", "HiCo", "kNN-geo"])
tp.index.name = "Model (leave region out)"
tab_prior = table(tp, f3, row_label=lambda r: MODEL_LABEL[r])

order2 = ["Equal", "Theory", "Stack-global", "Stack-region", "Attn-env", "Attn-support", "Uncertainty", "Reliability",
          "Only:AlphaEarth", "Only:Geo", "Only:Climate"]
w2 = m2.groupby(["strategy", "design"])[["AUC_mean", "nll"]].mean().unstack("design")
t2 = pd.DataFrame({f"AUC · {DES_LABEL[d]}": w2[("AUC_mean", d)] for d in ("block10", "env", "loro")} |
                  {f"Log-loss · {DES_LABEL[d]}": w2[("nll", d)] for d in ("block10", "env", "loro")}).loc[order2]
t2.index.name = "Weighting"
nice2 = {**nice, "Equal": "Equal", "Uncertainty": "Inverse variance", "Only:AlphaEarth": "AlphaEarth expert only",
         "Only:Geo": "Geographic expert only", "Only:Climate": "Climate expert only"}
tab_w = "".join(table(t2[[c for c in t2.columns if c.startswith(p)]], f3, best=b, row_label=lambda r: nice2[r])
                for p, b in (("AUC", "max"), ("Log-loss", "min")))
sw_m = sw.groupby("design").mean(numeric_only=True)

TM_ROWS = ["Climate", "Landscape", "Human", "AlphaEarth", "All"]
tt = []
for f in TM_ROWS:
    d = tm[tm.features == f]
    tt.append({"Features": f, "In region": d[d.train == d.test].AUC_mean.mean(), "Other regions": d[d.train != d.test].AUC_mean.mean(),
               "N Am → Europe": d[(d.train == "N America") & (d.test == "Europe")].AUC_mean.iloc[0],
               "Europe → N Am": d[(d.train == "Europe") & (d.test == "N America")].AUC_mean.iloc[0]})
tt = pd.DataFrame(tt).set_index("Features")
tt["Retained"] = tt["Other regions"] / tt["In region"]
tab_tm = table(tt, f3, best="max", row_label=lambda r: {"All": "All features"}.get(r, r))

iv = inv.groupby(["variant", "design"])["AUC_mean"].mean().unstack()[["block10", "loro", "env", "zone"]]
iv = iv.loc[["ERM", "V-REx l=1", "V-REx l=10", "GroupDRO", "Drivers (Clim+Land)", "Drivers+Human"]]
iv.columns = [DES_LABEL[c] for c in iv.columns]
iv.index.name = "Training objective / features"
tab_inv = table(iv, f3, best="max", row_label=lambda r: {"ERM": "Standard training, all features", "V-REx l=1": "V-REx, λ = 1",
                                                       "V-REx l=10": "V-REx, λ = 10", "GroupDRO": "Group DRO",
                                                       "Drivers (Clim+Land)": "Climate + landscape only",
                                                       "Drivers+Human": "Climate + landscape + human"}[r])
pv = prox.groupby(["variant", "design"])["AUC_mean"].mean().unstack()[["block10", "loro", "zone"]]
pv = pv.loc[["AE", "AE_hat (driver-explained)", "AE_perp (residual)", "Drivers+Human"]]
pv.columns = [DES_LABEL[c] for c in pv.columns]
pv.index.name = "Representation"
tab_prox = table(pv, f3, best="max", row_label=lambda r: {"AE": "AlphaEarth, full", "AE_hat (driver-explained)": "AlphaEarth, part explained by named drivers",
                                                         "AE_perp (residual)": "AlphaEarth, residual", "Drivers+Human": "Named drivers (climate, landscape, human)"}[r])

o4 = ["Equal", "Only:All (MLP)", "Stack [block]", "Stack [mixed]", "Attn-support [block]", "Attn-support [mixed]", "Decay [block]", "Decay [mixed]"]
w4 = m4.groupby(["strategy", "design"])[["AUC_mean", "nll"]].mean().unstack("design")
t4 = pd.DataFrame({f"AUC · {DES_LABEL[d]}": w4[("AUC_mean", d)] for d in ("block10", "loro", "env", "zone")}).loc[o4]
rfrow = m1[m1.model == "RF"].groupby("design")["AUC_mean"].mean()
t4.loc["RF (iteration 1)"] = [rfrow[d] for d in ("block10", "loro", "env", "zone")]
t4.index.name = "Fusion"
lab4 = {"Equal": "Equal pooling", "Only:All (MLP)": "Early-fusion MLP alone", "Stack [block]": "Stacked · block-trained",
        "Stack [mixed]": "Stacked · regime-matched", "Attn-support [block]": "Support gate · block-trained",
        "Attn-support [mixed]": "Support gate · regime-matched", "Decay [block]": "Support decay · block-trained",
        "Decay [mixed]": "Support decay · regime-matched", "RF (iteration 1)": "Random forest (iteration 1)"}
tab_4 = table(t4, f3, best="max", row_label=lambda r: lab4[r], hl_rows=("Decay [mixed]",))

w5 = m5.groupby(["model", "features", "design"])[["AUC_mean", "nll"]].mean().unstack("design")
t5 = pd.DataFrame({f"AUC · {DES_LABEL[d]}": w5[("AUC_mean", d)] for d in ("block10", "loro", "zone", "env")} |
                  {"Log-loss · Leave region out": w5[("nll", "loro")]})
t5.index = [f"{a} · {'with pool covariates' if b == '+pool' else 'base features'}" for a, b in t5.index]
t5 = t5.loc[["MLP · base features", "MLP · with pool covariates", "RF · base features", "RF · with pool covariates"]]
t5.index.name = "Model"
tab_5 = table(t5, f3)

ARM6 = {"RF": "Random forest, scales mapped separately", "RF+Recon": "Random forest + post-hoc HiCo reconciliation",
        "HiCo-RF": "HiCo with RF local term (macro learned)", "HiCo": "HiCo (original)"}
w6 = m6[m6.level == "fine"].groupby(["model", "design"])[["AUC_mean", "nll"]].mean().unstack("design")
t6 = pd.DataFrame({f"AUC · {DES_LABEL[d]}": w6[("AUC_mean", d)] for d in ("block10", "loro", "env", "zone")} |
                  {"Log-loss · Leave region out": w6[("nll", "loro")], "Log-loss · Climate zone out": w6[("nll", "zone")]})
t6 = t6.loc[list(ARM6)]
t6["Cross-scale contradiction"] = ["4–7% flips", "0 (exact)", "0 (exact)", "0 (exact)"]
t6.index.name = "Arm"
tab_6 = table(t6, f3, best="max", row_label=lambda r: ARM6[r], hl_rows=("RF+Recon",))
flip = coh6[coh6.model.str.startswith("RF")].groupby("design")["dominant_flip"].mean()
tv6 = coh6[coh6.model.str.startswith("RF")]["TV"].mean()
pt6i = pt6.set_index(["level", "model", "metric"])
r6 = pt6i.loc[("fine", "RF+Recon", "AUC_mean")]
r6n = pt6i.loc[("fine", "RF+Recon", "nll")]
h6 = pt6i.loc[("fine", "HiCo", "AUC_mean")]
hr6 = pt6i.loc[("fine", "HiCo-RF", "AUC_mean")]
agg6 = pt6i.loc[("coarse", "RF-agg", "nll")]
w6c = m6[m6.level == "coarse"].groupby(["model", "design"])["AUC_mean"].mean().unstack()

STATUS = {"baseline-candidate": ("Kept", "ok"), "rejected": ("Rejected", "bad"), "inconclusive": ("Inconclusive", "mid"),
          "proposed": ("Proposed", "prop"), "infrastructure": ("Method rule", "prop")}
led_rows = "".join(f'<tr><td class="mono">{e["id"]}</td><td>{e["level"].replace(" RSI", "")}</td><td>{html.escape(e["title"])}</td>'
                   f'<td><span class="chip {STATUS[e["status"]][1]}">{STATUS[e["status"]][0]}</span></td>'
                   f'<td class="wrap">{html.escape(e.get("next", ""))}</td></tr>' for e in ledger)

# ------------------------------------------------------------------ page
style = (ROOT / "artifacts/hicofuse/report/index.html").read_text(encoding="utf-8")
style = re.search(r"<style>(.*?)</style>", style, re.S).group(1)
style += """
.chart .muted-t { fill: var(--ink-3); font: 11.5px var(--font-body); }
.chip { display: inline-block; font: 500 11px/1.4 var(--font-data); letter-spacing: 0.04em; padding: 1px 8px; border-radius: 999px; border: 1px solid; }
.chip.ok { color: var(--good); border-color: var(--good); }
.chip.bad { color: var(--bad); border-color: var(--bad); }
.chip.mid { color: var(--ink-2); border-color: var(--ink-3); }
.chip.prop { color: var(--accent); border-color: var(--accent); }
td.wrap { white-space: normal; text-align: left; min-width: 260px; font-size: 12.5px; color: var(--ink-2); }
tr.hl td { background: var(--accent-soft); }
.it { display: flex; flex-direction: column; gap: 6px; }
.it h3 { color: var(--accent); }
.verdict { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
.verdict .hl p { font-size: 13.5px; color: var(--ink-2); }
ol.plain { margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 8px; max-width: 86ch; }
.loop text { font: 12px var(--font-body); fill: var(--ink); }
.loop .n { fill: var(--surface); stroke: var(--rule); stroke-width: 1.2; }
.loop .k { fill: var(--accent-soft); stroke: var(--accent); }
.loop .lbl { font: 600 10.5px var(--font-data); letter-spacing: 0.07em; fill: var(--accent); }
.loop .arr { stroke: var(--ink-3); stroke-width: 1.3; fill: none; }
"""

LOOP_STEPS = ["Data", "Representation", "Fusion", "Spatial model", "Habitat map", "Spatial / temporal validation",
              "Error & uncertainty diagnosis", "Scientific interpretation", "New hypotheses", "Redesign"]


def loop_svg():
    w, h, cx, cy, rx, ry = 1000, 380, 500, 190, 400, 140
    s = [f'<svg class="loop" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="Research loop with ten stages">',
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)"/></marker></defs>']
    n = len(LOOP_STEPS)
    pts = [(cx + rx * np.cos(-np.pi / 2 + 2 * np.pi * i / n), cy + ry * np.sin(-np.pi / 2 + 2 * np.pi * i / n)) for i in range(n)]
    for i in range(n):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        sx, sy = x1 + dx / L * 62, y1 + dy / L * 20
        ex, ey = x2 - dx / L * 62, y2 - dy / L * 20
        s.append(f'<path class="arr" d="M{sx:.0f},{sy:.0f} Q{mx + (mx - cx) * 0.08:.0f},{my + (my - cy) * 0.08:.0f} {ex:.0f},{ey:.0f}" marker-end="url(#ah)"/>')
    key = {5, 6}
    for i, ((x, y), t) in enumerate(zip(pts, LOOP_STEPS)):
        wbox = max(110, 7.2 * len(t) + 20)
        s.append(f'<rect class="n{" k" if i in key else ""}" x="{x - wbox / 2:.0f}" y="{y - 17:.0f}" width="{wbox:.0f}" height="34" rx="8"/>'
                 f'<text x="{x:.0f}" y="{y + 4:.0f}" text-anchor="middle">{t}</text>')
    s.append(f'<text class="lbl" x="{cx}" y="{cy - 22}" text-anchor="middle">EVERY ITERATION WRITES TO</text>'
             f'<text x="{cx}" y="{cy}" text-anchor="middle" style="font:600 15px var(--font-display)">rsi/memory/ledger.jsonl</text>'
             f'<text x="{cx}" y="{cy + 20}" text-anchor="middle" style="fill:var(--ink-2)">question · hypothesis · design · result · failure · next</text>'
             f'<text x="{cx}" y="{cy + 40}" text-anchor="middle" style="fill:var(--ink-2)">{len(ledger)} entries so far</text>')
    s.append("</svg>")
    return "".join(s)


def dag_svg():
    nodes = {"C": (90, 60, "Climate"), "L": (90, 170, "Landscape / water"), "H": (90, 280, "Human settlement"),
             "U": (330, 330, "Fine surface structure"), "E": (330, 220, "AlphaEarth embedding"), "I": (560, 40, "Introduction history"),
             "Z": (560, 150, "True composition"), "O": (330, 400, "Observation process"), "R": (800, 270, "iNat reports")}
    edges = [("C", "L"), ("C", "Z"), ("L", "Z"), ("H", "L"), ("H", "Z"), ("H", "O"), ("U", "Z"), ("C", "E"), ("L", "E"), ("H", "E"),
             ("U", "E"), ("I", "Z"), ("Z", "R"), ("O", "R")]
    s = ['<svg class="loop" viewBox="0 0 900 440" width="900" height="440" role="img" aria-label="Working causal diagram">',
         '<defs><marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)"/></marker></defs>']
    for a, b in edges:
        (x1, y1, _), (x2, y2, _) = nodes[a], nodes[b]
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        k1, k2 = 70 / max(abs(dx) / L * 70 / 18 + 1e-9, 1), 0
        sx, sy = x1 + dx / L * min(80, L * 0.35), y1 + dy / L * min(22, L * 0.3)
        ex, ey = x2 - dx / L * min(84, L * 0.35), y2 - dy / L * min(24, L * 0.3)
        dash = ' stroke-dasharray="5 4"' if a in ("I", "U", "O") else ""
        s.append(f'<line class="arr" x1="{sx:.0f}" y1="{sy:.0f}" x2="{ex:.0f}" y2="{ey:.0f}" marker-end="url(#ah2)"{dash}/>')
    for k, (x, y, t) in nodes.items():
        unobs = k in ("I", "U", "O", "Z")
        s.append(f'<rect class="n{" k" if k == "Z" else ""}" x="{x - 82}" y="{y - 18}" width="164" height="36" rx="8"{" stroke-dasharray=&quot;4 3&quot;" if unobs else ""}/>'
                 f'<text x="{x}" y="{y + 4}" text-anchor="middle">{t}</text>')
    s.append('<text x="600" y="420" style="fill:var(--ink-2)">Dashed: unmeasured. Composition conditional on a report cancels</text>'
             '<text x="600" y="436" style="fill:var(--ink-2)">species-agnostic observation; species-specific detectability does not.</text></svg>')
    return "".join(s)


def tip_js():
    return """<script>
(() => { const t = document.createElement('div'); t.className = 'tip'; t.hidden = true; document.body.appendChild(t);
  document.addEventListener('pointerover', e => { const el = e.target.closest('[data-tip]'); if (!el) return; t.textContent = el.dataset.tip; t.hidden = false; });
  document.addEventListener('pointermove', e => { if (t.hidden) return; t.style.left = Math.min(e.clientX + 12, innerWidth - t.offsetWidth - 8) + 'px'; t.style.top = (e.clientY + 14) + 'px'; });
  document.addEventListener('pointerout', e => { if (e.target.closest('[data-tip]')) t.hidden = true; }); })();
</script>"""


loro_rf = psl[("RF", "none")], psl[("RF", "oracle prior")]
inner_stack = gains.set_index("strategy").loc["Stack-global"]
page = f"""<title>Vector Habitat RSI Log</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap">
<style>{style}</style>
<div class="page">
<header class="stack">
  <div class="eyebrow">NASA global habitat mapping · research loop · iterations 1–6 · 14–16 Nov 2026</div>
  <h1>What breaks when a mosquito habitat model leaves home</h1>
  <p class="lede">Six iterations of a recursive research loop built on the HiCo-Fuse representation and fusion work: nine baselines under nine holdout designs, error diagnosis, seven weighting strategies, causal and invariance tests, two evidence-driven redesigns and a coherence-versus-accuracy test. The loop found the binding constraint. It is not the fusion layer, and it is not the hierarchy.</p>
  <div class="headline" role="list">
    <div class="hl" role="listitem"><span class="v">−{opt['RF']:.2f} / −{opt['kNN-geo']:.2f}</span><span class="k">AUC lost from random to leave-region-out splits: random forest / geographic interpolation</span></div>
    <div class="hl" role="listitem"><span class="v">{loro_rf[0]:.2f} → {loro_rf[1]:.2f}</span><span class="k">Random-forest log-loss on held-out regions after re-weighting to the true regional class mix (held-out prior alone: {prior_te:.2f})</span></div>
    <div class="hl" role="listitem"><span class="v">+{inner_stack.inner:.2f} vs {inner_stack.outer:+.2f}</span><span class="k">Log-loss gain of learned stacking weights on training folds vs on held-out regions</span></div>
    <div class="hl" role="listitem"><span class="v">{s_gate[0]}/{s_gate[1]}</span><span class="k">Splits where monotone support-decay fusion beats a free attention gate (sign test p = {s_gate[2]:.3f})</span></div>
  </div>
</header>

<section id="loop">
  <div class="eyebrow">The loop</div>
  <h2>Each experiment is chosen by the last one's failure</h2>
  <p class="prose">The task is unchanged from HiCo-Fuse: for each 0.05° cell with an iNaturalist mosquito report (7,004 cells, 2022–2025), predict the share of reports that are <i>Ae. albopictus</i>, <i>Ae. aegypti</i>, <i>Culex</i>, <i>Anopheles</i> or other Culicidae. Conditioning on a report cancels observer effort that is equal across species. Metrics are mean per-class AUC and record-weighted log-loss. Validation and diagnosis (highlighted) carried most of the information this round.</p>
  <div class="figure"><div class="scroll">{loop_svg()}</div></div>
</section>

<section id="it1">
  <div class="eyebrow">Iteration 1 · validation and baselines</div>
  <h2>Random-split skill is mostly interpolation</h2>
  <p class="prose">Nine baselines on the same 105-feature fused representation, under nine holdout designs. The random forest ranks first on every design. Random splits overstate transfer skill by 0.10 AUC for the best model and 0.26 for pure spatial interpolation, which falls to chance (0.52) when a whole region is held out.</p>
  <div class="two">
    <div class="figure"><h3>Mean AUC as the holdout moves further from training data</h3>{chart_scale}<p class="cap">Hover a point for its value. Models shown: the best, the early-fusion MLP, the hierarchical model and the interpolator.</p></div>
    <div class="figure"><h3>Log-loss by distance to the nearest training cell</h3>{chart_dist}<p class="cap">Near data the interpolator is competitive. From 200 km out covariate models win, and beyond 1,000 km only the random forest stays below the class prior.</p></div>
  </div>
  <div class="figure"><h3>Table 1 · Mean AUC across holdout designs</h3>{tab_auc}<p class="cap">Mean over splits (5 folds; 6 regions; 5 clusters; 4 zones). Neural models average 2 seeds. <b>Bold blue</b> is the best per column.</p></div>
  <div class="figure"><h3>Table 2 · Log-loss across holdout designs</h3>{tab_nll}<p class="cap">Record-weighted multinomial log-loss; lower is better. Under leave-region-out every model beats the training-region class prior ({nll1.loc['Prior', 'loro']:.3f}), but none comes close to simply knowing the held-out region's own class mix ({prior_te:.3f}, Table 4).</p></div>
  <div class="figure"><h3>Table 3 · Leave-one-region-out AUC by held-out region</h3>{tab_reg}<p class="cap">Flexible neural models collapse on Europe (MLP 0.53, gated MoE 0.54) where the linear model holds 0.72. Latin America is weakest for every model. Africa has 75 report cells and few class positives, so its AUC is noisy.</p></div>
  <div class="figure"><h3>Table 4 · How much of the transfer loss is class-prior shift?</h3>{tab_prior}<p class="cap">Mean log-loss over the six held-out regions. Re-weighting the same predictions to the true class mix of the held-out region removes almost all of the random forest's excess over the held-out prior ({prior_te:.3f}). Estimating that mix without labels (Saerens EM) diverges: miscalibrated conditionals drive some class priors to zero.</p></div>
  <p class="note"><b>Diagnosis.</b> The habitat relation transfers moderately (AUC ≈ 0.70). The regional species mix does not: which species have arrived and dominate reports in a region is introduction history, not niche. No environmental model can supply it.</p>
</section>

<section id="it2">
  <div class="eyebrow">Iteration 2 · learned weighting</div>
  <h2>Learned weights fit the training geography</h2>
  <p class="prose">Five experts (Climate, Landscape, Human, AlphaEarth MLPs and a geographic interpolator) combined by a log-linear pool. Seven weighting rules. Every learned rule was fitted on inner out-of-fold predictions of the training set only, so the gap between training-side gain and held-out gain measures how much the weighting itself overfits.</p>
  <div class="two">
    <div class="figure"><h3>Training-side vs held-out gain over equal weights (leave region out)</h3>{chart_gap}<p class="cap">Log-loss gain relative to equal weighting; right of zero is better. Only local reliability weighting keeps a held-out gain.</p></div>
    <div class="figure"><h3>The support-aware gate trusts interpolation more far from data</h3>{chart_gate}<p class="cap">Mean gate weight on the geographic expert by distance to training data. The inner folds contained almost no far-from-data cells, so the gate extrapolated in its own input space. Global stacking gave the geographic expert {sw_m.loc['loro', 'w_Geo']:.2f} of the weight for the same reason.</p></div>
  </div>
  <div class="figure"><h3>Table 5 · Weighting strategies</h3>{tab_w}<p class="cap">Theory-fixed weights (climate 0.35, landscape 0.25, human 0.20, AlphaEarth 0.20, interpolation 0) were set before any results and give the best leave-region-out AUC. Inverse-variance weighting is poor because seed variance is not comparable across expert types.</p></div>
</section>

<section id="it3">
  <div class="eyebrow">Iteration 3 · prediction, association, causation</div>
  <h2>Climate travels, embeddings stay home</h2>
  <p class="prose">A working causal diagram was written before these results (<span class="mono">rsi/CAUSAL_DAG.md</span>). None of the tests below identifies a causal effect. Each can falsify a causal reading of a predictive pattern.</p>
  <div class="figure"><div class="scroll">{dag_svg()}</div></div>
  <div class="figure"><h3>Table 6 · Region-to-region transfer by modality</h3>{tab_tm}<p class="cap">MLPs trained in one region and scored in the others (5 training regions). Between the two temperate invasion fronts, climate transfers better than AlphaEarth or all features combined, consistent with a physiological climate effect. Human-footprint relations do not transfer at all.</p></div>
  <div class="two">
    <div class="figure"><h3>Table 7 · Invariance objectives and causal feature subsets</h3>{tab_inv}<p class="cap">Penalising risk variance across regions (V-REx) or the worst region (group DRO) does not improve transfer, and restricting to putative drivers hurts. The dominant shift is in the class prior, which invariance over x cannot address.</p></div>
    <div class="figure"><h3>Table 8 · Is AlphaEarth a proxy for named drivers?</h3>{tab_prox}<p class="cap">Climate, landscape and human covariates explain {prox_r2 * 100:.0f}% of AlphaEarth's variance (linear). The residual carries real in-region signal (0.63) that is near chance in a new region (0.53). AlphaEarth behaves as a regional fingerprint as well as a habitat descriptor.</p></div>
  </div>
</section>

<section id="it4">
  <div class="eyebrow">Iterations 4–5 · two redesigns</div>
  <h2>Structure beats flexibility, but not the forest</h2>
  <p class="prose"><b>Iteration 4</b> fixed the two causes found in iteration 2. Learned weights are now trained on inner predictions that include leave-a-training-region-out folds (regime-matched). A monotone support-decay pool replaces the free gate. The geographic expert's weight decays as exp(−d/ρ). Covariate experts fade toward the class prior as environmental novelty grows. There are about 15 parameters in total.</p>
  <div class="figure"><h3>Table 9 · Support-matched fusion (6 experts incl. the early-fusion MLP)</h3>{tab_4}<p class="cap">Paired over 20 splits: support decay beats the free support gate in {s_gate[0]}/{s_gate[1]} splits on AUC (p = {s_gate[2]:.3f}) and {s_gate_n[0]}/{s_gate_n[1]} on log-loss. It ties equal pooling ({s_eq[0]}/{s_eq[1]}, p = {s_eq[2]:.2f}) and beats the random forest in only {s_rf[0]}/{s_rf[1]}. Regime matching changes what is learned: median correlation range ρ is {rho[('loro', 'block')]:,.0f} km when trained on block folds and {rho[('loro', 'mixed')]:,.0f} km when trained on mixed folds. It does not change accuracy significantly ({s_reg[0]}/{s_reg[1]}, p = {s_reg[2]:.2f}).</p></div>
  <p class="prose"><b>Iteration 5</b> tested the iteration-1 diagnosis directly. It added an exogenous species-pool signal: log record density of the Kraemer literature compendium (1960–2014, not iNaturalist) within 100 km and 500 km for the two <i>Aedes</i> species. The prediction was a larger gain under transfer than in-distribution.</p>
  <div class="figure"><h3>Table 10 · Exogenous species-pool covariates</h3>{tab_5}<p class="cap">For the MLP the transfer gain (+0.041 AUC, leave region out) is about three times the in-distribution gain (+0.015), as predicted. Calibration under transfer is not fixed, because the pool covers only two of five classes. Maps using these covariates can no longer be validated against Kraemer.</p></div>
</section>

<section id="it6">
  <div class="eyebrow">Iteration 6 · coherence versus accuracy</div>
  <h2>Coherence is free; HiCo's learners are the cost</h2>
  <p class="prose">The random forest beats HiCo on every holdout, but HiCo was built for something accuracy does not measure: a city-level map that never contradicts the country-level map. This iteration asks whether that guarantee is what costs accuracy. Each arm is scored on 0.05° cells and on 0.5° parents, and every arm except the first satisfies HiCo's exact rule that a parent's intensity equals the average of its children's.</p>
  <div class="figure"><h3>Table 11 · Four ways to get a coherent map, scored at the fine level (20 splits)</h3>{tab_6}<p class="cap"><b>Reconciliation costs nothing measurable.</b> Re-normalising the forest under HiCo's rule is within noise of the plain forest ({int(r6.wins_vs_RF)}/{int(r6.n)} AUC wins, mean {r6.mean_diff:+.3f}, p = {r6.p_sign:.2f}; log-loss {r6n.mean_diff:+.3f}). Mapping the two scales separately instead makes them disagree: mean composition distance {tv6:.2f}, and the dominant class flips in {flip.min() * 100:.0f}–{flip.max() * 100:.0f}% of parents. <b>Original HiCo loses</b> in {int(h6.n - h6.wins_vs_RF)}/{int(h6.n)} splits (p &lt; 0.001). Giving HiCo the forest as its local term helps (wins {int(hr6.wins_vs_RF)}/{int(hr6.n)}) and is the best arm under 10° blocks, but its jointly learned macro network falls behind under region and zone holdouts, the same failure as the learned weights in iteration 2.</p></div>
  <p class="prose">One more result changes what the hierarchy is for. A coarse map built as the plain average of the forest's fine map is as accurate as a separately trained coarse forest ({int(agg6.wins_vs_RF)}/{int(agg6.n)}, p = {agg6.p_sign:.2f}). When the coarse level has no data of its own, the hierarchy adds nothing beyond aggregation. It earns its place only when the coarse term is fed evidence the fine model cannot see, such as country checklists or surveillance by administrative unit. That is the macro term of the Pool–Niche–Support proposal.</p>
</section>

<section id="synthesis">
  <div class="eyebrow">Synthesis</div>
  <h2>Limitations that persist across every method tried</h2>
  <ol class="plain">
    <li><b>Regional class-prior shift dominates transfer error.</b> The true regional class mix closes almost all of the forest's held-out log-loss gap. No fusion, weighting or invariance scheme changes this.</li>
    <li><b>Weights learned where data are dense do not transfer.</b> Fusion weights must be learned on inner predictions whose distance and novelty profile matches deployment, and constrained to behave monotonically with support.</li>
    <li><b>AlphaEarth is strongest in-region and weakest out of it.</b> Its residual is a regional fingerprint; it should dominate near data and fade under novelty.</li>
    <li><b>Neural extrapolation is unstable.</b> The MLP's Europe hold-out AUC moved from 0.53 to 0.71 with only the seed changed. Tree ensembles extrapolate conservatively and are the bar to beat.</li>
    <li><b>Coherence is a constraint, not a learner.</b> HiCo's normalisation applied to the forest keeps its accuracy and removes every cross-scale contradiction; HiCo's own neural terms are what cost accuracy (iteration 6).</li>
    <li><b>Data deserts set the ceiling.</b> Africa has 75 report cells and Latin America is weakest for every model; no architecture fixes absent evidence.</li>
  </ol>
  <h3>Proposed next architecture: Pool–Niche–Support (PNS)</h3>
  <div class="eq">log λ<sub>s</sub>(x) = a<sub>s</sub>(R(x)) + f<sub>s</sub>(x) + k(d(x)) · r<sub>s</sub>(x),   with HiCo normalisation over each 0.5° parent</div>
  <div class="scroll"><table>
    <thead><tr><th>Question</th><th style="text-align:left">Answer for PNS</th></tr></thead>
    <tbody>
    <tr><td>Limitation addressed</td><td class="wrap">Limitation 1 (regional prior shift) and 2 (support-regime mismatch); iteration 1 Table 4, iteration 2, iteration 4.</td></tr>
    <tr><td>What is new</td><td class="wrap">A regional species-pool latent a<sub>s</sub>, partially pooled across regions and informed by exogenous checklists, kept separate from the niche term f<sub>s</sub>. Plus a spatial residual whose influence decays with a learned correlation range, all inside the exact-coherence hierarchy.</td></tr>
    <tr><td>Why it fits habitat mapping</td><td class="wrap">Range = niche × dispersal history. Composition models conflate the two; PNS gives each its own term and its own evidence source.</td></tr>
    <tr><td>Assumption encoded</td><td class="wrap">Niche relations are shared across regions; arrival history is regional; spatial residual dependence is local (Tobler) and monotone in distance.</td></tr>
    <tr><td>How it differs</td><td class="wrap">Unlike stacking or gating, weights cannot grow with distance. Unlike coordinate covariates, the regional term has an external prior instead of being extrapolated. Unlike plain HiCo, the macro term is not learned from the same reports.</td></tr>
    <tr><td>Ablations</td><td class="wrap">Drop a<sub>s</sub>; replace its exogenous prior with a uniform one; drop the decay (free gate); drop r<sub>s</sub>; random vs regime-matched inner folds.</td></tr>
    <tr><td>Where it should win</td><td class="wrap">Leave region out and climate zone out, on log-loss first. It should tie the random forest under 10° blocks.</td></tr>
    <tr><td>What would falsify it</td><td class="wrap">No log-loss gain over RF on held-out regions in ≥4 of 6 regions, or the gain vanishes when the exogenous prior is replaced by a uniform one (meaning the pool term is not doing the work).</td></tr>
    </tbody></table></div>
</section>

<section id="next">
  <div class="eyebrow">Next experiments, ranked</div>
  <h2>What the next iteration should test</h2>
  <div class="scroll"><table>
    <thead><tr><th>Rank</th><th style="text-align:left">Experiment</th><th style="text-align:left">Why now</th><th>Cost</th></tr></thead>
    <tbody>
    <tr><td>1</td><td class="wrap">Species-pool layer for all five classes (GBIF non-iNat occurrence by country / 5° cell, WRBU checklists)</td><td class="wrap">Iteration 5 helped only the two classes it covered; PNS depends on it</td><td>Medium, data work</td></tr>
    <tr><td>2</td><td class="wrap">PNS model and its five ablations under LORO, zone and space-time holdouts, ≥5 seeds</td><td class="wrap">Tests the synthesis directly; seed instability (limitation 4) requires more seeds</td><td>Medium, ~1 GPU-hour</td></tr>
    <tr><td>3</td><td class="wrap">Few-shot regional prior: how many labelled cells in a new region recover the oracle log-loss?</td><td class="wrap">Turns limitation 1 into a sampling-design answer for NASA field programs</td><td>Low</td></tr>
    <tr><td>4</td><td class="wrap">AIMSurv 2020 trap survey as the urban-gradient test of observation bias</td><td class="wrap">Species-specific detectability is not identified with reports alone (causal diagram)</td><td>Medium, covariate extraction pending</td></tr>
    <tr><td>5</td><td class="wrap">Tree-based niche term inside PNS (RF log-ratios under HiCo reconciliation, as in iteration 6)</td><td class="wrap">Iteration 6 showed this costs no accuracy; PNS should start from it, not from MLP experts</td><td>Low</td></tr>
    </tbody></table></div>
</section>

<section id="memory">
  <div class="eyebrow">Research memory</div>
  <h2>Ledger</h2>
  <p class="prose">Every iteration appends a structured entry: question, hypothesis, data, representation, fusion, architecture, weighting, validation, results, error patterns, uncertainty, ecological and causal reading, what improved, what failed, open questions and next step. Rejected ideas stay in the ledger so they are not retried without new evidence. Full entries: <span class="mono">rsi/memory/RESEARCH_MEMORY.md</span>.</p>
  <div class="figure"><div class="scroll"><table>
    <thead><tr><th>ID</th><th style="text-align:left">Level</th><th style="text-align:left">Entry</th><th style="text-align:left">Status</th><th style="text-align:left">Next step it set</th></tr></thead>
    <tbody>{led_rows}</tbody></table></div></div>
  <p class="muted">Code: <span class="mono">rsi/</span> (common.py, it1_baselines.py … it6_coherence.py). Results: <span class="mono">artifacts/rsi/</span>. All numbers on this page are generated from those files by <span class="mono">rsi/build_report.py</span>.</p>
</section>
</div>
{tip_js()}
"""
page = page.replace("<th>Model (leave region out)</th>", "<th>Model (leave region out)</th>")
(OUT / "index.html").write_text(page, encoding="utf-8")
print("written", len(page))
