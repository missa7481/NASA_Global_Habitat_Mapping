#!/usr/bin/env python3
"""Build the public, mission-control styled site in site/ (GitHub Pages ready).

Re-uses the body of the RSI report (every number still read from disk by build_report.py) and wraps it in a
dark Earth-observation theme with a global map hero, an iteration timeline and a data/attribution footer.
Only aggregate results and 0.2-degree model maps are published; no observation coordinates.
"""

from __future__ import annotations

import html
import re
import runpy
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "artifacts/rsi/site_preview"            # local preview only; not published
ASSETS = SITE / "assets"
SPECIES = [("albopictus", "Ae. albopictus", "Asian tiger mosquito"), ("aegypti", "Ae. aegypti", "Yellow fever mosquito"),
           ("culex", "Culex", "West Nile vectors"), ("anopheles", "Anopheles", "Malaria vectors")]
NIGHT = ["#0f2238", "#123a63", "#15568c", "#1874b0", "#2b95c8", "#4fb6d6", "#86d3dc", "#c4ead9", "#fff4c2"]


def webp_maps():
    for k, *_ in SPECIES:
        src = ASSETS / f"map_{k}.png"
        if not src.exists():          # already converted (run render_site_maps.py to refresh)
            continue
        Image.open(src).save(ASSETS / f"map_{k}.webp", "WEBP", quality=88, method=6)


g = runpy.run_path(str(ROOT / "rsi/build_report.py"), run_name="report")
page, style = g["page"], g["style"]
body = page.split("</style>", 1)[1]
body = re.sub(r'<header class="stack">.*?</header>', "", body, count=1, flags=re.S)   # replaced by the hero
opt, loro_rf, prior_te, gains = g["opt"], g["loro_rf"], g["prior_te"], g["gains"]
s_gate, r6, flip, ledger = g["s_gate"], g["r6"], g["flip"], g["ledger"]
w5 = g["w5"]
pool_gain = float(w5.loc[("MLP", "+pool"), ("AUC_mean", "loro")] - w5.loc[("MLP", "base"), ("AUC_mean", "loro")])
stack = gains.set_index("strategy").loc["Stack-global"]

TIMELINE = [
    ("01", "Baselines", "it1", f"Random forest leads all 9 designs; leaving a region out costs it {opt['RF']:.2f} AUC"),
    ("01b", "Prior shift", "it1", f"True regional class mix: RF log-loss {loro_rf[0]:.2f} → {loro_rf[1]:.2f}"),
    ("02", "Weighting", "it2", f"Learned stacking: {stack.inner:+.2f} on training folds, {stack.outer:+.2f} on new regions"),
    ("03", "Causality", "it3", "Climate transfers between continents; AlphaEarth's residual stays home"),
    ("04", "Support decay", "it4", f"Monotone fusion beats a free gate in {s_gate[0]}/{s_gate[1]} splits"),
    ("05", "Species pool", "it4", f"Exogenous pool covariates: MLP +{pool_gain:.3f} AUC on held-out regions"),
    ("06", "Coherence", "it6", f"Reconciled forest = forest accuracy ({int(r6.wins_vs_RF)}/{int(r6.n)}, p = {r6.p_sign:.2f}), zero contradictions"),
]

THEME = """
:root, :root[data-theme="light"], :root[data-theme="dark"] {
  --bg: #04070d; --surface: rgba(14, 22, 36, 0.78); --ink: #e9f0f7; --ink-2: #a9b8c8; --ink-3: #7d8da0;
  --rule: rgba(120, 160, 210, 0.16); --accent: #5fb4ff; --accent-soft: rgba(70, 140, 220, 0.12);
  --s1: #4a9bf0; --s2: #f0703c; --s3: #2fbf8a; --s4: #e8ad1c; --good: #45c96b; --bad: #ff6a55; --nodata: #2a3342;
  --d-2: #1d5fae; --d-1: #1a3f6b; --d0: #1d2532; --d1: #6e2c2b; --d2: #b8352f;
  --flare: #ff5a36; --hud: #7fd3ff; color-scheme: dark;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #04070d; --surface: rgba(14, 22, 36, 0.78); --ink: #e9f0f7; --ink-2: #a9b8c8; --ink-3: #7d8da0; --rule: rgba(120, 160, 210, 0.16); --accent: #5fb4ff; --accent-soft: rgba(70, 140, 220, 0.12); } }
html { scroll-behavior: smooth; scroll-padding-top: 64px; }
body { margin: 0; background: var(--bg); color: var(--ink);
  background-image:
    radial-gradient(1px 1px at 12% 18%, rgba(255,255,255,.55) 50%, transparent 51%),
    radial-gradient(1px 1px at 72% 9%, rgba(255,255,255,.4) 50%, transparent 51%),
    radial-gradient(1.5px 1.5px at 38% 64%, rgba(255,255,255,.35) 50%, transparent 51%),
    radial-gradient(1px 1px at 88% 52%, rgba(255,255,255,.5) 50%, transparent 51%),
    radial-gradient(1px 1px at 22% 86%, rgba(255,255,255,.35) 50%, transparent 51%),
    radial-gradient(1200px 600px at 80% -10%, rgba(40, 90, 170, .25), transparent 60%),
    radial-gradient(900px 500px at -10% 30%, rgba(255, 90, 54, .07), transparent 60%);
  background-attachment: fixed; }
.page { padding-top: 0; }
.figure, .hl { backdrop-filter: blur(6px); border-color: var(--rule); box-shadow: inset 0 1px 0 rgba(255,255,255,.04); }
section { position: relative; }
section > .eyebrow { color: var(--hud); }
section > .eyebrow::before { content: "▸ "; color: var(--flare); }
h2 { letter-spacing: -0.01em; }
.chip.ok { background: rgba(69,201,107,.14); color: #8fe3a6; } .chip.bad { background: rgba(255,106,85,.14); color: #ffa292; }
.chip.mid { background: rgba(232,173,28,.14); color: #f3cf74; } .chip.prop { background: rgba(95,180,255,.14); color: #a6d6ff; }
tr.hl td { background: rgba(95,180,255,.08); }

/* ---- top bar */
.bar { position: sticky; top: 0; z-index: 20; display: flex; align-items: center; gap: 18px; padding: 10px max(16px, env(safe-area-inset-left, 0px));
  background: rgba(4, 7, 13, .82); backdrop-filter: blur(10px); border-bottom: 1px solid var(--rule); }
.brand { display: flex; align-items: center; gap: 10px; font: 600 13px var(--font-data); letter-spacing: .14em; text-transform: uppercase; color: var(--ink); white-space: nowrap; text-decoration: none; }
.bar nav { min-width: 0; display: flex; gap: 4px; overflow-x: auto; scrollbar-width: none; margin-left: auto; }
.bar nav a { font: 500 11.5px var(--font-data); letter-spacing: .08em; text-transform: uppercase; color: var(--ink-3); text-decoration: none; padding: 6px 9px; border-radius: 6px; white-space: nowrap; }
.bar nav a:hover, .bar nav a.on { color: var(--ink); background: var(--accent-soft); }

/* ---- hero */
.hero > * { min-width: 0; }
.hero { position: relative; max-width: 1120px; margin: 0 auto; padding: 48px max(16px, env(safe-area-inset-left, 0px)) 8px; display: grid; gap: 28px; }
.hero .kicker { font: 500 12px var(--font-data); letter-spacing: .18em; text-transform: uppercase; color: var(--hud); display: flex; gap: 14px; flex-wrap: wrap; align-items: center; }
.hero .kicker .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--flare); box-shadow: 0 0 12px var(--flare); animation: blink 2.4s infinite; }
@keyframes blink { 50% { opacity: .25; } }
.hero h1 { font-size: clamp(34px, 6vw, 68px); line-height: 1.02; letter-spacing: -0.025em; max-width: 18ch; }
.hero h1 em { font-style: normal; background: linear-gradient(90deg, #7fd3ff, #fff4c2); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero .lede { font-size: clamp(16px, 2vw, 19px); max-width: 64ch; }
.globe { position: relative; border: 1px solid var(--rule); border-radius: 14px; overflow: hidden; background: radial-gradient(ellipse at 50% 40%, #0b1626, #04070d 70%); }
.globe .frame { position: relative; margin: 26px 34px 30px; }
.globe img { display: block; width: 100%; height: auto; transition: opacity .5s ease; }
.globe img.fade { opacity: 0; }
.globe .grat { position: absolute; inset: 0; pointer-events: none; }
.globe .grat line { stroke: rgba(127, 211, 255, .13); stroke-width: 1; vector-effect: non-scaling-stroke; }
.globe .grat line.eq { stroke: rgba(255, 90, 54, .35); stroke-dasharray: 4 4; }
.globe .ticks { position: absolute; font: 500 10px var(--font-data); color: var(--ink-3); letter-spacing: .06em; }
.scan { position: absolute; top: 0; bottom: 0; width: 2px; left: 0; background: linear-gradient(180deg, transparent, rgba(127,211,255,.7), transparent);
  box-shadow: 0 0 18px rgba(127,211,255,.6); animation: scan 9s linear infinite; pointer-events: none; }
@keyframes scan { from { left: 0%; } to { left: 100%; } }
@media (prefers-reduced-motion: reduce) { .scan, .hero .kicker .dot { animation: none; } .scan { display: none; } }
.hud { position: absolute; top: 10px; left: 14px; right: 14px; display: flex; justify-content: space-between; gap: 10px; font: 500 11px var(--font-data); letter-spacing: .08em; color: var(--hud); text-transform: uppercase; pointer-events: none; }
.hud .r { text-align: right; color: var(--ink-3); }
.controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; padding: 0 34px 18px; }
.sp { display: flex; flex-wrap: wrap; gap: 6px; }
.sp button { font: 500 12.5px var(--font-body); color: var(--ink-2); background: rgba(255,255,255,.03); border: 1px solid var(--rule); border-radius: 999px; padding: 6px 13px; cursor: pointer; }
.sp button i { font-style: italic; }
.sp button[aria-pressed="true"] { color: #04070d; background: var(--hud); border-color: var(--hud); }
.sp button:focus-visible { outline: 2px solid var(--hud); outline-offset: 2px; }
.mlegend { display: flex; align-items: center; gap: 10px; font: 500 11px var(--font-data); color: var(--ink-3); letter-spacing: .05em; text-transform: uppercase; }
.mlegend .ramp { width: 180px; height: 8px; border-radius: 4px; background: linear-gradient(90deg, RAMP); }
.mcap { padding: 0 34px 22px; font-size: 13px; color: var(--ink-3); max-width: 110ch; }

/* ---- telemetry tiles */
.tele { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1px; background: var(--rule); border: 1px solid var(--rule); border-radius: 12px; overflow: hidden; }
.tele div { background: rgba(8, 13, 22, .92); padding: 16px 18px; display: flex; flex-direction: column; gap: 6px; }
.tele .k { font: 500 10.5px var(--font-data); letter-spacing: .12em; text-transform: uppercase; color: var(--ink-3); }
.tele .v { font: 600 30px/1.05 var(--font-display); font-variant-numeric: tabular-nums; color: var(--ink); }
.tele .d { font-size: 12.5px; color: var(--ink-2); }

/* ---- mission timeline */
.tl { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 10px; }
.tl li { position: relative; border: 1px solid var(--rule); border-radius: 10px; padding: 14px 14px 12px; background: var(--surface); display: flex; flex-direction: column; gap: 6px; }
.tl li::before { content: ""; position: absolute; top: -1px; left: 14px; right: 14px; height: 2px; background: linear-gradient(90deg, var(--hud), transparent); }
.tl li.last::before { background: linear-gradient(90deg, var(--flare), transparent); }
.tl a { color: inherit; text-decoration: none; }
.tl .n { font: 600 11px var(--font-data); letter-spacing: .14em; color: var(--hud); }
.tl li.last .n { color: var(--flare); }
.tl .t { font: 600 15px var(--font-display); }
.tl .m { font-size: 12.5px; color: var(--ink-2); line-height: 1.45; }

footer.site { max-width: 1120px; margin: 40px auto 0; padding: 28px max(16px, env(safe-area-inset-left, 0px)) 56px; border-top: 1px solid var(--rule); display: grid; gap: 14px; font-size: 13px; color: var(--ink-3); }
footer.site b { color: var(--ink-2); font-weight: 600; }
footer.site .notice { font: 500 11px var(--font-data); letter-spacing: .1em; text-transform: uppercase; color: var(--flare); }
@media (max-width: 640px) { .globe .frame { margin: 22px 12px 26px; } .controls, .mcap { padding-left: 12px; padding-right: 12px; } .hud .r { display: none; } .globe .ticks { display: none; } }
""".replace("RAMP", ", ".join(NIGHT))

PATCH = """<svg width="28" height="28" viewBox="0 0 40 40" aria-hidden="true">
<circle cx="20" cy="20" r="11" fill="#0b2a52" stroke="#7fd3ff" stroke-width="1.5"/>
<path d="M11 17c4 2 9 1 13-2m-12 9c5-1 11 0 15 3" stroke="#7fd3ff" stroke-width="1" fill="none" opacity=".7"/>
<ellipse cx="20" cy="20" rx="18" ry="6.5" transform="rotate(-24 20 20)" fill="none" stroke="#ff5a36" stroke-width="1.6"/>
<circle cx="35" cy="13" r="2" fill="#ff5a36"/></svg>"""


def graticule():
    # equirectangular frame: lon -180..180, lat 85..-60 (the rendered window)
    s = ['<svg class="grat" viewBox="0 0 360 145" preserveAspectRatio="none" aria-hidden="true">']
    for lon in range(-150, 181, 30):
        s.append(f'<line x1="{lon + 180}" x2="{lon + 180}" y1="0" y2="145"/>')
    for lat in (60, 30, 0, -30):
        s.append(f'<line class="{"eq" if lat == 0 else ""}" x1="0" x2="360" y1="{85 - lat}" y2="{85 - lat}"/>')
    s.append("</svg>")
    t = []
    for lat in (60, 30, 0, -30):
        t.append(f'<span class="ticks" style="left:-30px;top:calc({(85 - lat) / 145 * 100:.2f}% - 7px)">{abs(lat)}°{"N" if lat > 0 else "S" if lat < 0 else ""}</span>')
    for lon in (-120, -60, 0, 60, 120):
        t.append(f'<span class="ticks" style="bottom:-20px;left:calc({(lon + 180) / 360 * 100:.2f}% - 12px)">{abs(lon)}°{"W" if lon < 0 else "E" if lon > 0 else ""}</span>')
    return "".join(s) + "".join(t)


def main():
    webp_maps()
    buttons = "".join(f'<button type="button" data-k="{k}" data-n="{html.escape(n)}" aria-pressed="{"true" if i == 0 else "false"}"><i>{html.escape(n)}</i></button>'
                      for i, (k, n, _) in enumerate(SPECIES))
    tl = "".join(f'<li class="{"last" if i == len(TIMELINE) - 1 else ""}"><a href="#{a}"><span class="n">ITER {n}</span><br><span class="t">{t}</span></a><span class="m">{html.escape(m)}</span></li>'
                 for i, (n, t, a, m) in enumerate(TIMELINE))
    n_rej = sum(e["status"] == "rejected" for e in ledger)
    hero = f"""
<header class="hero" id="top">
  <div class="kicker"><span class="dot" aria-hidden="true"></span><span>Global vector habitat · Earth observation fusion</span><span aria-hidden="true">/</span><span>Research log · 6 iterations · {len(ledger)} ledger entries</span></div>
  <h1>What breaks when a mosquito model <em>leaves home</em></h1>
  <p class="lede">A recursive research loop on global mosquito habitat mapping. Climate, landscape, human footprint and AlphaEarth satellite embeddings are fused and tested under nine ways of holding out the world. The loop found where transfer fails: not the fusion layer, and not the global-to-local hierarchy, but which species have arrived in each region.</p>
  <figure class="globe" style="margin:0">
    <div class="hud"><span id="hud-l">Layer · <i>Ae. albopictus</i> · HiCo · 0.2°</span><span class="r" id="hud-r">Lat —  Lon —</span></div>
    <div class="frame" id="frame">
      <img id="map" src="assets/map_albopictus.webp" width="1800" height="725" alt="Global map of predicted relative report share for the selected mosquito class">
      {graticule()}
      <div class="scan" aria-hidden="true"></div>
    </div>
    <div class="controls"><div class="sp" role="group" aria-label="Mosquito class">{buttons}</div>
      <div class="mlegend"><span>Low</span><span class="ramp"></span><span>High share percentile</span></div></div>
    <figcaption class="mcap">HiCo-Fuse two-part model, percentile of predicted share among land cells, 85°N–60°S. Grey: land without complete covariates. These maps show the model, not validated occurrence: the findings below show where they do not transfer, for example bright patches far outside known ranges.</figcaption>
  </figure>
  <div class="tele" role="list">
    <div role="listitem"><span class="k">Transfer penalty · leave region out</span><span class="v">−{opt['RF']:.2f} / −{opt['kNN-geo']:.2f}</span><span class="d">AUC lost by the random forest / pure geographic interpolation versus random splits</span></div>
    <div role="listitem"><span class="k">Species-pool shift</span><span class="v">{loro_rf[0]:.2f} → {loro_rf[1]:.2f}</span><span class="d">Forest log-loss on new regions once given the true regional class mix (prior alone {prior_te:.2f})</span></div>
    <div role="listitem"><span class="k">Cross-scale contradiction</span><span class="v">{flip.min() * 100:.0f}–{flip.max() * 100:.0f}% → 0</span><span class="d">Parents whose dominant class flips between separate coarse and fine maps, before and after reconciliation</span></div>
    <div role="listitem"><span class="k">Ideas rejected on evidence</span><span class="v">{n_rej}</span><span class="d">Kept in the ledger so they are not retried without new evidence</span></div>
  </div>
</header>
<div class="page">
<section id="timeline">
  <div class="eyebrow">Mission log</div>
  <h2>Six iterations at a glance</h2>
  <ol class="tl">{tl}</ol>
</section>"""
    nav = "".join(f'<a href="#{a}">{t}</a>' for a, t in (("timeline", "Log"), ("it1", "Baselines"), ("it2", "Weighting"), ("it3", "Causal"),
                                                            ("it4", "Fusion"), ("it6", "Coherence"), ("synthesis", "Synthesis"), ("next", "Next"), ("memory", "Ledger")))
    foot = """
<footer class="site">
  <span class="notice">Independent research prototype · not an official NASA product</span>
  <p><b>Data.</b> Mosquito reports: iNaturalist research-grade Culicidae observations 2022–2025 (via GBIF), aggregated to 0.05° cells. Covariates: WorldClim-style bioclimatic variables, landscape and human-footprint layers, Google AlphaEarth Foundations 2024 embeddings. External records: Kraemer et al. (2015) <i>Aedes</i> compendium. No observation coordinates are published here.</p>
  <p><b>Code and benchmarks.</b> <a href="https://github.com/missa7481/NASA_Global_Habitat_Mapping" style="color:var(--hud)">github.com/missa7481/NASA_Global_Habitat_Mapping</a>: model development (HiCo-Fuse, OA-DSDM, research loop) and <a href="https://github.com/missa7481/NASA_Global_Habitat_Mapping/blob/main/BENCHMARK.md" style="color:var(--hud)">all benchmark tables</a>.</p>
  <p><b>Method.</b> Every number on this page is generated from experiment output files. Holdout designs: random, 5°/10°/20° spatial blocks, leave one region out, environmental cluster, climate zone, temporal and space-time.</p>
</footer>"""
    js = """<script>
(() => {
  const img = document.getElementById('map'), hudL = document.getElementById('hud-l'), hudR = document.getElementById('hud-r');
  const btns = [...document.querySelectorAll('.sp button')];
  btns.forEach(b => { const pre = new Image(); pre.src = `assets/map_${b.dataset.k}.webp`;
    b.addEventListener('click', () => {
      btns.forEach(x => x.setAttribute('aria-pressed', String(x === b)));
      img.classList.add('fade');
      setTimeout(() => { img.src = `assets/map_${b.dataset.k}.webp`; img.onload = () => img.classList.remove('fade'); }, 250);
      hudL.innerHTML = `Layer · <i>${b.dataset.n}</i> · HiCo · 0.2°`;
    }); });
  const fr = document.getElementById('frame');
  fr.addEventListener('pointermove', e => { const r = img.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return;
    const lat = 85 - fy * 145, lon = -180 + fx * 360;
    hudR.textContent = `Lat ${Math.abs(lat).toFixed(1)}°${lat >= 0 ? 'N' : 'S'}  Lon ${Math.abs(lon).toFixed(1)}°${lon >= 0 ? 'E' : 'W'}`; });
  fr.addEventListener('pointerleave', () => { hudR.textContent = 'Lat —  Lon —'; });
  const links = [...document.querySelectorAll('.bar nav a')];
  const io = new IntersectionObserver(es => es.forEach(en => { if (en.isIntersecting) links.forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + en.target.id)); }), { rootMargin: '-45% 0px -50% 0px' });
  links.forEach(a => { const s = document.querySelector(a.getAttribute('href')); if (s) io.observe(s); });
})();
</script>"""
    body2 = body.replace('<div class="page">', "", 1)
    doc = f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Vector Habitat Research Log</title>
<meta name="description" content="A recursive research loop on global mosquito habitat mapping with Earth-observation fusion: what breaks when a model leaves home.">
<meta name="theme-color" content="#04070d">
<link rel="icon" href="data:image/svg+xml,{html.escape(PATCH.replace(chr(10), '').replace('"', "'"))}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap">
<style>{style}{THEME}</style>
</head>
<body>
<div class="bar"><a class="brand" href="#top">{PATCH}<span>Vector Habitat · RSI</span></a><nav aria-label="Sections">{nav}</nav></div>
{hero}
{body2}
{foot}
{js}
</body>
</html>"""
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(doc, encoding="utf-8")
    (SITE / ".nojekyll").write_text("")
    for k, *_ in SPECIES:
        (ASSETS / f"map_{k}.png").unlink(missing_ok=True)
    print("site written", len(doc))


if __name__ == "__main__":
    main()
