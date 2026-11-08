"""Build reports/eval_report.html: the trained model against two baselines, with guardrails and latency.

Baselines: the logging policy (what the training data looked like) and a random policy over the
same eligible candidates (same rules, no model). "Good" = the model beats both, trips no guardrail,
and stays inside the latency budget. Pure HTML + inline SVG, no external libraries.
"""
import html, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from shopfeed.config import HEADS, Config
from shopfeed.data.store import load_tables
from shopfeed.eval.ab import compare
from shopfeed.eval.guardrails import guardrail_report
from shopfeed.eval.simulator import simulate_responses, responses_to_events
from shopfeed.schema import FeedRequest
from shopfeed.serving.pipeline import FeedService
from shopfeed.training.train_ranker import load_artifacts

N_USERS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
cfg = Config()
tables = load_tables(cfg.data_dir)
now = json.loads((cfg.data_dir / "meta.json").read_text())["now"]
retrieval, light, heavy = load_artifacts(cfg.artifacts_dir / "ranker_v1")
metrics = json.loads((cfg.artifacts_dir / "ranker_v1" / "metrics.json").read_text())
users = tables["users"][tables["users"].cohort == "normal"].sample(N_USERS, random_state=0)


def run(policy, seed=1):
    svc = FeedService(tables, retrieval, light, heavy, cfg, now=now, seed=seed, policy=policy)
    rng = np.random.default_rng(seed)
    pages, events, lat, stages = [], [], [], []
    for u in users.itertuples(index=False):
        resp = svc.recommend(FeedRequest(user_id=u.user_id, country=u.country, session_id="report", now=now))
        lat.append(resp.timings_ms["total"]); stages.append(resp.timings_ms)
        page = pd.DataFrame([c.model_dump() for c in resp.cards])
        sim = simulate_responses(page, users.set_index("user_id").loc[u.user_id], tables["products"], tables["merchants"], rng)
        pages.append(sim.assign(user_id=u.user_id, ts=now)); events.append(responses_to_events(sim, u.user_id, now, len(events) * 1000))
    pages = pd.concat(pages, ignore_index=True).merge(tables["products"][["product_id", "published_at"]], on="product_id", how="left")
    return pages, pd.concat(events, ignore_index=True), np.array(lat), pd.DataFrame(stages)


t0 = time.time()
pm, em, lat, stages = run("model")
pr, er, _, _ = run("random")
imp, ev = tables["impressions"], tables["events"]
logged_return = ev[ev.type == "return"].impression_id.nunique() / max(ev[ev.type == "purchase"].impression_id.nunique(), 1)
logged = {
    "click": ev[ev.type == "view"].impression_id.nunique() / len(imp),
    "purchase": ev[ev.type == "purchase"].impression_id.nunique() / len(imp),
    "hide": ev[ev.type == "hide"].impression_id.nunique() / len(imp),
}
rates = {name: {"click": p.click.mean(), "purchase": p.purchase.mean(), "hide": p.hide.mean()} for name, p in (("random", pr), ("model", pm))}
gmv = {"random": pr.groupby("user_id").order_value.sum().mean(), "model": pm.groupby("user_id").order_value.sum().mean()}
ab = compare(pd.concat([pr.groupby("user_id").order_value.sum().rename("metric").reset_index().assign(bucket="control"),
                        pm.groupby("user_id").order_value.sum().rename("metric").reset_index().assign(bucket="treatment")]), n_boot=500)
guards = guardrail_report(pm, em, cfg.guardrails, cfg.rerank.fresh_window_s)
guards_r = guardrail_report(pr, er, cfg.guardrails, cfg.rerank.fresh_window_s)
p50, p95 = np.percentile(lat, 50), np.percentile(lat, 95)
stage_med = stages.drop(columns=["total"]).median()

# ------------------------------------------------------------------ SVG helpers
def grouped_bars(title, cats, series, fmt, ref=None, width=560, ref_label="logging policy (training data)"):
    """series: list of (name, cssvar, values). Thin bars, 2px gap, direct labels, legend."""
    n, k = len(cats), len(series)
    h, top, bottom, left = 240, 28, 44, 48
    plot_h = h - top - bottom
    vmax = max(max(v) for _, _, v in series) * 1.25 or 1
    gw = (width - left - 16) / n
    bw = min(28, (gw - 16) / k)
    out = [f'<svg class="chart" viewBox="0 0 {width} {h}" role="img" aria-label="{html.escape(title)}">']
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * (1 - frac)
        out.append(f'<line x1="{left}" x2="{width-16}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{left-6}" y="{y+4:.1f}" class="axis" text-anchor="end">{fmt(vmax*frac)}</text>')
    out.append(f'<line x1="{left}" x2="{width-16}" y1="{top+plot_h}" y2="{top+plot_h}" class="baseline"/>')
    for i, c in enumerate(cats):
        x0 = left + gw * i + (gw - (bw * k + 2 * (k - 1))) / 2
        for j, (name, var, vals) in enumerate(series):
            v = vals[i]; bh = plot_h * v / vmax
            x = x0 + j * (bw + 2); y = top + plot_h - bh
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="4" style="fill:var({var})"><title>{html.escape(name)} · {html.escape(c)}: {fmt(v)}</title></rect>')
            out.append(f'<rect x="{x:.1f}" y="{y+bh-4 if bh>4 else y:.1f}" width="{bw:.1f}" height="{min(4,bh):.1f}" style="fill:var({var})"/>')
            nudge = 0 if k == 1 else (-4 if j == 0 else 4) * (1 if k == 2 else 0)
            out.append(f'<text x="{x+bw/2+nudge:.1f}" y="{y-5:.1f}" class="val" text-anchor="middle">{fmt(v)}</text>')
        if ref is not None:
            ry = top + plot_h * (1 - ref[i] / vmax)
            out.append(f'<line x1="{left+gw*i+6:.1f}" x2="{left+gw*(i+1)-6:.1f}" y1="{ry:.1f}" y2="{ry:.1f}" class="ref"><title>logging policy · {html.escape(c)}: {fmt(ref[i])}</title></line>')
        out.append(f'<text x="{left+gw*i+gw/2:.1f}" y="{h-bottom+18}" class="axis" text-anchor="middle">{html.escape(c)}</text>')
    lx = left
    for name, var, _ in series:
        out.append(f'<rect x="{lx}" y="{h-14}" width="10" height="10" rx="2" style="fill:var({var})"/><text x="{lx+14}" y="{h-5}" class="axis">{html.escape(name)}</text>')
        lx += 14 + 7 * len(name) + 18
    if ref is not None:
        out.append(f'<line x1="{lx}" x2="{lx+12}" y1="{h-9}" y2="{h-9}" class="ref"/><text x="{lx+16}" y="{h-5}" class="axis">{html.escape(ref_label)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def hbars(title, rows, fmt, width=560, marker_label="threshold"):
    """rows: (label, value, marker, ok). One series, marker tick per row."""
    rh, top, left = 34, 16, 150
    h = top + rh * len(rows) + 24
    vmax = max(max(v, m) for _, v, m, _ in rows) * 1.3 or 1
    out = [f'<svg class="chart" viewBox="0 0 {width} {h}" role="img" aria-label="{html.escape(title)}">']
    for i, (lab, v, m, ok) in enumerate(rows):
        y = top + rh * i
        w = (width - left - 90) * v / vmax; mx = left + (width - left - 90) * m / vmax
        out.append(f'<text x="{left-8}" y="{y+19}" class="axis" text-anchor="end">{html.escape(lab)}</text>')
        out.append(f'<rect x="{left}" y="{y+8}" width="{w:.1f}" height="14" rx="4" style="fill:var({"--series-1" if ok else "--status-critical"})"><title>{html.escape(lab)}: {fmt(v)} ({marker_label} {fmt(m)})</title></rect>')
        out.append(f'<line x1="{mx:.1f}" x2="{mx:.1f}" y1="{y+4}" y2="{y+26}" class="ref"/>')
        out.append(f'<text x="{left+w+8:.1f}" y="{y+19}" class="val">{fmt(v)} {"✓" if ok else "✗ fails"}</text>')
    out.append(f'<line x1="{left}" x2="{left+12}" y1="{h-10}" y2="{h-10}" class="ref"/><text x="{left+16}" y="{h-6}" class="axis">{marker_label}</text>')
    out.append("</svg>")
    return "\n".join(out)


def tile(label, value, sub, ok=None):
    cls = "" if ok is None else (" good" if ok else " bad")
    return f'<div class="tile{cls}"><div class="tile-label">{label}</div><div class="tile-value">{value}</div><div class="tile-sub">{sub}</div></div>'


pct = lambda v: f"{v*100:.1f}%"
money = lambda v: f"${v:,.0f}"
ms = lambda v: f"{v:.0f} ms"
auc = metrics["per_head"]
all_pass = all(g.passed for g in guards.values())
example = pm[pm.user_id == users.user_id.iloc[0]]
ex_user = users.iloc[0]
ex_hist = ev[(ev.user_id == ex_user.user_id) & (ev.type == "purchase")].merge(tables["products"][["product_id", "category"]], on="product_id").category.value_counts().to_dict()

page_html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Feed model report</title>
<style>
:root{{color-scheme:light;--surface:#fcfcfb;--card:#ffffff;--border:#e6e5e1;--text:#0b0b0b;--text-2:#52514e;--muted:#8a897f;
--series-1:#2a78d6;--series-2:#eb6834;--status-good:#008300;--status-critical:#e34948;--grid:#ebebe7;}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;--surface:#1a1a19;--card:#232322;--border:#34332f;--text:#f0efec;--text-2:#c3c2b7;--muted:#8d8c83;--series-1:#3987e5;--series-2:#d95926;--status-good:#2fa32f;--status-critical:#e66767;--grid:#2e2e2b;}}}}
:root[data-theme="dark"]{{color-scheme:dark;--surface:#1a1a19;--card:#232322;--border:#34332f;--text:#f0efec;--text-2:#c3c2b7;--muted:#8d8c83;--series-1:#3987e5;--series-2:#d95926;--status-good:#2fa32f;--status-critical:#e66767;--grid:#2e2e2b;}}
body{{margin:0;background:var(--surface);color:var(--text);font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:32px 16px 56px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:17px;margin:36px 0 10px}} p.lead{{color:var(--text-2);margin:0 0 20px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}
.tile{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px}}
.tile-label{{font-size:12px;color:var(--text-2);text-transform:uppercase;letter-spacing:.04em}}
.tile-value{{font-size:28px;font-weight:600;margin:2px 0}} .tile-sub{{font-size:13px;color:var(--text-2)}}
.tile.good .tile-value{{color:var(--status-good)}} .tile.bad .tile-value{{color:var(--status-critical)}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px}}
.card h3{{margin:0 0 4px;font-size:15px}} .card p{{margin:0 0 10px;color:var(--text-2);font-size:13px}}
svg.chart{{width:100%;height:auto;display:block}} .grid{{stroke:var(--grid);stroke-width:1}} .baseline{{stroke:var(--muted);stroke-width:1}}
.axis{{fill:var(--text-2);font-size:11px}} .val{{fill:var(--text);font-size:11px}} .ref{{stroke:var(--text);stroke-width:2;stroke-dasharray:4 3}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--border)}} th{{color:var(--text-2);font-weight:500;white-space:nowrap}}
.note{{font-size:13px;color:var(--text-2)}}
</style></head><body><main>
<h1>Feed model report</h1>
<p class="lead">Model <code>{heavy.version}</code> against two baselines on {N_USERS} simulated users · generated {time.strftime("%Y-%m-%d %H:%M")}</p>

<div class="tiles">
{tile("GMV per user vs random policy", f"{'+' if ab['lift']>=0 else ''}{money(ab['lift'])}", f"95% CI {money(ab['ci_low'])} to {money(ab['ci_high'])} · {'significant' if ab['significant'] else 'not significant'}", ab['significant'] and ab['lift']>0)}
{tile("Purchase AUC (validation)", f"{auc['purchase']['auc']:.3f}", "0.500 = no signal · time-ordered holdout", auc['purchase']['auc']>0.65)}
{tile("Guardrails", "all pass" if all_pass else f"{sum(not g.passed for g in guards.values())} failing", "hide, return, merchant concentration, freshness", all_pass)}
{tile("Latency p95", ms(p95), f"p50 {ms(p50)} · budget {ms(cfg.latency_budget_ms)}", p95 <= cfg.latency_budget_ms)}
</div>

<h2>Does the model beat the baselines?</h2>
<div class="grid2">
<div class="card"><h3>Engagement per card</h3><p>Higher click and purchase, lower hide. Dashed tick = the logging policy the model was trained on.</p>
{grouped_bars("rates", ["click", "purchase", "hide"], [("random policy", "--series-2", [rates['random'][k] for k in ("click","purchase","hide")]), ("model", "--series-1", [rates['model'][k] for k in ("click","purchase","hide")])], pct, ref=[logged[k] for k in ("click","purchase","hide")])}
</div>
<div class="card"><h3>GMV per user</h3><p>Simulated order value summed per user, averaged over the sample.</p>
{grouped_bars("gmv", ["GMV / user"], [("random policy", "--series-2", [gmv['random']]), ("model", "--series-1", [gmv['model']])], money, width=360)}
</div>
</div>

<h2>Did the ranker learn each action?</h2>
<div class="card"><h3>Validation AUC per head</h3><p>Dashed tick at 0.5 is chance. Prevalence shows how rare each label is.</p>
{grouped_bars("auc", [f"{h} ({auc[h]['prevalence']*100:.1f}%)" for h in HEADS], [("AUC", "--series-1", [auc[h]['auc'] for h in HEADS])], lambda v: f"{v:.2f}", ref=[0.5]*len(HEADS), width=760, ref_label="chance (0.5)")}
</div>

<h2>Does it stay inside the rules?</h2>
<div class="grid2">
<div class="card"><h3>Guardrails: model vs random policy</h3><p>Dashed tick = threshold. Fresh share must stay above it; the others below. Return rate is estimated from only {int(pm.purchase.sum())} simulated purchases, so it moves several points between runs; the logged data's own return rate is {pct(logged_return)}.</p>
{grouped_bars("guardrails", [k.replace("_", " ") for k in guards], [("random policy", "--series-2", [guards_r[k].value for k in guards]), ("model", "--series-1", [guards[k].value for k in guards])], pct, ref=[guards[k].threshold for k in guards], ref_label="threshold")}
<p class="note">Model arm: {", ".join(f"{k.replace('_',' ')} {'passes' if g.passed else 'fails'}" for k, g in guards.items())}.</p>
</div>
<div class="card"><h3>Latency by stage, median ms</h3><p>Where a request spends its time. Candidate generation is the first thing to optimise.</p>
{hbars("latency", [(k, float(v), float(v), True) for k, v in stage_med.items()], lambda v: f"{v:.0f} ms", marker_label="median")}
</div>
</div>

<h2>What does one page look like?</h2>
<div class="card"><h3>User {ex_user.user_id} ({ex_user.country})</h3><p>Purchase history: {html.escape(str(ex_hist)) if ex_hist else "none"}. A good page mirrors the history, mixes in fresh and exploration cards, and never stacks one merchant.</p>
<table><tr><th>#</th><th>category</th><th>merchant</th><th>price</th><th>source</th><th>reason</th><th>simulated response</th></tr>
{"".join(f"<tr><td>{r.position}</td><td>{r.category}</td><td>{r.merchant_id}</td><td>${r.price:.0f}</td><td>{r.source}</td><td>{r.reason}</td><td>{', '.join(k for k in ('click','save','atc','purchase','hide') if getattr(r,k)) or '—'}</td></tr>" for r in example.itertuples())}
</table></div>

<p class="note">The simulator replays the synthetic world's own engagement formula, so these numbers show the system mechanics on known signal, not real-user lift. Report built in {time.time()-t0:.0f}s by <code>scripts/report.py</code>.</p>
</main></body></html>"""
out = cfg.root / "reports" / "eval_report.html"
out.write_text(page_html)
print(f"wrote {out}  (model GMV {money(gmv['model'])} vs random {money(gmv['random'])}, p95 {ms(p95)}, guardrails {'pass' if all_pass else 'FAIL'})")
