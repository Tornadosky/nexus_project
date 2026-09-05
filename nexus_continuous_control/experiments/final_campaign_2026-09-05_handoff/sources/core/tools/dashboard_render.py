#!/usr/bin/env python3
"""HTML + CSS + JS for the per-environment results dashboard.

Kept apart from ``build_results_dashboard.py`` so the data work and the presentation work can be
read separately. ``render(data)`` takes the payload that builder assembles and returns one HTML
document; every chart is drawn client-side as SVG from the embedded JSON.

Why SVG-in-JS rather than inline matplotlib PNGs (which is what ``build_dashboard.py`` does):
the charts on that board occlude their own labels -- rotated arm names collide with each other
and with the axis title, and the legend sits inside the plot box on top of the data. Drawing
here means the layout rules are explicit and enforced:

* **Legends are HTML, outside the SVG.** They cannot overlap data, ever.
* **Axis titles and tick labels live in reserved margins** computed before anything is drawn.
* **Categorical comparisons are horizontal bars.** Arm names such as ``nesy·budget8x`` are long;
  putting them on the y-axis gives them a whole gutter instead of a rotated 40px slot.
* **Tick counts are derived from the available width**, so labels thin out rather than collide.
* Charts sit in ``min-width`` scroll containers, so narrowing the window scrolls rather than
  crushing the type.
"""

from __future__ import annotations

import html
import json
from typing import Any

CSS = r"""
:root{
  --ground:#F2F4F7; --surface:#FFFFFF; --sunk:#E9ECF1; --ink:#151A23; --ink2:#3D4756;
  --muted:#6B7686; --rule:#D8DDE5; --rule2:#B9C1CD; --accent:#343C96;
  --ok:#1B6B45; --okb:#DCEFE4; --wait:#7A6413; --waitb:#F4EDD4; --stop:#8A3227; --stopb:#F6E1DD;
  --flat:#7A8493; --neural:#2C5FA8; --nesy:#0F7C82; --symbolic:#B4631B; --ppo:#6B3FA0;
  --sk1:#2C5FA8; --sk2:#B4631B; --sk3:#0F7C82; --sk4:#7B4EA8; --sk5:#8A3227; --sk6:#4E5866;
  --grid:#E3E7EE; --axis:#98A2B1;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif;
  --serif:ui-serif,Georgia,"Times New Roman",serif;
}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0F131A; --surface:#161B23; --sunk:#1E242E; --ink:#E7EBF1; --ink2:#B7C0CD;
  --muted:#8792A2; --rule:#2B323F; --rule2:#3C4553; --accent:#9AA2F0;
  --ok:#6BC79A; --okb:#14301F; --wait:#D6BC58; --waitb:#2E2712; --stop:#E08A7C; --stopb:#331813;
  --flat:#98A2B1; --neural:#6FA0E0; --nesy:#4FBFC4; --symbolic:#E0A063; --ppo:#B78BE8;
  --sk1:#6FA0E0; --sk2:#E0A063; --sk3:#4FBFC4; --sk4:#B78BE8; --sk5:#E08A7C; --sk6:#98A2B1;
  --grid:#262D38; --axis:#5A6474;
}}
:root[data-theme="dark"]{
  --ground:#0F131A; --surface:#161B23; --sunk:#1E242E; --ink:#E7EBF1; --ink2:#B7C0CD;
  --muted:#8792A2; --rule:#2B323F; --rule2:#3C4553; --accent:#9AA2F0;
  --ok:#6BC79A; --okb:#14301F; --wait:#D6BC58; --waitb:#2E2712; --stop:#E08A7C; --stopb:#331813;
  --flat:#98A2B1; --neural:#6FA0E0; --nesy:#4FBFC4; --symbolic:#E0A063; --ppo:#B78BE8;
  --sk1:#6FA0E0; --sk2:#E0A063; --sk3:#4FBFC4; --sk4:#B78BE8; --sk5:#E08A7C; --sk6:#98A2B1;
  --grid:#262D38; --axis:#5A6474;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth; scroll-padding-top:4.2rem}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--serif);line-height:1.62}
.wrap{max-width:1180px;margin:0 auto;padding:clamp(1rem,3.5vw,2.5rem);display:flex;
  flex-direction:column;gap:2.6rem}
h1,h2,h3,h4,h5{font-family:var(--sans);margin:0;text-wrap:balance}
h1{font-size:clamp(1.6rem,4vw,2.4rem);font-weight:660;letter-spacing:-.024em;line-height:1.12}
h2{font-size:1.3rem;font-weight:640;letter-spacing:-.017em}
h3{font-size:1.02rem;font-weight:640}
h4{font-size:.9rem;font-weight:640}
h5{font-size:.8rem;font-weight:620;font-family:var(--sans)}
p{margin:0;max-width:74ch}
ul{margin:0;padding-left:1.15rem;max-width:74ch}
li{margin:.15rem 0}
a{color:var(--accent)}
.eyebrow{font-family:var(--mono);font-size:.66rem;text-transform:uppercase;letter-spacing:.15em;
  color:var(--muted)}
code{font-family:var(--mono);font-size:.85em;background:var(--sunk);padding:.1em .35em;
  border:1px solid var(--rule)}
pre{font-family:var(--mono);font-size:.74rem;line-height:1.5;background:var(--sunk);
  border:1px solid var(--rule);padding:.9rem 1rem;overflow-x:auto;margin:0}

/* ---- top bar ---- */
.topbar{position:sticky;top:0;z-index:40;background:color-mix(in srgb,var(--ground) 92%,transparent);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--rule)}
.topbar-in{max-width:1180px;margin:0 auto;padding:.5rem clamp(1rem,3.5vw,2.5rem);
  display:flex;align-items:center;gap:.6rem;flex-wrap:wrap}
.topbar .brand{font-family:var(--mono);font-size:.7rem;letter-spacing:.13em;text-transform:uppercase;
  color:var(--muted);margin-right:.4rem}
.navlink{font-family:var(--sans);font-size:.76rem;color:var(--ink2);text-decoration:none;
  padding:.24rem .5rem;border:1px solid transparent;white-space:nowrap}
.navlink:hover{border-color:var(--rule2);background:var(--surface)}
.spacer{flex:1}
button.tg{font-family:var(--mono);font-size:.66rem;text-transform:uppercase;letter-spacing:.1em;
  padding:.3rem .6rem;background:var(--surface);color:var(--ink2);border:1px solid var(--rule2);
  cursor:pointer}
button.tg:hover{border-color:var(--ink2)}
button.tg[aria-pressed="true"]{background:var(--ink);color:var(--ground);border-color:var(--ink)}

.masthead{border-top:3px solid var(--ink);padding-top:1.1rem;display:flex;flex-direction:column;gap:.9rem}
.prov{font-family:var(--mono);font-size:.7rem;color:var(--muted);display:flex;flex-wrap:wrap;
  gap:.35rem 1.3rem;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);padding:.6rem 0}
.prov b{color:var(--ink2);font-weight:600}
section{display:flex;flex-direction:column;gap:1rem;scroll-margin-top:4.2rem}
.sec-head{border-bottom:1px solid var(--rule2);padding-bottom:.5rem;display:flex;
  flex-direction:column;gap:.25rem}
.subsec{display:flex;flex-direction:column;gap:.7rem;margin-top:.4rem}
.subsec>h3{border-left:3px solid var(--rule2);padding-left:.55rem}

.chip{display:inline-flex;align-items:center;gap:.4em;font-family:var(--mono);font-size:.64rem;
  text-transform:uppercase;letter-spacing:.09em;font-weight:600;padding:.2em .55em;
  border:1px solid currentColor;white-space:nowrap}
.chip::before{content:"";width:6px;height:6px;background:currentColor}
.chip.ok{color:var(--ok);background:var(--okb)}
.chip.wait{color:var(--wait);background:var(--waitb)}
.chip.stop{color:var(--stop);background:var(--stopb)}
.chip.flat{color:var(--flat)} .chip.neural{color:var(--neural)}
.chip.nesy{color:var(--nesy)} .chip.symbolic{color:var(--symbolic)}

.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule)}
.tile{background:var(--surface);padding:.85rem 1rem;display:flex;flex-direction:column;gap:.2rem}
.tile .k{font-family:var(--mono);font-size:.63rem;text-transform:uppercase;letter-spacing:.1em;
  color:var(--muted)}
.tile .v{font-family:var(--sans);font-size:1.35rem;font-weight:650;letter-spacing:-.03em;
  font-variant-numeric:tabular-nums}
.tile .n{font-size:.75rem;color:var(--muted);font-family:var(--sans);line-height:1.35}
.tile.ok .v{color:var(--ok)} .tile.wait .v{color:var(--wait)} .tile.stop .v{color:var(--stop)}

.scroller{overflow-x:auto;border:1px solid var(--rule);background:var(--surface)}
table{border-collapse:collapse;width:100%;font-family:var(--sans);font-size:.82rem;min-width:600px}
caption{text-align:left;font-family:var(--mono);font-size:.64rem;text-transform:uppercase;
  letter-spacing:.11em;color:var(--muted);padding:.6rem .85rem;border-bottom:1px solid var(--rule);
  background:var(--sunk)}
th,td{text-align:left;padding:.45rem .8rem;border-bottom:1px solid var(--rule);vertical-align:top}
thead th{font-family:var(--mono);font-size:.62rem;text-transform:uppercase;letter-spacing:.09em;
  color:var(--muted);font-weight:600;background:var(--sunk);position:sticky;top:0}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:var(--sunk)}
td.num,th.num{font-variant-numeric:tabular-nums;font-family:var(--mono);font-size:.76rem;
  white-space:nowrap;text-align:right}
td.id{font-family:var(--mono);font-weight:600;font-size:.75rem;white-space:nowrap}
td.path{font-family:var(--mono);font-size:.72rem;color:var(--muted);white-space:normal;
  overflow-wrap:anywhere;max-width:20rem}
td.wrapok{max-width:17rem}
td.wrapok{white-space:normal;font-size:.78rem}
.dim{color:var(--muted)}
.seedlist{font-family:var(--mono);font-size:.68rem;color:var(--muted);padding-right:.4rem}
/* 43 seeds must stay readable without turning one table row into a 40-line tower: the list
   keeps its own horizontal scroll instead of wrapping into the column. */
.seedlist span{display:block;max-width:24rem;overflow-x:auto;white-space:nowrap;
  padding-bottom:.15rem}

/* ---- charts ---- */
.chart{background:var(--surface);border:1px solid var(--rule);padding:.8rem .9rem .9rem;
  display:flex;flex-direction:column;gap:.5rem;min-width:0}
.chart-title{display:flex;flex-direction:column;gap:.1rem}
.chart-title h4{font-size:.86rem}
.chart-title .sub{font-size:.73rem;color:var(--muted);font-family:var(--sans);line-height:1.35}
.legend{display:flex;flex-wrap:wrap;gap:.3rem .85rem;font-family:var(--sans);font-size:.72rem;
  color:var(--ink2);align-items:center}
.legend .it{display:inline-flex;align-items:center;gap:.35rem;white-space:nowrap;cursor:default}
.legend .sw{width:12px;height:3px;border-radius:2px;flex:none}
.legend .sw.box{height:10px;width:10px;border-radius:2px}
.legend .it.off{opacity:.35}
.plotbox{overflow-x:auto;overflow-y:hidden}
.plotbox svg{display:block;height:auto;max-width:100%}
.chart-note{font-size:.72rem;color:var(--muted);font-family:var(--sans);line-height:1.4}
.chart-grid{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(430px,1fr))}
.chart-grid.wide{grid-template-columns:1fr}
.chart-grid.small{grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}
text{font-family:var(--sans)}

/* ---- video cards ---- */
.vidgrid{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.vcard{background:var(--surface);border:1px solid var(--rule);display:flex;flex-direction:column}
.vcard .vhead{padding:.65rem .8rem;border-bottom:1px solid var(--rule);display:flex;
  align-items:center;gap:.5rem;flex-wrap:wrap}
.vcard .vhead h4{flex:1 1 auto;font-size:.88rem}
.vcard video{width:100%;display:block;background:#000}
.vcard img.strip{width:100%;display:block;border-top:1px solid var(--rule)}
.vcard .cap{padding:.35rem .8rem;font-family:var(--sans);font-size:.68rem;color:var(--muted);
  border-top:1px solid var(--rule)}
.vcard .vmeta{padding:.6rem .8rem;font-family:var(--mono);font-size:.7rem;color:var(--muted);
  display:flex;flex-direction:column;gap:.2rem;border-top:1px solid var(--rule)}
.vcard .vmeta b{color:var(--ink2);font-weight:600}
.vcard .warn{padding:.5rem .8rem;font-family:var(--sans);font-size:.72rem;color:var(--stop);
  background:var(--stopb);border-top:1px solid var(--rule)}
.missing{background:var(--surface);border:1px dashed var(--rule2);padding:1rem;color:var(--muted);
  font-family:var(--sans);font-size:.78rem}
.imgcard{background:var(--surface);border:1px solid var(--rule);display:flex;flex-direction:column}
.imgcard img{width:100%;display:block}
.imgcard .cap{padding:.55rem .8rem;font-family:var(--sans);font-size:.74rem;color:var(--muted);
  border-top:1px solid var(--rule)}
.callout{background:var(--sunk);border:1px solid var(--rule);border-left:3px solid var(--accent);
  padding:.8rem 1rem;font-size:.85rem}
.callout.warn{border-left-color:var(--stop)}
.callout.good{border-left-color:var(--ok)}
.callout h4{margin-bottom:.3rem}
.controls{display:flex;gap:.45rem;flex-wrap:wrap;align-items:center;font-family:var(--sans);
  font-size:.74rem;color:var(--muted)}
details{background:var(--surface);border:1px solid var(--rule)}
details>summary{cursor:pointer;padding:.6rem .85rem;font-family:var(--mono);font-size:.68rem;
  text-transform:uppercase;letter-spacing:.09em;color:var(--ink2);background:var(--sunk)}
details>.dbody{padding:.85rem;display:flex;flex-direction:column;gap:.8rem}
footer{border-top:1px solid var(--rule);padding-top:1rem;font-family:var(--mono);font-size:.68rem;
  color:var(--muted)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
summary:focus-visible{outline-offset:-2px}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}
  *{animation-duration:.01ms !important;transition-duration:.01ms !important}}
@media print{.topbar{display:none}}
"""


JS = r"""
'use strict';
const D = window.__DASH__;
const VAR_ORDER = ['flat','neural','nesy','symbolic'];
const VAR_LABEL = {flat:'flat (PQN baseline)', neural:'NEXUS neural', nesy:'NEXUS nesy',
                   symbolic:'NEXUS symbolic'};
function cssVar(n){ return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function vcol(v){ return cssVar('--'+v) || '#888'; }

/* ------------------------------------------------------------------ utils */
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, txt){
  const e = document.createElementNS(NS, tag);
  for (const k in (attrs||{})) if (attrs[k] !== null && attrs[k] !== undefined)
    e.setAttribute(k, attrs[k]);
  if (txt !== undefined && txt !== null) e.textContent = txt;
  return e;
}
function h(tag, cls, txt){
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt !== undefined && txt !== null) e.textContent = txt;
  return e;
}
function fmtSteps(v){
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e9) return (v/1e9).toFixed(a >= 1e10 ? 0 : 1).replace(/\.0$/,'') + 'B';
  if (a >= 1e6) return (v/1e6).toFixed(a >= 1e7 ? 0 : 1).replace(/\.0$/,'') + 'M';
  if (a >= 1e3) return (v/1e3).toFixed(a >= 1e4 ? 0 : 1).replace(/\.0$/,'') + 'k';
  return String(Math.round(v));
}
function fmtNum(v, d){
  if (v === null || v === undefined || !isFinite(v)) return '--';
  if (d !== undefined) return v.toFixed(d);
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(4);
}
/* Axis ticks: decimals chosen from the magnitude, so a zero tick reads "0" and a 2500 tick
   reads "2.5k" instead of both being padded to four decimals. */
function fmtTick(v){
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1000) return fmtSteps(v);
  if (a >= 10) return String(Math.round(v * 10) / 10);
  if (a >= 1) return String(Math.round(v * 100) / 100);
  return String(Math.round(v * 1000) / 1000);
}
/* Plain value formatting for prose/metadata: a zero return should read "0", not "0.0000",
   and a 1164 return should read "1164.4", not "1.2k". */
function fmtVal(v){
  if (v === null || v === undefined || !isFinite(v)) return '--';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}
function niceTicks(lo, hi, want){
  if (!isFinite(lo) || !isFinite(hi)) return [0];
  if (lo === hi){ lo -= 0.5; hi += 0.5; }
  const span = hi - lo;
  const raw = span / Math.max(1, want);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
  const out = [];
  for (let t = Math.ceil(lo/step)*step; t <= hi + step*1e-9; t += step) out.push(+t.toPrecision(12));
  return out.length ? out : [lo, hi];
}
function mean(a){ const b = a.filter(x => x !== null && isFinite(x));
  return b.length ? b.reduce((s,x)=>s+x,0)/b.length : null; }
function std(a){ const b = a.filter(x => x !== null && isFinite(x));
  if (b.length < 2) return 0;
  const m = mean(b); return Math.sqrt(b.reduce((s,x)=>s+(x-m)*(x-m),0)/(b.length-1)); }

/* Chart shell: title + legend (HTML, OUTSIDE the plot) + scrollable svg + note. */
function chartShell(parent, title, sub, note){
  const box = h('div','chart');
  if (title || sub){
    const t = h('div','chart-title');
    if (title) t.appendChild(h('h4', null, title));
    if (sub) t.appendChild(h('div','sub', sub));
    box.appendChild(t);
  }
  const legend = h('div','legend'); box.appendChild(legend);
  const plot = h('div','plotbox'); box.appendChild(plot);
  if (note) box.appendChild(h('div','chart-note', note));
  parent.appendChild(box);
  return {box, legend, plot};
}
function legendItems(legendEl, items){
  legendEl.innerHTML = '';
  items.forEach(it => {
    const e = h('span','it');
    const sw = h('span', 'sw' + (it.box ? ' box' : ''));
    sw.style.background = it.color;
    if (it.dash) sw.style.background =
      'repeating-linear-gradient(90deg,'+it.color+' 0 4px,transparent 4px 7px)';
    e.appendChild(sw); e.appendChild(document.createTextNode(it.label));
    legendEl.appendChild(e);
  });
}

/* ------------------------------------------------------- line chart (x/y) */
/* series: [{label,color,x:[],y:[],lo:[],hi:[],dash}]  Margins are reserved up
   front; nothing is ever drawn into them, so titles/ticks cannot be covered.  */
function lineChart(parent, series, opt){
  opt = opt || {};
  const W = opt.width || 620, H = opt.height || 300;
  const M = {t: 12, r: 14, b: 46, l: 66};
  const sh = chartShell(parent, opt.title, opt.sub, opt.note);
  const live = series.filter(s => s.x && s.x.length);
  if (!live.length){ sh.plot.appendChild(h('div','missing','no data')); return sh; }
  legendItems(sh.legend, live.map(s => ({label: s.label, color: s.color, dash: s.dash})));

  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  live.forEach(s => {
    s.x.forEach(v => { if (isFinite(v)){ xmin = Math.min(xmin,v); xmax = Math.max(xmax,v); } });
    const ys = s.y.concat(s.lo||[], s.hi||[]);
    ys.forEach(v => { if (v !== null && isFinite(v)){ ymin = Math.min(ymin,v); ymax = Math.max(ymax,v); } });
  });
  if (opt.y0) ymin = Math.min(0, ymin);
  if (opt.ymax1) ymax = Math.max(ymax, 1);
  if (!isFinite(ymin)){ ymin = 0; ymax = 1; }
  if (ymin === ymax){ ymin -= 0.5; ymax += 0.5; }
  const pad = (ymax - ymin) * 0.06; ymin -= pad; ymax += pad;
  if (opt.y0 && ymin < 0 && !opt.allowNeg) ymin = Math.min(0, ymin);

  const pw = W - M.l - M.r, ph = H - M.t - M.b;
  const X = v => M.l + (xmax === xmin ? 0 : (v - xmin) / (xmax - xmin)) * pw;
  const Y = v => M.t + ph - (ymax === ymin ? 0 : (v - ymin) / (ymax - ymin)) * ph;

  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, width: '100%', role: 'img',
                         'aria-label': opt.title || 'chart'});
  svg.style.maxWidth = W + 'px'; svg.style.minWidth = (opt.minWidth || 420) + 'px';
  const grid = cssVar('--grid'), axis = cssVar('--axis'), muted = cssVar('--muted');

  const nY = Math.max(3, Math.min(6, Math.round(ph / 46)));
  niceTicks(ymin, ymax, nY).forEach(t => {
    if (t < ymin || t > ymax) return;
    svg.appendChild(el('line', {x1: M.l, x2: M.l + pw, y1: Y(t), y2: Y(t), stroke: grid,
                                'stroke-width': 1}));
    svg.appendChild(el('text', {x: M.l - 8, y: Y(t) + 3.5, 'text-anchor': 'end',
      'font-size': 11, fill: muted}, opt.yfmt ? opt.yfmt(t) : fmtTick(t)));
  });
  const nX = Math.max(2, Math.min(7, Math.floor(pw / 92)));
  niceTicks(xmin, xmax, nX).forEach(t => {
    if (t < xmin || t > xmax) return;
    svg.appendChild(el('line', {x1: X(t), x2: X(t), y1: M.t, y2: M.t + ph, stroke: grid,
                                'stroke-width': 1}));
    svg.appendChild(el('text', {x: X(t), y: M.t + ph + 17, 'text-anchor': 'middle',
      'font-size': 11, fill: muted}, opt.xfmt ? opt.xfmt(t) : fmtSteps(t)));
  });
  svg.appendChild(el('line', {x1: M.l, x2: M.l + pw, y1: M.t + ph, y2: M.t + ph,
                              stroke: axis, 'stroke-width': 1}));
  svg.appendChild(el('line', {x1: M.l, x2: M.l, y1: M.t, y2: M.t + ph, stroke: axis,
                              'stroke-width': 1}));
  if (opt.xlabel) svg.appendChild(el('text', {x: M.l + pw/2, y: H - 8, 'text-anchor': 'middle',
    'font-size': 11.5, fill: muted, 'font-weight': 600}, opt.xlabel));
  if (opt.ylabel) svg.appendChild(el('text', {x: 14, y: M.t + ph/2, 'text-anchor': 'middle',
    'font-size': 11.5, fill: muted, 'font-weight': 600,
    transform: `rotate(-90 14 ${M.t + ph/2})`}, opt.ylabel));

  const clip = 'clip' + Math.random().toString(36).slice(2,8);
  const cp = el('clipPath', {id: clip});
  cp.appendChild(el('rect', {x: M.l, y: M.t, width: pw, height: ph}));
  svg.appendChild(cp);
  const g = el('g', {'clip-path': `url(#${clip})`});

  live.forEach(s => {
    if (s.lo && s.hi){
      const pts = [];
      for (let i = 0; i < s.x.length; i++) if (s.hi[i] !== null && isFinite(s.hi[i]))
        pts.push(`${X(s.x[i]).toFixed(1)},${Y(s.hi[i]).toFixed(1)}`);
      for (let i = s.x.length - 1; i >= 0; i--) if (s.lo[i] !== null && isFinite(s.lo[i]))
        pts.push(`${X(s.x[i]).toFixed(1)},${Y(s.lo[i]).toFixed(1)}`);
      if (pts.length > 2) g.appendChild(el('polygon', {points: pts.join(' '), fill: s.color,
        'fill-opacity': 0.15, stroke: 'none'}));
    }
  });
  live.forEach(s => {
    let d = '', pen = false;
    for (let i = 0; i < s.x.length; i++){
      const v = s.y[i];
      if (v === null || !isFinite(v)){ pen = false; continue; }
      d += (pen ? 'L' : 'M') + X(s.x[i]).toFixed(1) + ' ' + Y(v).toFixed(1) + ' ';
      pen = true;
    }
    if (d) g.appendChild(el('path', {d, fill: 'none', stroke: s.color, 'stroke-width': s.w || 1.9,
      'stroke-linejoin': 'round', 'stroke-linecap': 'round',
      'stroke-dasharray': s.dash ? '5 3' : null, opacity: s.opacity || 1}));
  });
  svg.appendChild(g);
  sh.plot.appendChild(svg);
  return sh;
}

/* --------------------------------------------- horizontal categorical bars */
/* Long arm names get a whole left gutter instead of a rotated slot, so they
   cannot collide with each other or with the axis. Per-seed dots ride on the
   bar's own row.                                                             */
function barsH(parent, items, opt){
  opt = opt || {};
  const sh = chartShell(parent, opt.title, opt.sub, opt.note);
  if (!items.length){ sh.plot.appendChild(h('div','missing','no data')); return sh; }
  const rowH = opt.rowH || 26, gap = 8;
  const W = opt.width || 1040;
  const labelW = Math.min(opt.labelW || 190,
    Math.max(96, 8 + 6.6 * Math.max.apply(null, items.map(i => i.label.length))));
  const valW = 52;                       // reserved gutter for the value labels
  const M = {t: 10, r: valW + 10, b: 40, l: labelW + 10};
  const H = M.t + M.b + items.length * rowH + (items.length - 1) * 0 + gap;
  const pw = W - M.l - M.r, ph = items.length * rowH;
  let vmax = opt.vmax !== undefined ? opt.vmax : 0, vmin = 0;
  items.forEach(i => {
    vmax = Math.max(vmax, i.value || 0, ...(i.dots || []), (i.value||0) + (i.err||0));
    vmin = Math.min(vmin, i.value || 0, ...(i.dots || []));
  });
  if (vmax <= vmin) vmax = vmin + 1;
  const X = v => M.l + (v - vmin) / (vmax - vmin) * pw;
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, width: '100%'});
  svg.style.maxWidth = W + 'px'; svg.style.minWidth = (opt.minWidth || 520) + 'px';
  const grid = cssVar('--grid'), axis = cssVar('--axis'), muted = cssVar('--muted'),
        ink = cssVar('--ink'), ink2 = cssVar('--ink2');
  niceTicks(vmin, vmax, Math.max(2, Math.floor(pw / 88))).forEach(t => {
    if (t < vmin || t > vmax) return;
    svg.appendChild(el('line', {x1: X(t), x2: X(t), y1: M.t, y2: M.t + ph, stroke: grid}));
    svg.appendChild(el('text', {x: X(t), y: M.t + ph + 16, 'text-anchor': 'middle',
      'font-size': 11, fill: muted}, opt.vfmt ? opt.vfmt(t) : fmtTick(t)));
  });
  items.forEach((it, i) => {
    const y = M.t + i * rowH, bh = rowH - 10, cy = y + rowH/2;
    svg.appendChild(el('text', {x: M.l - 10, y: cy + 3.6, 'text-anchor': 'end', 'font-size': 11.5,
      fill: it.dim ? muted : ink2, 'font-weight': it.bold ? 700 : 400,
      'font-family': 'var(--mono)'}, it.label));
    const x0 = X(Math.min(0, it.value)), x1 = X(Math.max(0, it.value));
    const r = el('rect', {x: x0, y: cy - bh/2, width: Math.max(1, x1 - x0), height: bh,
      fill: it.color, 'fill-opacity': it.hatch ? 0.32 : 0.82});
    svg.appendChild(r);
    if (it.hatch) svg.appendChild(el('rect', {x: x0, y: cy - bh/2, width: Math.max(1, x1-x0),
      height: bh, fill: 'none', stroke: it.color, 'stroke-width': 1.2,
      'stroke-dasharray': '3 2'}));
    if (it.err){
      svg.appendChild(el('line', {x1: X(it.value - it.err), x2: X(it.value + it.err),
        y1: cy, y2: cy, stroke: ink, 'stroke-width': 1.1, opacity: .55}));
      [it.value - it.err, it.value + it.err].forEach(v =>
        svg.appendChild(el('line', {x1: X(v), x2: X(v), y1: cy - 4, y2: cy + 4, stroke: ink,
          'stroke-width': 1.1, opacity: .55})));
    }
    (it.dots || []).forEach(d => svg.appendChild(el('circle', {cx: X(d), cy: cy,
      r: 2.5, fill: ink, opacity: .62})));
    svg.appendChild(el('text', {x: W - 6, y: cy + 3.6, 'text-anchor': 'end', 'font-size': 11,
      fill: ink2, 'font-family': 'var(--mono)'},
      opt.vfmt ? opt.vfmt(it.value) : fmtNum(it.value)));
  });
  svg.appendChild(el('line', {x1: X(0), x2: X(0), y1: M.t, y2: M.t + ph, stroke: axis}));
  if (opt.xlabel) svg.appendChild(el('text', {x: M.l + pw/2, y: H - 6, 'text-anchor': 'middle',
    'font-size': 11.5, fill: muted, 'font-weight': 600}, opt.xlabel));
  sh.plot.appendChild(svg);
  return sh;
}

/* ------------------------------------------- vertical grouped bars (few cats) */
/* Used only for <=6 categories, the paper's Fig. 7/8 shape. Category labels are
   wrapped onto up to two lines inside a bottom margin sized for them.         */
function barsGrouped(parent, cats, groups, opt){
  opt = opt || {};
  const sh = chartShell(parent, opt.title, opt.sub, opt.note);
  if (!cats.length){ sh.plot.appendChild(h('div','missing','no data')); return sh; }
  legendItems(sh.legend, groups.map(g => ({label: g.label, color: g.color, box: true})));
  const W = opt.width || 620;
  const wrapped = cats.map(c => c.split(/[\s·]/).filter(Boolean));
  const maxLines = Math.max(...wrapped.map(w => w.length));
  const M = {t: 12, r: opt.r2 ? 58 : 14, b: 26 + maxLines * 13 + (opt.xlabel ? 16 : 0), l: 60};
  const H = opt.height || 290;
  const pw = W - M.l - M.r, ph = H - M.t - M.b;
  const useAxis2 = groups.some(g => g.axis === 2);
  let m1 = 0, m2 = 0;
  groups.forEach(g => g.values.forEach((v, i) => {
    // Per-seed dots are data too: leaving them out of the scale drew them on top of the frame.
    const top = Math.max((v || 0) + (g.err ? (g.err[i] || 0) : 0),
                         ...((g.dots && g.dots[i]) || [0]));
    if (g.axis === 2) m2 = Math.max(m2, top); else m1 = Math.max(m1, top);
  }));
  m1 = m1 || 1; m2 = m2 || 1;
  const Y1 = v => M.t + ph - (v / m1) * ph, Y2 = v => M.t + ph - (v / m2) * ph;
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, width: '100%'});
  svg.style.maxWidth = W + 'px'; svg.style.minWidth = (opt.minWidth || 400) + 'px';
  const grid = cssVar('--grid'), axis = cssVar('--axis'), muted = cssVar('--muted'),
        ink = cssVar('--ink');
  niceTicks(0, m1, 5).forEach(t => {
    if (t > m1) return;
    svg.appendChild(el('line', {x1: M.l, x2: M.l + pw, y1: Y1(t), y2: Y1(t), stroke: grid}));
    svg.appendChild(el('text', {x: M.l - 7, y: Y1(t) + 3.5, 'text-anchor': 'end', 'font-size': 10.5,
      fill: opt.c1 || muted}, opt.v1fmt ? opt.v1fmt(t) : fmtTick(t)));
  });
  if (useAxis2) niceTicks(0, m2, 5).forEach(t => {
    if (t > m2) return;
    svg.appendChild(el('text', {x: M.l + pw + 7, y: Y2(t) + 3.5, 'text-anchor': 'start',
      'font-size': 10.5, fill: opt.c2 || muted}, opt.v2fmt ? opt.v2fmt(t) : fmtTick(t)));
  });
  const cw = pw / cats.length, inner = Math.min(cw * 0.74, 74), bw = inner / groups.length;
  cats.forEach((c, i) => {
    const cx = M.l + cw * (i + 0.5);
    groups.forEach((g, j) => {
      const v = g.values[i];
      if (v === null || v === undefined || !isFinite(v)) return;
      const Yf = g.axis === 2 ? Y2 : Y1;
      const x = cx - inner/2 + j * bw;
      svg.appendChild(el('rect', {x: x + 1, y: Yf(v), width: Math.max(2, bw - 2),
        height: Math.max(1, M.t + ph - Yf(v)), fill: g.color,
        'fill-opacity': g.pale ? 0.4 : 0.85}));
      const e = g.err ? g.err[i] : 0;
      if (e){
        const xc = x + bw/2, cl = q => Math.max(M.t, Math.min(M.t + ph, q));
        svg.appendChild(el('line', {x1: xc, x2: xc, y1: cl(Yf(v - e)), y2: cl(Yf(v + e)),
          stroke: ink, 'stroke-width': 1.1, opacity: .6}));
        svg.appendChild(el('line', {x1: xc - 3.5, x2: xc + 3.5, y1: cl(Yf(v + e)),
          y2: cl(Yf(v + e)), stroke: ink, 'stroke-width': 1.1, opacity: .6}));
      }
      (g.dots ? (g.dots[i] || []) : []).forEach(d => svg.appendChild(el('circle',
        {cx: x + bw/2, cy: Math.max(M.t, Math.min(M.t + ph, Yf(d))), r: 2.2, fill: ink,
         opacity: .55})));
    });
    wrapped[i].forEach((w, k) => svg.appendChild(el('text', {x: cx, y: M.t + ph + 16 + k * 13,
      'text-anchor': 'middle', 'font-size': 11, fill: muted}, w)));
  });
  svg.appendChild(el('line', {x1: M.l, x2: M.l + pw, y1: M.t + ph, y2: M.t + ph, stroke: axis}));
  if (opt.ylabel) svg.appendChild(el('text', {x: 13, y: M.t + ph/2, 'text-anchor': 'middle',
    'font-size': 11.5, fill: opt.c1 || muted, 'font-weight': 600,
    transform: `rotate(-90 13 ${M.t + ph/2})`}, opt.ylabel));
  if (opt.ylabel2) svg.appendChild(el('text', {x: W - 10, y: M.t + ph/2, 'text-anchor': 'middle',
    'font-size': 11.5, fill: opt.c2 || muted, 'font-weight': 600,
    transform: `rotate(90 ${W - 10} ${M.t + ph/2})`}, opt.ylabel2));
  sh.plot.appendChild(svg);
  return sh;
}

/* ------------------------------------------------------------ stacked area */
function stackArea(parent, x, layers, opt){
  opt = opt || {};
  const sh = chartShell(parent, opt.title, opt.sub, opt.note);
  if (!x.length){ sh.plot.appendChild(h('div','missing','no data')); return sh; }
  legendItems(sh.legend, layers.map(l => ({label: l.label, color: l.color, box: true})));
  const W = opt.width || 620, H = opt.height || 240;
  const M = {t: 10, r: 14, b: 42, l: 46};
  const pw = W - M.l - M.r, ph = H - M.t - M.b;
  const xmin = Math.min(...x), xmax = Math.max(...x);
  const X = v => M.l + (xmax === xmin ? 0 : (v - xmin)/(xmax - xmin)) * pw;
  const Y = v => M.t + ph - Math.max(0, Math.min(1, v)) * ph;
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, width: '100%'});
  svg.style.maxWidth = W + 'px'; svg.style.minWidth = (opt.minWidth || 360) + 'px';
  const grid = cssVar('--grid'), axis = cssVar('--axis'), muted = cssVar('--muted');
  [0, .25, .5, .75, 1].forEach(t => {
    svg.appendChild(el('line', {x1: M.l, x2: M.l + pw, y1: Y(t), y2: Y(t), stroke: grid}));
    svg.appendChild(el('text', {x: M.l - 7, y: Y(t) + 3.5, 'text-anchor': 'end', 'font-size': 10.5,
      fill: muted}, (t*100).toFixed(0) + '%'));
  });
  let base = new Array(x.length).fill(0);
  layers.forEach(l => {
    const top = base.map((b, i) => b + (l.y[i] || 0));
    const pts = [];
    for (let i = 0; i < x.length; i++) pts.push(`${X(x[i]).toFixed(1)},${Y(top[i]).toFixed(1)}`);
    for (let i = x.length - 1; i >= 0; i--) pts.push(`${X(x[i]).toFixed(1)},${Y(base[i]).toFixed(1)}`);
    svg.appendChild(el('polygon', {points: pts.join(' '), fill: l.color, 'fill-opacity': .78,
      stroke: l.color, 'stroke-width': .6}));
    base = top;
  });
  niceTicks(xmin, xmax, Math.max(2, Math.floor(pw/92))).forEach(t => {
    if (t < xmin || t > xmax) return;
    svg.appendChild(el('text', {x: X(t), y: M.t + ph + 16, 'text-anchor': 'middle',
      'font-size': 10.5, fill: muted}, fmtSteps(t)));
  });
  svg.appendChild(el('line', {x1: M.l, x2: M.l + pw, y1: M.t + ph, y2: M.t + ph, stroke: axis}));
  if (opt.xlabel) svg.appendChild(el('text', {x: M.l + pw/2, y: H - 6, 'text-anchor': 'middle',
    'font-size': 11, fill: muted, 'font-weight': 600}, opt.xlabel));
  sh.plot.appendChild(svg);
  return sh;
}

/* ------------------------------------------------------------- dumbbell */
/* Gate summary: one row per env, flat vs best hierarchical. Env names sit in
   the left gutter -- this is the chart that replaces the occluded matrix.    */
function dumbbell(parent, rows, opt){
  opt = opt || {};
  const sh = chartShell(parent, opt.title, opt.sub, opt.note);
  legendItems(sh.legend, [
    {label: 'best flat (baseline)', color: vcol('flat'), box: true},
    {label: 'best hierarchical arm', color: vcol('nesy'), box: true}]);
  const W = opt.width || 1040, rowH = 32;
  const labelW = Math.max(110, 7.1 * Math.max.apply(null, rows.map(r => r.label.length)));
  const M = {t: 12, r: 22, b: 40, l: labelW + 12};
  const H = M.t + M.b + rows.length * rowH;
  const pw = W - M.l - M.r, ph = rows.length * rowH;
  let vmax = 0;
  rows.forEach(r => { vmax = Math.max(vmax, r.flat, r.hier, ...(r.flatSeeds||[]), ...(r.hierSeeds||[])); });
  vmax = Math.max(vmax * 1.08, 0.05);
  const X = v => M.l + (v / vmax) * pw;
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, width: '100%'});
  svg.style.maxWidth = W + 'px'; svg.style.minWidth = (opt.minWidth || 470) + 'px';
  const grid = cssVar('--grid'), axis = cssVar('--axis'), muted = cssVar('--muted'),
        ink2 = cssVar('--ink2');
  niceTicks(0, vmax, Math.max(2, Math.floor(pw/86))).forEach(t => {
    if (t > vmax) return;
    svg.appendChild(el('line', {x1: X(t), x2: X(t), y1: M.t, y2: M.t + ph, stroke: grid}));
    svg.appendChild(el('text', {x: X(t), y: M.t + ph + 16, 'text-anchor': 'middle',
      'font-size': 11, fill: muted}, fmtTick(t)));
  });
  rows.forEach((r, i) => {
    const cy = M.t + i * rowH + rowH/2;
    svg.appendChild(el('text', {x: M.l - 12, y: cy + 3.8, 'text-anchor': 'end', 'font-size': 11.5,
      fill: ink2, 'font-family': 'var(--mono)'}, r.label));
    (r.flatSeeds||[]).forEach(v => svg.appendChild(el('circle', {cx: X(v), cy: cy - 6, r: 2,
      fill: vcol('flat'), opacity: .38})));
    (r.hierSeeds||[]).forEach(v => svg.appendChild(el('circle', {cx: X(v), cy: cy + 6, r: 2,
      fill: vcol('nesy'), opacity: .38})));
    svg.appendChild(el('line', {x1: X(Math.min(r.flat, r.hier)), x2: X(Math.max(r.flat, r.hier)),
      y1: cy, y2: cy, stroke: r.hier > r.flat ? vcol('nesy') : vcol('symbolic'),
      'stroke-width': 2.4, opacity: .55}));
    svg.appendChild(el('circle', {cx: X(r.flat), cy: cy, r: 5, fill: vcol('flat')}));
    svg.appendChild(el('circle', {cx: X(r.hier), cy: cy, r: 5, fill: vcol('nesy')}));
  });
  svg.appendChild(el('line', {x1: M.l, x2: M.l + pw, y1: M.t + ph, y2: M.t + ph, stroke: axis}));
  if (opt.xlabel) svg.appendChild(el('text', {x: M.l + pw/2, y: H - 6, 'text-anchor': 'middle',
    'font-size': 11.5, fill: muted, 'font-weight': 600}, opt.xlabel));
  sh.plot.appendChild(svg);
  return sh;
}

/* ------------------------------------------------------------ aggregation */
function runsOf(env, arm){
  const c = D.cells[env + '|' + arm];
  return c ? c.runs.map(k => D.runs[k]).filter(Boolean) : [];
}
/* Mean +/- 1 std across seeds on a shared x grid. Runs in one arm share a
   budget and a logging cadence, so index alignment is exact; a run of a
   different length is dropped rather than stretched.                         */
function aggregate(rs, pick){
  if (!rs.length) return null;
  const lens = rs.map(r => (pick(r) || []).length).filter(n => n > 0);
  if (!lens.length) return null;
  const n = Math.max(...lens.sort((a,b)=>a-b).slice(0, Math.ceil(lens.length/2) || 1));
  const use = rs.filter(r => (pick(r) || []).length === n);
  if (!use.length) return null;
  const x = use[0].x.length === n ? use[0].x : use[0].x.slice(0, n);
  const y = [], lo = [], hi = [];
  for (let i = 0; i < n; i++){
    const col = use.map(r => pick(r)[i]).filter(v => v !== null && isFinite(v));
    if (!col.length){ y.push(null); lo.push(null); hi.push(null); continue; }
    const m = mean(col), s = std(col);
    y.push(m); lo.push(m - s); hi.push(m + s);
  }
  return {x, y, lo, hi, n: use.length};
}
"""

JS2 = r"""
/* ===================================================================== page */
function shippedArm(env, v){ return (D.shipped[env] || {})[v] || null; }
function shippedArms(env){
  const out = [];
  VAR_ORDER.forEach(v => { const a = shippedArm(env, v); if (a) out.push({variant: v, arm: a}); });
  return out;
}
function cellOf(env, arm){ return D.cells[env + '|' + arm]; }
function budgetNote(env){
  const bs = shippedArms(env).map(({variant, arm}) => {
    const c = cellOf(env, arm); return {variant, arm, steps: c ? c.steps : null};
  });
  const set = new Set(bs.map(b => b.steps));
  if (set.size <= 1) return null;
  return bs.map(b => `${b.arm} ${b.steps === null ? '(mixed)' : fmtSteps(b.steps)}`).join('  ·  ');
}

/* ---- Fig. 7 analogue: environment return + aligned goal metric, side by side */
function figPerformance(parent, env){
  const arms = shippedArms(env);
  if (!arms.length) return;
  const cats = arms.map(a => VAR_LABEL[a.variant].replace('NEXUS ','').replace(' (PQN baseline)',''));
  const cells = arms.map(a => cellOf(env, a.arm));
  const g1 = {label: 'episode return (training, tail-mean)', color: cssVar('--accent'),
    values: cells.map(c => c.return_mean), axis: 1, pale: false,
    err: cells.map(c => std(c.return.filter(v => v !== null))),
    dots: cells.map(c => c.return.filter(v => v !== null))};
  const g2 = {label: D.env_meta[env].success + ' (aligned goal)', color: vcol('symbolic'),
    values: cells.map(c => c.success_mean), axis: 2,
    err: cells.map(c => c.success_std || 0),
    dots: cells.map(c => c.success.filter(v => v !== null))};
  barsGrouped(parent, cats, [g1, g2], {
    title: 'Return vs aligned goal metric — paper Fig. 7 analogue',
    sub: 'Left axis: environment return. Right axis: the goal-grounded success metric the '
       + 'campaign is scored on. Black bars are ±1σ over seeds; dots are individual seeds.',
    ylabel: 'episode return', ylabel2: D.env_meta[env].success_short || 'success rate',
    c1: cssVar('--accent'), c2: vcol('symbolic'), r2: true,
    v1fmt: v => fmtSteps(v), v2fmt: v => fmtNum(v, 2), width: 780, height: 320,
    note: 'Return and success can disagree: panda earns high return with near-zero lift success, '
        + 'so the two axes are shown together rather than one standing in for the other.'
  });
}

/* ---- arm ladder: every arm on the env, experimental ones hatched ---------- */
function figArmLadder(parent, env, showExp){
  const arms = Object.values(D.cells).filter(c => c.env === env)
    .filter(c => showExp || !c.experimental)
    .sort((a, b) => (VAR_ORDER.indexOf(a.variant) - VAR_ORDER.indexOf(b.variant))
                 || a.arm.localeCompare(b.arm));
  const shipped = new Set(shippedArms(env).map(a => a.arm));
  const items = arms.map(c => ({
    label: c.arm + '  n=' + c.n,
    value: c.success_mean === null ? 0 : c.success_mean,
    err: c.success_std || 0,
    dots: c.success.filter(v => v !== null),
    color: vcol(c.variant),
    hatch: c.experimental,
    bold: shipped.has(c.arm),
    dim: c.experimental
  }));
  barsH(parent, items, {
    title: 'Every arm on this environment — ' + D.env_meta[env].success,
    sub: 'Bar = mean over seeds, whisker = ±1σ, dots = per-seed values. Dashed/faded bars are '
       + 'experimental arms (changed budget or hyperparameter); they are excluded from the gate '
       + 'and must not be compared against a shipped baseline.',
    xlabel: 'primary success rate (training tail-mean, last 10% of logged updates)',
    vfmt: v => fmtNum(v, 2), rowH: 25, labelW: 230,
    note: 'Where the dots span the bar, the seeds disagree — read the spread, not the mean.'
  });
}

/* ---- training curves ------------------------------------------------------ */
function figCurves(parent, env){
  const arms = shippedArms(env);
  const mk = key => arms.map(a => {
    const agg = aggregate(runsOf(env, a.arm), r => r.curves[key]);
    if (!agg) return null;
    return {label: VAR_LABEL[a.variant] + ' · ' + a.arm + ' (n=' + agg.n + ')',
            color: vcol(a.variant), x: agg.x, y: agg.y, lo: agg.lo, hi: agg.hi};
  }).filter(Boolean);
  const grid = h('div','chart-grid'); parent.appendChild(grid);
  lineChart(grid, mk('policy_diag/primary_success_rate'), {
    title: 'Primary success rate during training',
    sub: D.env_meta[env].success + '. Line = mean over seeds, band = ±1σ.',
    xlabel: 'environment steps', ylabel: 'success rate', y0: true,
    yfmt: v => fmtNum(v, 2),
    note: 'Logged with ε-greedy exploration still active — the greedy numbers are in the '
        + 'evaluation table below and can differ materially.'});
  lineChart(grid, mk('rollout/episode_return'), {
    title: 'Episode return during training',
    sub: 'Environment reward per episode. Line = mean over seeds, band = ±1σ.',
    xlabel: 'environment steps', ylabel: 'episode return', y0: true});
  lineChart(grid, mk('rollout/episode_length'), {
    title: 'Episode length during training',
    sub: 'Steps survived before termination. On the environments that can fall over, this is '
       + 'where a collapse shows up before the return does.',
    xlabel: 'environment steps', ylabel: 'steps per episode', y0: true});
  const mv = mk('mask/violation_rate').filter(s => s.y.some(v => v));
  if (mv.length) lineChart(grid, mv, {
    title: 'Rule-mask violation rate',
    sub: 'Fraction of decisions where the selected skill was not admitted by the rules. For '
       + 'nesy this should sit at zero by construction; anything else is a defect in the mask.',
    xlabel: 'environment steps', ylabel: 'violation rate', y0: true, yfmt: v => fmtNum(v, 3)});
}

/* ---- Fig. 3/4/10 analogue: skill returns, one panel per skill ------------- */
function figSkillReturns(parent, env){
  const arms = shippedArms(env).filter(a => a.variant !== 'flat');
  if (!arms.length) return;
  const skills = (cellOf(env, arms[0].arm) || {}).skills || [];
  if (!skills.length) return;
  const grid = h('div','chart-grid small'); parent.appendChild(grid);
  skills.forEach(sk => {
    const series = arms.map(a => {
      const agg = aggregate(runsOf(env, a.arm), r => r.skill_return[sk]);
      if (!agg) return null;
      return {label: VAR_LABEL[a.variant].replace('NEXUS ',''), color: vcol(a.variant),
              x: agg.x, y: agg.y, lo: agg.lo, hi: agg.hi};
    }).filter(Boolean);
    lineChart(grid, series, {title: sk.replace(/^\d+_/,'').replace(/_/g,' '),
      sub: 'skill ' + sk.split('_')[0] + ' · skill-specific reward accumulated per episode',
      xlabel: 'environment steps', ylabel: 'skill return', width: 470, height: 250,
      minWidth: 320});
  });
  const note = h('div','chart-note');
  note.textContent = 'Paper Fig. 3/4/10 analogue. The flat baseline is absent by construction: '
    + 'it logs only skill_return/0_flat_actor, so the paper’s "baselines collapse onto one '
    + 'skill" comparison cannot be drawn from these checkpoints — only the hierarchical arms can '
    + 'be compared against each other.';
  parent.appendChild(note);
}

/* ---- skill usage --------------------------------------------------------- */
/* Resolved per call, not once at load: the theme toggle rebuilds the charts and a const
   captured under the old palette would repaint every stack in the wrong theme. */
const SKILL_TOKENS = ['--sk1','--sk2','--sk3','--sk4','--sk5','--sk6'];
function skillColor(i){ return cssVar(SKILL_TOKENS[i % SKILL_TOKENS.length]); }
function figSkillUsage(parent, env){
  const arms = shippedArms(env).filter(a => a.variant !== 'flat');
  if (!arms.length) return;
  const grid = h('div','chart-grid small'); parent.appendChild(grid);
  arms.forEach(a => {
    const rs = runsOf(env, a.arm);
    const skills = (cellOf(env, a.arm) || {}).skills || [];
    if (!skills.length) return;
    const aggs = skills.map(sk => aggregate(rs, r => r.skill_usage[sk]));
    if (aggs.some(x => !x)) return;
    const x = aggs[0].x;
    stackArea(grid, x, skills.map((sk, i) => ({label: sk.replace(/^\d+_/,'').replace(/_/g,' '),
      color: skillColor(i), y: aggs[i].y})), {
      title: VAR_LABEL[a.variant] + ' — skill usage over training',
      sub: 'share of decisions per skill, mean over ' + aggs[0].n + ' seeds',
      xlabel: 'environment steps', width: 470, height: 230, minWidth: 320});
  });
}

/* ---- nesy mask ----------------------------------------------------------- */
function figMask(parent, env){
  const a = shippedArm(env, 'nesy');
  if (!a) return;
  const rs = runsOf(env, a);
  if (!rs.length) return;
  const skills = (cellOf(env, a) || {}).skills || [];
  const avail = skills.map(sk => mean(rs.map(r => (r.mask_available_final || {})[sk])
    .filter(v => v !== null && v !== undefined)));
  const used = skills.map(sk => mean(rs.map(r => (r.skill_usage_final || {})[sk])
    .filter(v => v !== null && v !== undefined)));
  if (avail.every(v => v === null)) return;
  const cats = skills.map(s => s.replace(/^\d+_/,'').replace(/_/g,' '));
  barsGrouped(parent, cats, [
    {label: 'admitted by the mask', color: cssVar('--muted'), values: avail, pale: true, axis: 1},
    {label: 'actually selected', color: vcol('nesy'), values: used, axis: 1}], {
    title: 'NeSy mask — what the rules admit vs what the meta-Q chooses',
    sub: 'Tail-mean over the last 10% of training, averaged over ' + rs.length + ' seed(s) of '
       + a + '. A skill admitted often but never chosen is a rule the value function overrides.',
    ylabel: 'fraction of decision steps', v1fmt: v => fmtNum(v, 2), height: 270});
}

/* ---- videos -------------------------------------------------------------- */
function figVideos(parent, env){
  const vids = D.videos.filter(v => v.env === env && !v.duplicate_hidden);
  if (!vids.length){
    parent.appendChild(h('div','missing','No rollout video has been rendered for this '
      + 'environment. Nothing is substituted from another environment.'));
    return;
  }
  const grid = h('div','vidgrid'); parent.appendChild(grid);
  // Shipped arms first, then the experimental clips, so the default reading order is the
  // comparison the gate actually scores.
  vids.sort((a, b) => (a.experimental - b.experimental)
                   || (VAR_ORDER.indexOf(a.variant) - VAR_ORDER.indexOf(b.variant))
                   || a.arm.localeCompare(b.arm)
                   || (a.render_seed - b.render_seed));
  vids.forEach(v => {
    const card = h('div','vcard');
    const head = h('div','vhead');
    head.appendChild(h('h4', null, VAR_LABEL[v.variant] || v.variant));
    const chip = h('span','chip ' + v.variant); chip.textContent = v.arm || v.variant;
    head.appendChild(chip);
    if (vids.filter(o => o.checkpoint === v.checkpoint).length > 1){
      const rc = h('span','chip'); rc.style.color = cssVar('--muted');
      rc.textContent = 'render r' + v.render_seed;
      head.appendChild(rc);
    }
    /* Clips in one section do not all play at the same rate -- some were frame-subsampled to
       fit a published page. Put the rate where someone comparing two clips will see it. */
    if (v.speedup){
      const sc = h('span','chip'); sc.style.color = cssVar('--wait');
      sc.textContent = v.speedup + '× speed';
      head.appendChild(sc);
    }
    card.appendChild(head);
    const vid = document.createElement('video');
    vid.controls = true; vid.loop = true; vid.muted = true; vid.preload = 'metadata';
    vid.playsInline = true;
    /* `#t=0.1` makes the browser seek to a real frame on load, so the card shows the scene
       instead of a black rectangle. A poster image is not used: the only still we have is the
       contact sheet, and passing a montage off as the first frame would misread as the video. */
    const src = document.createElement('source');
    src.src = v.video + (v.video.startsWith('data:') ? '' : '#t=0.1');
    src.type = 'video/mp4';
    vid.appendChild(src);
    card.appendChild(vid);
    if (v.strip){
      const im = document.createElement('img');
      im.className = 'strip'; im.src = v.strip; im.loading = 'lazy';
      im.alt = 'skill activation timeline';
      card.appendChild(im);
      card.appendChild(h('div','cap','skill selected at every step of this episode'));
    }
    if (v.frame){
      const im = document.createElement('img');
      im.className = 'strip'; im.src = v.frame; im.loading = 'lazy';
      im.alt = 'frames sampled across the episode';
      card.appendChild(im);
      card.appendChild(h('div','cap','contact sheet — frames sampled across the same episode'));
    }
    const meta = h('div','vmeta');
    const rows = [
      ['checkpoint', v.checkpoint],
      ['training seed', v.train_seed === null ? 'unknown' : ('s' + v.train_seed)],
      ['render seed', 'r' + v.render_seed],
      ['episode', fmtNum(v.frames, 0) + ' env steps · return ' + fmtVal(v.episode_return)
        + (v.speedup ? '  (clip plays at ~' + v.speedup + '×)' : '')],
    ];
    rows.forEach(([k, val]) => {
      const r = h('div'); const b = h('b', null, k + ': '); r.appendChild(b);
      r.appendChild(document.createTextNode(val)); meta.appendChild(r);
    });
    if (v.skill_names && v.skill_names.length > 1){
      const r = h('div'); r.appendChild(h('b', null, 'usage this episode: '));
      r.appendChild(document.createTextNode(v.skill_names
        .map((s, i) => s + ' ' + (100 * (v.skill_usage[i] || 0)).toFixed(0) + '%').join(', ')));
      meta.appendChild(r);
    }
    card.appendChild(meta);
    if (v.warnings && v.warnings.length){
      const w = h('div','warn'); w.textContent = v.warnings.join(' — ');
      card.appendChild(w);
    }
    grid.appendChild(card);
  });
}

/* ---- robustness (paper Fig. 8 analogue) ---------------------------------- */
function figRobustness(parent, env){
  /* The sweeps were queued per checkpoint, not per shipped arm: HopperHop's sweeps ran on
     nesy·v2 while the budget-matched headline arm is nesy·matched. Rather than drop the data,
     fall back to whichever non-experimental arm of that variant was actually swept, and name
     it in the legend so the reader is never guessing which arm a curve belongs to. */
  const variants = VAR_ORDER.map(v => {
    const cands = Object.values(D.cells)
      .filter(c => c.env === env && c.variant === v && !c.experimental)
      .map(c => ({arm: c.arm, rs: runsOf(env, c.arm).filter(r => D.robustness[r.key])}))
      .filter(c => c.rs.length)
      .sort((x, y) => y.rs.length - x.rs.length);
    return cands.length ? {variant: v, arm: cands[0].arm, rs: cands[0].rs} : null;
  }).filter(Boolean);
  const swapped = variants.filter(v => v.arm !== shippedArm(env, v.variant));
  const series = variants.map(a => {
    const rs = a.rs;
    const levels = D.robustness[rs[0].key].levels;
    const y = levels.map((_, i) => mean(rs.map(r => D.robustness[r.key].success[i])));
    const lo = levels.map((_, i) => {
      const col = rs.map(r => D.robustness[r.key].success[i]);
      return mean(col) - std(col); });
    const hi = levels.map((_, i) => {
      const col = rs.map(r => D.robustness[r.key].success[i]);
      return mean(col) + std(col); });
    return {label: VAR_LABEL[a.variant] + ' · ' + a.arm + ' (n=' + rs.length + ')',
            color: vcol(a.variant), x: levels, y, lo, hi};
  }).filter(Boolean);
  if (!series.length) return;
  lineChart(parent, series, {
    title: 'Robustness to action noise — paper Fig. 8 analogue',
    sub: 'Deterministic (greedy) evaluation, 64 episodes per point, no retraining. '
       + 'Band = ±1σ over seeds.',
    xlabel: 'action-noise level', ylabel: 'success rate', y0: true,
    xfmt: v => fmtNum(v, 2), yfmt: v => fmtNum(v, 2),
    note: 'The paper perturbs by simplifying the game; the continuous analogue available here is '
        + 'action noise. The zero-noise point is the honest greedy number for each arm.'
        + (swapped.length ? '  Sweeps for ' + swapped.map(v => v.variant + ' ran on ' + v.arm
            + ', not the headline arm ' + (shippedArm(env, v.variant) || 'n/a')).join('; ')
            + ' — read this panel against the arm named in the legend.' : '')});
}

/* ---- external PPO baseline ----------------------------------------------- */
/* PPO has no training curve in our metric schema, so it appears only here, against the
   deterministic evaluations -- which came out of the same harness and are the only numbers it
   can honestly be put next to. */
function figPPO(parent, env){
  const rows = D.ppo[env];
  if (!rows || !rows.length) return;
  const items = [];
  shippedArms(env).forEach(({variant, arm}) => {
    const c = cellOf(env, arm);
    if (!c || !c.det_n) return;
    items.push({label: arm + '  n=' + c.det_n, value: c.det_success_mean,
                color: vcol(variant), bold: true});
  });
  const byArm = {};
  rows.forEach(r => { (byArm[r.arm] = byArm[r.arm] || []).push(r); });
  Object.keys(byArm).sort().forEach(arm => {
    const rs = byArm[arm];
    items.push({label: arm + '  n=' + rs.length, value: mean(rs.map(r => r.success)),
                err: std(rs.map(r => r.success)), dots: rs.map(r => r.success),
                color: cssVar('--ppo'), bold: true});
  });
  if (items.length < 2) return;
  barsH(parent, items, {
    title: 'Against an external baseline — Brax PPO',
    sub: 'Greedy evaluation only, from the same harness for every bar. The paper carries a PPO '
       + 'baseline alongside PQN; we have one on this environment. Training tail-means are '
       + 'deliberately absent here — they are not comparable to these numbers.',
    xlabel: 'primary success rate (greedy evaluation)', vfmt: v => fmtNum(v, 3),
    rowH: 26, labelW: 210});
  if (D.ppo_note[env]){
    const c = h('div','callout warn');
    c.appendChild(h('h4', null, 'What this does to the conclusion'));
    const p = document.createElement('p'); p.innerHTML = D.ppo_note[env];
    c.appendChild(p);
    parent.appendChild(c);
  }
}

/* ---- go1 out-of-distribution commands ------------------------------------ */
const OOS_LABEL = {indist: 'in-distribution', cmd00: 'zero command (simplified)',
  cmd15: 'command ×1.5', cmd20: 'command ×2', rough0: 'rough terrain',
  flat0: 'flat terrain'};
function figOOS(parent, env, prefix){
  const conds = D.oos[prefix];
  if (!conds) return;
  const order = ['indist','cmd00','cmd15','cmd20','rough0','flat0']
    .filter(c => conds[c] && conds[c].length);
  if (order.length < 2) return;
  const variants = VAR_ORDER.filter(v => order.some(c => conds[c].some(r => r.variant === v)));
  const cats = variants.map(v => VAR_LABEL[v].replace('NEXUS ','').replace(' (PQN baseline)',''));
  // The conditions are a scale (zero command -> x2) plus a terrain swap, so they get their own
  // colours rather than five shades of one. In-distribution stays grey: it is the reference.
  const COND_COLOR = {indist: cssVar('--muted'), cmd00: cssVar('--sk2'),
                      cmd15: cssVar('--sk4'), cmd20: cssVar('--sk5'),
                      rough0: cssVar('--sk3'), flat0: cssVar('--sk1')};
  const groups = order.map((c, i) => ({
    label: OOS_LABEL[c] || c, color: COND_COLOR[c] || vcol('symbolic'),
    pale: c === 'indist',
    values: variants.map(v => {
      const rr = conds[c].filter(r => r.variant === v);
      return rr.length ? mean(rr.map(r => r.success)) : null; }),
    err: variants.map(v => {
      const rr = conds[c].filter(r => r.variant === v);
      return rr.length ? std(rr.map(r => r.success)) : 0; })
  }));
  barsGrouped(parent, cats, groups, {
    title: 'Held-out command conditions — paper Fig. 8 analogue (zero-shot)',
    sub: 'Trained once, evaluated without retraining under changed command ranges. '
       + '64 episodes per cell, ±1σ over 3 seeds.',
    ylabel: 'tracking success rate', v1fmt: v => fmtNum(v, 2), height: 300, width: 900,
    note: 'The in-distribution bar is the reference; every other bar is the same checkpoint '
        + 'evaluated on a command range it never trained on. Whiskers are clipped at the axis.'});
}

/* ---- per-arm table ------------------------------------------------------- */
function armTable(parent, env, showExp){
  const arms = Object.values(D.cells).filter(c => c.env === env)
    .filter(c => showExp || !c.experimental)
    .sort((a,b) => (VAR_ORDER.indexOf(a.variant) - VAR_ORDER.indexOf(b.variant))
                || a.arm.localeCompare(b.arm));
  const sc = h('div','scroller');
  const t = document.createElement('table');
  t.innerHTML = '<caption>Every arm: seeds, budget, training tail-mean and greedy eval</caption>'
    + '<thead><tr><th>arm</th><th class="num">n</th><th class="num">env steps</th>'
    + '<th class="num">success mean</th><th class="num">sd</th><th class="num">min</th>'
    + '<th class="num">max</th><th class="num">return</th><th class="num">greedy success</th>'
    + '<th>seeds</th></tr></thead>';
  const tb = document.createElement('tbody');
  arms.forEach(c => {
    const tr = document.createElement('tr');
    const nm = h('td','id'); nm.textContent = c.arm;
    nm.style.color = vcol(c.variant);
    if (c.experimental) nm.appendChild(h('span','dim',' exp'));
    tr.appendChild(nm);
    const cells = [
      [c.n, 0], [c.steps === null ? 'mixed' : fmtSteps(c.steps), null],
      [fmtNum(c.success_mean, 4), null], [c.success_std === null ? '--' : fmtNum(c.success_std, 4), null],
      [fmtNum(c.success_min, 4), null], [fmtNum(c.success_max, 4), null],
      [fmtNum(c.return_mean, 1), null],
      [c.det_n ? fmtNum(c.det_success_mean, 4) + ' (n=' + c.det_n + ')' : '--', null]];
    cells.forEach(([v]) => { const td = h('td','num'); td.textContent = v; tr.appendChild(td); });
    const sd = h('td','seedlist');
    sd.appendChild(h('span', null,
      c.seeds.map((s, i) => 's' + s + '=' + fmtNum(c.success[i], 3)).join('  ')));
    tr.appendChild(sd);
    tb.appendChild(tr);
  });
  t.appendChild(tb); sc.appendChild(t); parent.appendChild(sc);
}

/* ---- one environment ----------------------------------------------------- */
function buildEnv(env){
  const meta = D.env_meta[env];
  const sec = h('section'); sec.id = 'env-' + env;
  const head = h('div','sec-head');
  head.appendChild(h('span','eyebrow', meta.family));
  const hrow = h('div'); hrow.style.display = 'flex';
  hrow.style.alignItems = 'baseline'; hrow.style.gap = '.7rem'; hrow.style.flexWrap = 'wrap';
  hrow.appendChild(h('h2', null, env));
  const g = D.gate.per_env[env];
  if (g){
    const st = g.status === 'beats_flat' ? (g.separated ? 'ok' : 'wait') : 'stop';
    const chip = h('span','chip ' + st);
    chip.textContent = g.status === 'beats_flat'
      ? (g.separated ? 'hierarchy separated' : 'hierarchy ahead on the mean only')
      : 'flat baseline ahead';
    hrow.appendChild(chip);
  }
  head.appendChild(hrow);
  head.appendChild(h('p', null, meta.task + ' Primary success metric: ' + meta.success + '.'));
  sec.appendChild(head);

  const bn = budgetNote(env);
  if (bn){
    const c = h('div','callout warn');
    c.appendChild(h('h4', null, 'The shipped arms here are not all on the same budget'));
    c.appendChild(h('p', null, 'Environment steps per arm — ' + bn + '. A comparison across '
      + 'different budgets is not a result unless the mismatch runs in the baseline’s '
      + 'favour, and it is labelled as mismatched wherever it appears.'));
    sec.appendChild(c);
  }

  const ctl = h('div','controls');
  const btn = document.createElement('button');
  btn.className = 'tg'; btn.setAttribute('aria-pressed','false');
  btn.textContent = 'show experimental arms';
  ctl.appendChild(btn);
  ctl.appendChild(h('span', null, 'experimental = budget-scaled or hyperparameter-changed arms, '
    + 'excluded from the gate'));
  sec.appendChild(ctl);

  const sub = (title, note) => {
    const d = h('div','subsec');
    d.appendChild(h('h3', null, title));
    if (note) d.appendChild(h('p', null, note));
    sec.appendChild(d); return d;
  };

  figPerformance(sub('Headline performance'), env);
  const ladderBox = sub('Arm ladder');
  const tableBox = sub('All arms, all seeds');
  const redraw = () => {
    const on = btn.getAttribute('aria-pressed') === 'true';
    ladderBox.querySelectorAll('.chart').forEach(e => e.remove());
    tableBox.querySelectorAll('.scroller').forEach(e => e.remove());
    figArmLadder(ladderBox, env, on);
    armTable(tableBox, env, on);
  };
  btn.addEventListener('click', () => {
    btn.setAttribute('aria-pressed', btn.getAttribute('aria-pressed') === 'true' ? 'false' : 'true');
    redraw();
  });
  redraw();

  figCurves(sub('Training curves'), env);
  figSkillReturns(sub('Skill returns — what each skill actually earned'), env);
  figSkillUsage(sub('Skill usage'), env);
  figMask(sub('Rule mask vs learned choice'), env);
  figVideos(sub('Rollouts', 'One greedy episode per arm, rendered from the checkpoint named on '
    + 'each card. The strip under each clip is the skill the meta-policy selected at every step. '
    + 'A single episode is an anecdote, not a measurement — the numbers above are the measurement.'),
    env);
  if (D.ppo[env]) figPPO(sub('External baseline'), env);
  const rb = sub('Robustness and held-out conditions');
  figRobustness(rb, env);
  if (env === 'Go1JoystickFlatTerrain') figOOS(rb, env, 'go1');
  if (env === 'Go1JoystickRoughTerrain') figOOS(rb, env, 'rt');
  if (!rb.querySelector('.chart')) rb.appendChild(h('div','missing',
    'No deterministic perturbation sweep has been run for this environment.'));
  return sec;
}

/* ---- gate ---------------------------------------------------------------- */
function buildGate(parent){
  const g = D.gate;
  const rows = D.envs.filter(e => g.per_env[e] && g.per_env[e].best_flat).map(e => ({
    label: (D.env_meta[e].short || e),
    flat: g.per_env[e].best_flat_success, hier: g.per_env[e].best_hier_success,
    flatSeeds: g.per_env[e].best_flat_seeds, hierSeeds: g.per_env[e].best_hier_seeds}));
  dumbbell(parent, rows, {
    title: 'Best flat baseline vs best hierarchical arm, per environment',
    sub: 'Primary success rate, training tail-mean. Small dots are individual seeds — where the '
       + 'two clouds overlap, the mean difference is not a separation.',
    xlabel: 'primary success rate'});
  const sc = h('div','scroller');
  const t = document.createElement('table');
  t.innerHTML = '<caption>Gate detail — imported from tools/analyze_v2.py, not recomputed here'
    + '</caption><thead><tr><th>environment</th><th>best flat</th><th class="num">success</th>'
    + '<th>best hierarchical</th><th class="num">success</th><th class="num">Δ</th>'
    + '<th>seeds separated?</th><th>budget matched?</th></tr></thead>';
  const tb = document.createElement('tb'.replace('tb','tbody'));
  D.envs.forEach(e => {
    const v = g.per_env[e]; if (!v || !v.best_flat) return;
    const tr = document.createElement('tr');
    const add = (txt, cls) => { const td = h('td', cls || null); td.textContent = txt;
      tr.appendChild(td); return td; };
    add(e, 'id');
    add(v.best_flat, 'id').style.color = vcol('flat');
    add(fmtNum(v.best_flat_success, 4), 'num');
    add(v.best_hier, 'id').style.color = vcol(v.best_hier.split('·')[0]);
    add(fmtNum(v.best_hier_success, 4), 'num');
    const d = v.best_hier_success - v.best_flat_success;
    const dd = add((d >= 0 ? '+' : '') + fmtNum(d, 4), 'num');
    dd.style.color = d > 0 ? vcol('nesy') : cssVar('--stop');
    const sep = add(v.separated ? 'separated' : 'overlapping seeds');
    sep.style.color = v.separated ? cssVar('--ok') : cssVar('--muted');
    const bm = add(v.budget_matched === false
      ? ('NO — ' + fmtSteps(v.best_flat_budget) + ' vs ' + fmtSteps(v.best_hier_budget))
      : (v.budget_matched === true ? 'yes' : 'unknown'));
    if (v.budget_matched === false) bm.style.color = cssVar('--stop');
    tb.appendChild(tr);
  });
  t.appendChild(tb); sc.appendChild(t); parent.appendChild(sc);
}

/* ---- audit --------------------------------------------------------------- */
function buildAudit(parent){
  D.audit.forEach(grp => {
    const d = document.createElement('details');
    if (grp.severity === 'stop') d.open = true;
    const s = document.createElement('summary');
    s.textContent = grp.title + '  —  ' + grp.items.length + ' '
      + (grp.items.length === 1 ? 'finding' : 'findings');
    if (grp.severity === 'stop') s.style.color = cssVar('--stop');
    if (grp.severity === 'ok') s.style.color = cssVar('--ok');
    d.appendChild(s);
    const b = h('div','dbody');
    const why = document.createElement('p'); why.innerHTML = grp.why; b.appendChild(why);
    if (grp.items.length){
      const ul = document.createElement('ul');
      grp.items.forEach(it => { const li = document.createElement('li');
        li.innerHTML = it; ul.appendChild(li); });
      b.appendChild(ul);
    } else {
      b.appendChild(h('p','dim','Nothing found. This check ran and passed.'));
    }
    d.appendChild(b); parent.appendChild(d);
  });
}

function buildProvenance(parent){
  const sc = h('div','scroller');
  const t = document.createElement('table');
  t.innerHTML = '<caption>Every asset on this page, and the checkpoint it came from</caption>'
    + '<thead><tr><th>file</th><th>environment</th><th>arm</th><th class="num">train seed</th>'
    + '<th class="num">render seed</th><th class="num">frames</th><th class="num">return</th>'
    + '<th>checkpoint</th><th>notes</th></tr></thead>';
  const tb = document.createElement('tbody');
  D.videos.forEach(v => {
    const tr = document.createElement('tr');
    const add = (txt, cls) => { const td = h('td', cls || null); td.textContent = txt;
      tr.appendChild(td); return td; };
    add(v.name, 'id');
    add(v.env || '?');
    add(v.arm || '?', 'id').style.color = vcol(v.variant);
    add(v.train_seed === null ? '?' : 's' + v.train_seed, 'num');
    add('r' + v.render_seed, 'num');
    add(fmtNum(v.frames, 0), 'num');
    add(fmtNum(v.episode_return, 1), 'num');
    add(v.checkpoint.replace(/^runs\//, ''), 'path');
    const n = add((v.notes || []).join('; ') || '—', 'wrapok');
    if ((v.notes || []).length) n.style.color = cssVar('--wait');
    tb.appendChild(tr);
  });
  t.appendChild(tb); sc.appendChild(t); parent.appendChild(sc);
}

function buildFig6(parent){
  if (!D.fig6.length){ parent.appendChild(h('div','missing','no panels rendered')); return; }
  const grid = h('div','chart-grid');
  D.fig6.forEach(p => {
    const c = h('div','imgcard');
    const im = document.createElement('img'); im.src = p.img; im.loading = 'lazy';
    im.alt = 'meta-policy decision panel'; c.appendChild(im);
    const cap = h('div','cap');
    cap.textContent = p.env + ' · ' + p.arm + ' · seed s' + p.seed + ' · step ' + p.step
      + ' — from ' + p.checkpoint;
    c.appendChild(cap); grid.appendChild(c);
  });
  parent.appendChild(grid);
}

function buildRules(parent){
  D.rules.forEach(r => {
    const d = document.createElement('details');
    const s = document.createElement('summary');
    s.textContent = r.env + ' — ' + r.title + '  (' + r.source + ')';
    d.appendChild(s);
    const b = h('div','dbody');
    const pre = document.createElement('pre'); pre.textContent = r.code;
    b.appendChild(pre); d.appendChild(b); parent.appendChild(d);
  });
}

/* ---- boot ---------------------------------------------------------------- */
function boot(){
  const nav = document.getElementById('nav');
  D.envs.forEach(e => {
    const a = document.createElement('a');
    a.className = 'navlink'; a.href = '#env-' + e;
    a.textContent = D.env_meta[e].short || e;
    nav.appendChild(a);
  });
  buildGate(document.getElementById('gate-body'));
  const host = document.getElementById('envs');
  D.envs.forEach(e => host.appendChild(buildEnv(e)));
  buildFig6(document.getElementById('fig6-body'));
  buildRules(document.getElementById('rules-body'));
  buildProvenance(document.getElementById('prov-body'));
  buildAudit(document.getElementById('audit-body'));

  /* The toggle has to flip relative to what the viewer is ACTUALLY seeing. Most viewers arrive
     with no data-theme stamped at all, so the starting point is the OS preference, not "light" --
     reading the attribute alone makes the first click a no-op for anyone on a dark system. */
  const tbtn = document.getElementById('theme');
  const sysDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  const resolved = () => document.documentElement.getAttribute('data-theme')
                      || (sysDark ? 'dark' : 'light');
  const label = () => { tbtn.textContent = resolved() === 'dark' ? 'light mode' : 'dark mode'; };
  label();
  /* Anything drawn from a resolved token -- every SVG, and the variant colours on table cells --
     is stale after a flip, so all four generated regions are rebuilt, not just the charts. */
  const paint = () => {
    ['gate-body', 'envs', 'fig6-body', 'rules-body', 'prov-body', 'audit-body']
      .forEach(id => { document.getElementById(id).innerHTML = ''; });
    buildGate(document.getElementById('gate-body'));
    const host2 = document.getElementById('envs');
    D.envs.forEach(e => host2.appendChild(buildEnv(e)));
    buildFig6(document.getElementById('fig6-body'));
    buildRules(document.getElementById('rules-body'));
    buildProvenance(document.getElementById('prov-body'));
    buildAudit(document.getElementById('audit-body'));
  };
  tbtn.addEventListener('click', () => {
    document.documentElement.setAttribute('data-theme',
      resolved() === 'dark' ? 'light' : 'dark');
    label();
    paint();
  });
}
document.addEventListener('DOMContentLoaded', boot);
"""


def _esc(s: Any) -> str:
    return html.escape(str(s))


BODY = """
<div class="topbar"><div class="topbar-in">
  <span class="brand">NEXUS results</span>
  <a class="navlink" href="#gate">Gate</a>
  <span id="nav" style="display:contents"></span>
  <a class="navlink" href="#interp">Interpretability</a>
  <a class="navlink" href="#deviations">Scope</a>
  <a class="navlink" href="#audit">Audit</a>
  <span class="spacer"></span>
  <button class="tg" id="theme">dark mode</button>
</div></div>

<div class="wrap">
<header class="masthead">
  <span class="eyebrow">continuous-control reproduction &middot; results board</span>
  <h1>NEXUS on continuous control: every environment, every meta-policy, every seed</h1>
  <p>The paper this reproduces &mdash; <em>From Objects to Skills: Interpretable Meta-Policies for
  Neural Control</em> &mdash; evaluates three discrete environments (Seaquest, Kangaroo, Crafter)
  with three seeds each. This board carries the continuous-control port: {n_env} MuJoCo
  environments, {n_arm} arms, {n_ck} trained checkpoints. Each environment section holds the
  paper&rsquo;s own figure types &mdash; per-skill return curves, the return-vs-goal bar chart, the
  zero-shot robustness comparison &mdash; next to the rollout video for each meta-policy.</p>
  <div class="prov">
    <span><b>generated</b> {generated}</span>
    <span><b>commit</b> {commit}</span>
    <span><b>checkpoints</b> {n_ck} from {dirs}</span>
    <span><b>greedy evals</b> {n_det}</span>
    <span><b>noise sweeps</b> {n_rob}</span>
    <span><b>videos</b> {n_vid}</span>
  </div>
  <div class="tiles">
    <div class="tile {gate_cls}"><span class="k">campaign gate</span>
      <span class="v">{gate}</span>
      <span class="n">{n_sep} of {n_scored} environments where a hierarchical arm separates from
      flat on seeds. Needs 4.</span></div>
    <div class="tile wait"><span class="k">ahead on the mean only</span>
      <span class="v">{n_pass} / {n_scored}</span>
      <span class="n">Hierarchy beats flat in mean, seeds overlapping. A mean win inside
      overlapping seeds is not a win.</span></div>
    <div class="tile"><span class="k">environments</span><span class="v">{n_env}</span>
      <span class="n">{env_list}</span></div>
    <div class="tile"><span class="k">trained runs</span><span class="v">{n_ck}</span>
      <span class="n">one checkpoint per seed, across {n_arm} distinct arms. Per-arm n is stated
      everywhere a number appears &mdash; it is not {n_ck} seeds of anything.</span></div>
  </div>
</header>

<section id="howto">
  <div class="sec-head"><span class="eyebrow">read this first</span>
    <h2>What the numbers on this page mean</h2></div>
  <ul>
    <li><b>Training tail-mean</b> &mdash; the mean of the last 10% of logged updates, computed
    exactly as <code>tools/analyze_v2.py</code> computes it. It is logged <em>with</em>
    &epsilon;-greedy exploration still on.</li>
    <li><b>Greedy / deterministic eval</b> &mdash; 64 episodes, exploration off, from
    <code>tools/robustness_eval.py</code>. On Go1 it reads ~0.17 lower than the training metric
    for the same checkpoint, so the two are never mixed in one column.</li>
    <li><b>Primary success</b> is the gate metric, never return. Panda has earned 655 return at
    0.001 lift success; walker once scored 931 on a vertical-axis artefact.</li>
    <li><b>n and seed spread are always shown.</b> A mean over an all-or-nothing distribution
    (hopper everywhere, panda at 1&times; and 2&times;) describes no episode that ever occurred, so
    per-seed dots sit on every bar.</li>
    <li><b>Budget-matched comparisons only.</b> Arms at a scaled budget are marked experimental,
    hatched in the charts, and excluded from the gate. Where a mismatch survives anyway it is
    labelled in place.</li>
    <li><b>Videos are one greedy episode.</b> They show that a behaviour exists, never how often
    it occurs &mdash; a Go1 render once showed 90.4% <code>recover</code> where the 64-episode
    eval says 24.6%.</li>
  </ul>
</section>

<section id="gate">
  <div class="sec-head"><span class="eyebrow">campaign gate</span>
    <h2>Environment matrix</h2></div>
  <p>The pre-registered gate: in at least 4 of {n_scored} environments, a hierarchical arm
  (<code>neural</code> or <code>nesy</code>) must beat the flat baseline on primary success at a
  matched budget. Reported on <em>seed separation</em> &mdash; min(hierarchical) &gt; max(flat)
  &mdash; because the letter of the rule is satisfied by a 0.0001 mean difference on a saturated
  environment.</p>
  <div id="gate-body" style="display:flex;flex-direction:column;gap:1rem"></div>
</section>

<div id="envs" style="display:flex;flex-direction:column;gap:2.6rem"></div>

<section id="interp">
  <div class="sec-head"><span class="eyebrow">paper Fig. 5 &amp; 6</span>
    <h2>Interpretability: the rules, and the decisions they produced</h2></div>
  <p>The paper&rsquo;s Fig. 6 shows Seaquest frames where several rules fire at once, beside the
  meta-Q value of each admissible skill. These are the continuous-control equivalents: greedy
  episodes, at the steps where the hand-written mask left more than one skill admissible and the
  top two Q-values were closest.</p>
  <div id="fig6-body"></div>
  <p>And the rule programs themselves &mdash; the paper&rsquo;s Fig. 5 &mdash; as they are actually
  executed, read straight out of the policy modules.</p>
  <div id="rules-body" style="display:flex;flex-direction:column;gap:.5rem"></div>
</section>

<section id="deviations">
  <div class="sec-head"><span class="eyebrow">scope</span>
    <h2>Where this reproduction departs from the paper</h2></div>
  <p>The paper evaluates three discrete, object-centric environments &mdash; Seaquest and Kangaroo
  from JAXAtari, and Crafter &mdash; at three seeds each, with &plusmn;1&sigma; bands and no
  significance testing. This is a port to continuous control, and several of its figures therefore
  cannot be reproduced exactly. Each difference is listed rather than papered over.</p>
  <div class="scroller"><table>
    <caption>Paper vs this board</caption>
    <thead><tr><th>what</th><th>paper</th><th>here</th></tr></thead>
    <tbody>
    <tr><td class="id">environments</td>
      <td class="wrapok">Seaquest, Kangaroo, Crafter (+ Pong, Breakout, Freeway in the appendix);
      object-centric symbolic state</td>
      <td class="wrapok">{n_env} MuJoCo tasks &mdash; {env_list} &mdash; on continuous state and
      continuous actions</td></tr>
    <tr><td class="id">arms</td>
      <td class="wrapok">PPO, PQN, HPQN baselines; NEXUS neural / symbolic / nesy</td>
      <td class="wrapok"><code>flat</code> (the PQN-family baseline), <code>neural</code>,
      <code>nesy</code>, <code>symbolic</code>. A Brax <b>PPO</b> baseline exists on Go1 and hopper
      and is shown against the greedy evaluations. <b>HPQN</b> exists only as the experimental
      <code>neural&middot;hpqn</code> arm on two environments and is excluded from the gate.</td></tr>
    <tr><td class="id">headline metric</td>
      <td class="wrapok">Human-Normalized Score, plus an aligned goal metric (divers rescued,
      level completion) and raw game reward</td>
      <td class="wrapok">No HNS &mdash; these tasks have no human baseline. The aligned goal metric
      is <code>policy_diag/primary_success_rate</code>, and it is what the gate scores; episode
      return is shown beside it, never instead of it.</td></tr>
    <tr><td class="id">skill returns</td>
      <td class="wrapok">Fig. 3/4/10 plot per-skill return for <em>every</em> arm, including the
      baselines, to show baselines collapsing onto one skill</td>
      <td class="wrapok"><b>Not reproducible from these checkpoints.</b> The flat arm logs only
      <code>skill_return/0_flat_actor</code>, so the per-skill panels compare the hierarchical
      arms against each other and say so in place.</td></tr>
    <tr><td class="id">robustness (Fig. 8)</td>
      <td class="wrapok">Trained on the standard game, evaluated zero-shot on a simplified variant
      (threats removed / no thirst)</td>
      <td class="wrapok">Two analogues: swept action noise on every environment, and on Go1 a
      zero-shot sweep over held-out command ranges, including a zero-command
      &ldquo;simplified&rdquo; condition.</td></tr>
    <tr><td class="id">seeds and statistics</td>
      <td class="wrapok">n = 3 (seeds 0,1,2) everywhere; &plusmn;1&sigma; bands; no tests</td>
      <td class="wrapok">n ranges from 1 to 43 and is printed next to every number. Bands are
      &plusmn;1&sigma;, and the gate is reported on <em>seed separation</em> rather than on means,
      because at n=3 a mean difference of 0.0001 satisfies the letter of the rule.</td></tr>
    <tr><td class="id">interpretability</td>
      <td class="wrapok">Fig. 5 rule listings; Fig. 6 three hand-picked Seaquest frames with the
      meta-Q vector</td>
      <td class="wrapok">Same two forms, generated rather than picked: the rule programs are read
      out of the executed policy modules, and the decision panels are selected automatically at
      the steps where the mask admitted more than one skill and the top two Q-values were
      closest.</td></tr>
    <tr><td class="id">budget</td>
      <td class="wrapok">200M frames (Atari), 1B (Crafter), fixed per environment</td>
      <td class="wrapok">Per-environment, and <em>not</em> uniform across arms. Budget-scaled arms
      are marked experimental and excluded from the gate; the mismatches that remain are named on
      the environment that carries them.</td></tr>
    </tbody>
  </table></div>
</section>

<section id="prov">
  <div class="sec-head"><span class="eyebrow">provenance</span>
    <h2>Which checkpoint every clip came from</h2></div>
  <p>Nothing on this page is bound by filename. Each clip is resolved through the checkpoint path
  recorded in its sidecar, and the environment, arm and <em>training</em> seed shown beside it are
  read from that checkpoint&rsquo;s own config. The <code>seed</code> field inside the sidecars is
  the render seed and is reported separately, because it is not the training seed and has been
  read as one before.</p>
  <div id="prov-body"></div>
</section>

<section id="audit">
  <div class="sec-head"><span class="eyebrow">integrity</span>
    <h2>Checks run against this board</h2></div>
  <p>Each block below is a check that ran while the page was built. A block with no findings means
  the check ran and passed; it is not an absence of evidence.</p>
  <div id="audit-body" style="display:flex;flex-direction:column;gap:.5rem"></div>
</section>

<footer>
  Built by <code>tools/build_results_dashboard.py</code> from {dirs} &middot; {generated} &middot;
  commit {commit}. Charts are drawn client-side from the embedded run index; video, skill strips
  and panels are referenced by relative path from <code>runs/</code>.
</footer>
</div>
"""


def render(data: dict[str, Any]) -> str:
    g = data["gate"]
    payload = {
        "generated": data["generated"],
        "commit": data["commit"],
        "envs": data["envs"],
        "env_meta": data["env_meta"],
        "cells": data["cells"],
        "runs": data["runs"],
        "det": data["det"],
        "robustness": data["robustness"],
        "oos": data["oos"],
        "videos": data["videos"],
        "gate": g,
        "shipped": data["shipped"],
        "audit": data["audit"],
        "fig6": data["fig6"],
        "rules": data["rules"],
        "ppo": data["ppo"],
        "ppo_note": data["ppo_note"],
    }
    n_arm = len(data["cells"])
    body = BODY.format(
        generated=_esc(data["generated"]),
        commit=_esc(data["commit"] or "unknown"),
        dirs=_esc(", ".join(data["dirs"])),
        n_ck=data["n_checkpoints"],
        n_env=len(data["envs"]),
        n_arm=n_arm,
        n_det=len(data["det"]),
        n_rob=len(data["robustness"]),
        n_vid=len(data["videos"]),
        gate=_esc(g["gate"]),
        gate_cls="ok" if g["gate"] == "PASS" else "stop",
        n_sep=g["n_separated"],
        n_pass=g["n_pass"],
        n_scored=g["n_scored"],
        env_list=_esc(", ".join(data["env_meta"][e]["short"] for e in data["envs"])),
    )
    blob = json.dumps(payload, separators=(",", ":"), allow_nan=False, default=lambda o: None)
    inner = (
        body
        + "\n<script>window.__DASH__=" + blob + ";</script>\n"
        + "<script>\n" + JS + "\n" + JS2 + "\n</script>\n"
    )
    if data.get("inline"):
        # Artifact edition: the host wraps the file in its own doctype/head/body, so this emits
        # page content only -- a second <html> would be dropped along with everything in it.
        return "<title>NEXUS Results Board</title>\n<style>\n" + CSS + "\n</style>\n" + inner
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        "<title>NEXUS Results Board</title>\n<style>\n" + CSS + "\n</style>\n</head>\n<body>\n"
        + inner + "</body>\n</html>\n"
    )

