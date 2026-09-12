"""
pareto_html.py — HTML-версия отчёта по фронту Парето (для публикации и шаринга).

Использует те же данные, что и markdown в pareto_report.py: упорядоченный фронт,
тексты преимуществ/недостатков, кластеры, представителей и выводы. Дополнительно
рисует диаграмму фронта в осях (F_fin, F_public) на фоне всех допустимых
кандидатов; цвет точки — семейство набора лотов (см. pareto_report.FAMILIES).
"""
from __future__ import annotations

import html
import re
from datetime import date

import pandas as pd

import team_model as tm
import user_modes as um

FAMILY_LABEL = {'money': 'Деньги', 'public': 'Польза', 'quality': 'Качество', 'other': 'Прочие наборы'}


def esc(v):
    return html.escape(str(v), quote=True)


def inline(text):
    """**жирный** и `код` из markdown -> HTML (после экранирования)."""
    t = esc(text)
    t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'`(.+?)`', r'<code>\1</code>', t)
    return t


def num(v, digits=0):
    if v is None or (isinstance(v, float) and v != v):
        return '—'
    s = f'{v:,.{digits}f}'.replace(',', ' ').replace('-', '−')
    return s


MONO_COLS = {'lots', 'modes', 'lot_id', 'mode_id', 'criterion', 'capability_groups', 'fallback_modes'}


def table(df, cols, headers, fmt=None, classes='', family_col=None, family_of=None):
    fmt = fmt or {}
    th = ''.join(f'<th>{esc(h)}</th>' for h in headers)
    rows = []
    for row in df.itertuples(index=False):
        tds = []
        for col in cols:
            v = getattr(row, col)
            cell = fmt[col](v) if col in fmt else esc(v)
            if family_col and col == family_col and family_of:
                cell = f'<span class="fam fam-{family_of(v)}" aria-hidden="true"></span>{cell}'
            cls = ' class="num"' if col in fmt else (' class="mono"' if col in MONO_COLS else '')
            tds.append(f'<td{cls}>{cell}</td>')
        rows.append('<tr>' + ''.join(tds) + '</tr>')
    return (f'<div class="tbl-wrap"><table class="tbl {classes}"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


# ---------------------------------------------------------------------------
# Диаграмма фронта
# ---------------------------------------------------------------------------

def scatter_svg(cand, f, family_of, labels):
    """SVG: все допустимые (серые) и фронт (по семействам) в осях F_fin x F_public."""
    W, H = 760, 470
    L, R, T, B = 54, 18, 16, 46
    pw, ph = W - L - R, H - T - B
    xmax = max(3.0, float(cand['F_fin'].max()))
    ymax = max(3.0, float(cand['F_public'].max()))

    def sx(v):
        return L + pw * float(v) / xmax

    def sy(v):
        return T + ph * (1 - float(v) / ymax)

    parts = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Фронт Парето в осях F_fin и F_public; {len(f)} максимальных элементов на фоне '
             f'{len(cand)} допустимых портфелей">']
    # сетка и оси
    ticks = [i * 0.5 for i in range(0, 7)]
    for tv in ticks:
        x, y = sx(tv), sy(tv)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T + ph}"/>')
        parts.append(f'<line class="grid" x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{T + ph + 18}" text-anchor="middle">{tv:g}</text>')
        parts.append(f'<text class="tick" x="{L - 8}" y="{y + 4:.1f}" text-anchor="end">{tv:g}</text>')
    parts.append(f'<line class="axis" x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}"/>')
    parts.append(f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{T + ph}"/>')
    parts.append(f'<text class="axis-label" x="{L + pw}" y="{H - 8}" text-anchor="end">F_fin — деньги (сумма z: NPV, PI, kcash)</text>')
    parts.append(f'<text class="axis-label" x="{L + 4}" y="{T - 4}" text-anchor="start">F_public — общественная польза (vpub, SROI, ядро)</text>')
    # доминируемые
    front_keys = set(zip(f['lots'], f['modes']))
    for r in cand.itertuples():
        if (r.lots, r.modes) in front_keys:
            continue
        parts.append(f'<circle class="dot-dom" cx="{sx(r.F_fin):.1f}" cy="{sy(r.F_public):.1f}" r="2.4"/>')
    # фронт
    for r in f.itertuples():
        fam = family_of(r.lots)
        x, y = sx(r.F_fin), sy(r.F_public)
        parts.append(f'<circle class="dot fam-{fam}" cx="{x:.1f}" cy="{y:.1f}" r="6"/>')
    # подписи представителей
    for no, (dx, dy, anchor) in labels.items():
        r = f[f['no'] == no].iloc[0]
        x, y = sx(r.F_fin) + dx, sy(r.F_public) + dy
        parts.append(f'<text class="pt-label" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">№{no}</text>')
    # зоны наведения — поверх всего
    for r in f.itertuples():
        x, y = sx(r.F_fin), sy(r.F_public)
        parts.append(
            f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="12" tabindex="0" '
            f'data-no="{int(r.no)}" data-lots="{esc(r.lots)}" data-modes="{esc(r.modes)}" '
            f'data-ffin="{r.F_fin:.3f}" data-fpub="{r.F_public:.3f}" data-fq="{r.F_quality:.3f}" '
            f'data-npv="{r.npv_mrub:.0f}" data-vpub="{r.vpub_mrub_per_year:.0f}" data-kcash="{r.kcash:.2f}" '
            f'data-core="{int(r.public_core_lots)}" data-fam="{esc(FAMILY_LABEL[family_of(r.lots)])}">'
            f'<title>№{int(r.no)} {esc(r.lots)} — {esc(r.modes)}</title></circle>')
    parts.append('</svg>')
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Страница
# ---------------------------------------------------------------------------

CSS = """
:root{
  --bg:#F4F6F9; --surface:#FFFFFF; --ink:#171D26; --muted:#5B6573; --rule:#D8DEE6; --rule-soft:#E8ECF1;
  --accent:#1E4F8F; --code-bg:#EDF1F6; --focus:#1E4F8F;
  --fam-money:#2a78d6; --fam-public:#eb6834; --fam-quality:#1baf7a; --fam-other:#8A94A3; --dot-dom:#B9C2CE;
  --good:#0ca30c; --critical:#d03b3b;
  color-scheme:light;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#11161D; --surface:#181F29; --ink:#E7ECF2; --muted:#9AA6B5; --rule:#2A3441; --rule-soft:#222B36;
    --accent:#7FB0EE; --code-bg:#222B36; --focus:#7FB0EE;
    --fam-money:#3987e5; --fam-public:#d95926; --fam-quality:#199e70; --fam-other:#7E8A9A; --dot-dom:#3A4553;
    --good:#0ca30c; --critical:#e66767;
    color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --bg:#11161D; --surface:#181F29; --ink:#E7ECF2; --muted:#9AA6B5; --rule:#2A3441; --rule-soft:#222B36;
  --accent:#7FB0EE; --code-bg:#222B36; --focus:#7FB0EE;
  --fam-money:#3987e5; --fam-public:#d95926; --fam-quality:#199e70; --fam-other:#7E8A9A; --dot-dom:#3A4553;
  --good:#0ca30c; --critical:#e66767;
  color-scheme:dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Golos Text",system-ui,"Segoe UI",Arial,sans-serif;
  font-size:15.5px;line-height:1.55;padding-block:0 64px;padding-inline:clamp(16px,4vw,40px)}
code,.mono,.tbl td.num,.el-title .modes,.el-title .lots{font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace}
code{background:var(--code-bg);padding:.05em .35em;border-radius:3px;font-size:.92em}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.page{max-width:1180px;margin:0 auto}
.prose{max-width:72ch}
h1,h2,h3{text-wrap:balance;line-height:1.2;margin:0}
h1{font-size:clamp(28px,4vw,40px);font-weight:700;letter-spacing:-.01em}
h2{font-size:22px;font-weight:600;margin-top:56px;padding-top:20px;border-top:1px solid var(--rule)}
h3{font-size:17px;font-weight:600;margin-top:28px}
p{margin:.6em 0}
.eyebrow{font-size:12.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 12px}
.masthead{padding-block:40px 8px}
.lede{font-size:17px;max-width:72ch;color:var(--ink)}
.funnel{display:flex;flex-wrap:wrap;gap:8px 0;align-items:stretch;margin-top:24px;border:1px solid var(--rule);
  background:var(--surface);border-radius:6px;overflow:hidden}
.step{flex:1 1 200px;padding:14px 18px;border-right:1px solid var(--rule-soft);min-width:0}
.step:last-child{border-right:0}
.step .n{font-family:"IBM Plex Mono",monospace;font-size:26px;font-weight:500;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.step .l{font-size:13px;color:var(--muted);margin-top:2px}
.step .arrow{color:var(--muted);font-size:12px}
.meta{font-size:13.5px;color:var(--muted)}
dl.spec{display:grid;grid-template-columns:max-content 1fr;gap:6px 18px;margin:12px 0;font-size:14.5px}
dl.spec dt{color:var(--muted)}
dl.spec dd{margin:0}
@media (max-width:640px){dl.spec{grid-template-columns:1fr}dl.spec dt{margin-top:6px}}
.tbl-wrap{overflow-x:auto;margin:14px 0;border:1px solid var(--rule);border-radius:6px;background:var(--surface)}
.tbl{border-collapse:collapse;width:100%;font-size:13.5px;min-width:600px}
.tbl th{position:sticky;top:0;background:var(--surface);text-align:left;font-weight:600;font-size:12.5px;
  letter-spacing:.02em;color:var(--muted);padding:9px 10px;border-bottom:1px solid var(--rule);white-space:nowrap}
.tbl td{padding:7px 10px;border-bottom:1px solid var(--rule-soft);vertical-align:top;white-space:nowrap}
.tbl tbody tr:last-child td{border-bottom:0}
.tbl td.num{text-align:right;font-variant-numeric:tabular-nums}
.tbl td.wrap{white-space:normal;min-width:260px}
.tbl.compact{min-width:0}
.tbl.gloss{min-width:0}
.tbl.gloss td{white-space:normal;vertical-align:top}
.tbl.gloss td.sym{font-weight:500;min-width:120px}
.tbl.gloss td.formula{font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace;font-size:12.5px;min-width:180px}
.tbl.gloss td.wrap{min-width:220px}
.fam{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:8px;vertical-align:-1px;background:var(--fam-other)}
.fam-money{background:var(--fam-money)} .fam-public{background:var(--fam-public)} .fam-quality{background:var(--fam-quality)} .fam-other{background:var(--fam-other)}
.legend{display:flex;flex-wrap:wrap;gap:8px 22px;font-size:13.5px;margin:14px 0 6px;align-items:center}
.legend .dom{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--dot-dom);margin-right:8px;vertical-align:1px}
.chart-wrap{position:relative;background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:8px}
.chart{width:100%;height:auto;display:block;max-width:100%}
.chart .grid{stroke:var(--rule-soft);stroke-width:1}
.chart .axis{stroke:var(--rule);stroke-width:1}
.chart .tick{fill:var(--muted);font-size:11px;font-family:"IBM Plex Mono",monospace}
.chart .axis-label{fill:var(--muted);font-size:12px}
.chart .dot-dom{fill:var(--dot-dom)}
.chart .dot{stroke:var(--surface);stroke-width:2}
.chart .dot.fam-money{fill:var(--fam-money)} .chart .dot.fam-public{fill:var(--fam-public)}
.chart .dot.fam-quality{fill:var(--fam-quality)} .chart .dot.fam-other{fill:var(--fam-other)}
.chart .pt-label{fill:var(--ink);font-size:12px;font-family:"IBM Plex Mono",monospace;font-weight:500;paint-order:stroke;stroke:var(--surface);stroke-width:3px}
.chart .hit{fill:transparent;cursor:pointer}
.chart .hit:focus-visible{outline:none;stroke:var(--focus);stroke-width:2}
.tip{position:absolute;pointer-events:none;background:var(--ink);color:var(--bg);font-size:12.5px;line-height:1.4;
  padding:8px 10px;border-radius:4px;max-width:280px;box-shadow:0 4px 14px rgba(0,0,0,.18);z-index:2}
.tip .t{font-family:"IBM Plex Mono",monospace;font-weight:500}
.tip .f{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
.el{border-top:1px solid var(--rule-soft);padding:18px 0 6px}
.el:first-of-type{border-top:0}
.el-title{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:baseline}
.el-title .no{font-family:"IBM Plex Mono",monospace;color:var(--muted);font-size:13px;min-width:2.4em}
.el-title .lots{font-weight:500;font-size:15px}
.el-title .modes{font-size:14px;color:var(--muted)}
.vec{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13px;color:var(--muted);margin:6px 0 10px;
  font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
.vec b{color:var(--ink);font-weight:500}
.pc{display:grid;grid-template-columns:1fr 1fr;gap:8px 32px}
@media (max-width:720px){.pc{grid-template-columns:1fr}}
.pc h4{margin:0 0 4px;font-size:12.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}
.pc ul{margin:0;padding-left:18px;font-size:14px}
.pc li{margin:3px 0}
.pc .pros h4::before,.pc .cons h4::before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px;vertical-align:1px}
.pc .pros h4::before{background:var(--good)} .pc .cons h4::before{background:var(--critical)}
.theses{list-style:none;padding:0;margin:12px 0}
.theses li{padding:12px 0;border-top:1px solid var(--rule-soft);max-width:80ch}
.theses li:first-child{border-top:0}
.note{font-size:13.5px;color:var(--muted)}
footer{margin-top:56px;padding-top:16px;border-top:1px solid var(--rule);font-size:13px;color:var(--muted)}
@media (prefers-reduced-motion: no-preference){.chart .dot{transition:r .12s ease}}
"""

JS = """
(function(){
  var wrap=document.getElementById('chart-wrap'); if(!wrap) return;
  var tip=document.getElementById('tip'); var svg=wrap.querySelector('svg');
  function show(el){
    var d=el.dataset;
    tip.innerHTML='<div class="t">№'+d.no+' · '+d.lots+'</div><div class="t">'+d.modes+'</div>'+
      '<div class="f">F = ('+d.ffin+', '+d.fpub+', '+d.fq+')</div>'+
      '<div>'+d.fam+' · NPV '+d.npv+' · vpub '+d.vpub+' · kcash '+d.kcash+' · ядро '+d.core+'/4</div>';
    tip.hidden=false;
    var r=el.getBoundingClientRect(), w=wrap.getBoundingClientRect();
    var x=r.left-w.left+r.width/2+12, y=r.top-w.top+r.height/2-12;
    if(x+tip.offsetWidth>w.width-8) x=r.left-w.left-tip.offsetWidth-12;
    if(y+tip.offsetHeight>w.height-8) y=w.height-tip.offsetHeight-8;
    if(y<4) y=4;
    tip.style.left=x+'px'; tip.style.top=y+'px';
  }
  function hide(){tip.hidden=true;}
  svg.querySelectorAll('.hit').forEach(function(h){
    h.addEventListener('mouseenter',function(){show(h)});
    h.addEventListener('mousemove',function(){show(h)});
    h.addEventListener('mouseleave',hide);
    h.addEventListener('focus',function(){show(h)});
    h.addEventListener('blur',hide);
  });
})();
"""


def build_html(f, pc, cand, stats, reference, config, norm, family_of, families, cluster_df, reps, theses, lots_df, modes_df,
               glossary=(), function_map=(), sum_df=None):
    n = len(f)
    c = config['constraints_common']
    um_df = um.build_user_modes(modes_df)
    d1 = um.d1_breakeven_table(lots_df, modes_df)
    d3 = um.d3_breakeven_table(lots_df, modes_df)

    enumerated = int(stats.enumerated.sum())
    inadmissible = int(stats.inadmissible.sum())
    feasible = int(stats.feasible.sum())
    n_unsafe = int((~(f.fallback_feasible_BASE & f.fallback_feasible_STRESS)).sum())

    # подписи представителей на диаграмме: № -> (dx, dy, anchor)
    rep_by_role = {role: int(r.no) for role, r in reps}
    labels = {}
    for role, off in (('лучший по деньгам', (10, -10, 'start')), ('лучший по общественной ценности', (10, 4, 'start')),
                      ('лучший по качеству', (-10, -10, 'end')), ('компромисс (минимальная сумма рангов)', (10, 18, 'start'))):
        if role in rep_by_role and rep_by_role[role] not in labels:
            labels[rep_by_role[role]] = off

    fmt3 = lambda v: num(v, 3)
    fmt2 = lambda v: num(v, 2)
    fmt0 = lambda v: num(v, 0)
    fmt_int = lambda v: str(int(v))

    parts = []
    parts.append('<title>Фронт Парето Космотона</title>')
    parts.append('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Golos+Text:wght@400;500;600;700'
                 '&family=IBM+Plex+Mono:wght@400;500&display=swap">')
    parts.append(f'<style>{CSS}</style>')
    parts.append('<div class="page">')

    # --- masthead
    parts.append('<header class="masthead">')
    parts.append(f'<p class="eyebrow">Космос как инфраструктура · Космохакатон 2026 · прогон {date.today().isoformat()} · нормировка {esc(norm)}</p>')
    parts.append(f'<h1>Фронт Парето: {n} портфелей, которые никто не доминирует</h1>')
    parts.append('<p class="lede">Полный перебор четырёх лотов из восьми с режимами A/B/C и двумя пользовательскими режимами '
                 'D1 и D3, отсечение по девяти каноническим ограничениям в BASE и STRESS, три блочные fitness-функции '
                 '(деньги, общественная польза, качество) и максимальные элементы по частичному порядку «лучше по всем трём».</p>')
    parts.append('<div class="funnel">'
                 f'<div class="step"><div class="n">{num(enumerated)}</div><div class="l">комбинаций перебрано</div></div>'
                 f'<div class="step"><div class="n">{num(enumerated - inadmissible)}</div><div class="l">после условия допустимости D3 (≥ 2 лота с общей группой)</div></div>'
                 f'<div class="step"><div class="n">{num(feasible)}</div><div class="l">допустимы в BASE и STRESS</div></div>'
                 f'<div class="step"><div class="n">{n}</div><div class="l">максимальных элементов</div></div>'
                 '</div>')
    parts.append('</header>')

    # --- chart
    parts.append('<h2>Фронт в осях денег и общественной пользы</h2>')
    parts.append('<p class="prose">Каждая точка — допустимый портфель. Третья координата, качество, не показана: '
                 'она зависит только от набора лотов, поэтому цвет кодирует семейство набора. Наведите на точку фронта, '
                 'чтобы увидеть состав и вектор.</p>')
    parts.append('<div class="legend">'
                 + ''.join(f'<span><span class="fam fam-{k}" aria-hidden="true"></span>{esc(FAMILY_LABEL[k])} — '
                           f'{esc(", ".join(sets)) if k != "other" else "остальные три набора"}</span>'
                           for k, (_, sets) in list(families.items()) + [('other', ('', ()))])
                 + f'<span><span class="dom" aria-hidden="true"></span>допустимые, но доминируемые ({num(feasible - n)})</span>'
                 '</div>')
    parts.append('<div class="chart-wrap" id="chart-wrap">')
    parts.append(scatter_svg(cand, f, family_of, labels))
    parts.append('<div class="tip" id="tip" hidden></div>')
    parts.append('</div>')

    # --- setup
    parts.append('<h2>Постановка</h2>')
    parts.append('<dl class="spec">')
    parts.append('<dt>Перебор</dt><dd>C(8,4) = 70 наборов лотов × назначения режимов из {A, B, C, D}; D — один активный '
                 'пользовательский режим на прогон, как в шаблоне организаторов. Вывод режимов — <code>test-main/user_modes.py</code>.</dd>')
    parts.append(f'<dt>Отсечение</dt><dd>девять ограничений <code>case_core.check_constraints</code> в BASE (c0 ≤ 1300) и STRESS (c0 ≤ 1180): '
                 f'opex ≤ {c["opex_max_mrub_per_year"]}, vpub ≥ {c["vpub_min_mrub_per_year"]}, kcash ≥ {c["kcash_min"]}, '
                 f't_rep ≥ {c["t_rep_min"]}, ядро ≥ {c["min_public_core_lots"]}, архетипы ≥ {c["min_territorial_archetypes"]}, '
                 f'группы ≥ {c["min_capability_groups"]}.</dd>')
    for name, keys in tm.BLOCKS.items():
        parts.append(f'<dt><code>{name}</code></dt><dd>сумма z({esc(", ".join(keys))})</dd>')
    parts.append('<dt>Порядок</dt><dd>X доминирует Y, если X ≥ Y по всем трём блокам и X &gt; Y хотя бы по одному. '
                 'Деньги и общественная ценность нигде не складываются.</dd>')
    parts.append(f'<dt>Нормировка</dt><dd>{esc(norm)}: каждая компонента приведена к [0, 1] по min/max среди допустимых кандидатов '
                 '(все допустимые портфели убыточны по NPV и PI — это суть кейса, поэтому привязка к безубыточности не работает).</dd>')
    parts.append('</dl>')
    ref = reference.reset_index().rename(columns={'index': 'criterion'})
    parts.append('<h3>Эталон нормировки: min/max по допустимым</h3>')
    parts.append(table(ref, ['criterion', 'min', 'max'], ['критерий', 'min', 'max'],
                       {'min': lambda v: num(v, 3), 'max': lambda v: num(v, 3)}, classes='compact'))

    # --- user modes
    parts.append('<h2>Обозначения, формулы и функции</h2>')
    parts.append('<p class="prose">Расшифровка всех величин, которые встречаются в таблицах ниже, и карта функций по файлам.</p>')
    for title, items in glossary:
        parts.append(f'<h3>{inline(title)}</h3>')
        rows = ''.join(f'<tr><td class="sym">{inline(sym)}</td><td class="formula">{inline(formula)}</td>'
                       f'<td class="wrap">{inline(meaning)}</td></tr>' for sym, formula, meaning in items)
        parts.append(f'<div class="tbl-wrap"><table class="tbl gloss"><thead><tr><th>обозначение</th>'
                     f'<th>формула / источник</th><th>смысл</th></tr></thead><tbody>{rows}</tbody></table></div>')
    if function_map:
        parts.append('<h3>Карта функций</h3>')
        rows = ''.join(f'<tr><td class="formula">{inline(file)}</td><td class="formula wrap">{inline(funcs)}</td>'
                       f'<td class="wrap">{inline(purpose)}</td></tr>' for file, funcs, purpose in function_map)
        parts.append(f'<div class="tbl-wrap"><table class="tbl gloss"><thead><tr><th>файл</th><th>функции</th>'
                     f'<th>назначение</th></tr></thead><tbody>{rows}</tbody></table></div>')

    parts.append('<h2>Пользовательские режимы в прогоне</h2>')
    parts.append('<p class="prose">Коэффициенты выводятся формулами из канонических A/B, а не задаются числами; параметры '
                 'механизмов, формулы, условия допустимости и точки безубыточности — в docstring <code>user_modes.py</code>. '
                 'Варианты D2, D4–D7 из <code>d_type.md</code> исключены — причины там же.</p>')
    parts.append(table(um_df, ['mode_id', 'k_c0', 'k_opex', 'k_vpub', 'k_anchor', 'k_commercial', 'public_core', 'label'],
                       ['режим', 'k_c0', 'k_opex', 'k_vpub', 'k_anchor', 'k_commercial', 'ядро', 'механизм'],
                       {k: (lambda v: num(v, 4).rstrip('0').rstrip('.') if isinstance(v, float) else esc(v))
                        for k in ('k_c0', 'k_opex', 'k_vpub', 'k_anchor', 'k_commercial')}, classes='compact'))
    parts.append(f'<h3>D1: безубыточность надстройки по лотам (theta = {um.D1_PARAMS["theta"]:.2f})</h3>')
    parts.append('<p class="prose note">theta* — доля разрыва коммерческих коэффициентов B−A, при которой ΔNPV = 0. '
                 'Лот с theta* выше принятого значения в D1 хуже, чем в A.</p>')
    parts.append(table(d1, ['lot_id', 'Q_commercial', 'dC0_mrub', 'dOPEX_mrub_per_year', 'dCASH_mrub_per_year',
                            'dNPV_vs_A_mrub', 'theta_breakeven', 'pays_off_at_theta'],
                       ['лот', 'Q', 'ΔC0', 'ΔOPEX/год', 'ΔCASH/год', 'ΔNPV vs A', 'theta*', 'окупается'],
                       {'Q_commercial': fmt0, 'dC0_mrub': fmt2, 'dOPEX_mrub_per_year': fmt2, 'dCASH_mrub_per_year': fmt2,
                        'dNPV_vs_A_mrub': fmt2, 'theta_breakeven': fmt3,
                        'pays_off_at_theta': lambda v: 'да' if v else 'нет'}, classes='compact'))
    parts.append('<h3>D3: экономия против A по лотам</h3>')
    parts.append(f'<p class="prose note">Допустимо только для ≥ 2 лотов с общей группой возможностей. Нулевая экономия при '
                 f'rho_c0 = {d3.attrs["rho_c0_zero"]:.3f}, rho_opex = {d3.attrs["rho_opex_zero"]:.3f}.</p>')
    parts.append(table(d3, ['lot_id', 'capability_groups', 'dC0_saved_mrub', 'dOPEX_saved_mrub_per_year', 'dNPV_vs_A_mrub'],
                       ['лот', 'группы', 'экономия C0', 'экономия OPEX/год', 'ΔNPV vs A'],
                       {'dC0_saved_mrub': fmt2, 'dOPEX_saved_mrub_per_year': fmt2, 'dNPV_vs_A_mrub': fmt2}, classes='compact'))

    # --- summary table
    parts.append('<h2>Максимальные элементы: сводная таблица</h2>')
    parts.append('<p class="prose">Сгруппировано по набору лотов (кластеры упорядочены по лучшему F_fin), внутри кластера — '
                 'по F_fin. «Запас c0» — до лимита STRESS 1180 млн руб.</p>')
    parts.append(table(f, ['no', 'lots', 'modes', 'F_fin', 'F_public', 'F_quality', 'npv_mrub', 'profitability_index', 'kcash',
                           'vpub_mrub_per_year', 'sroi', 'public_core_lots', 't_rep', 'c0_mrub', 'headroom_c0_stress'],
                       ['№', 'лоты', 'режимы', 'F_fin', 'F_public', 'F_quality', 'NPV', 'PI', 'kcash', 'vpub', 'SROI',
                        'ядро', 't_rep', 'c0', 'запас c0'],
                       {'no': fmt_int, 'F_fin': fmt3, 'F_public': fmt3, 'F_quality': fmt3, 'npv_mrub': fmt0,
                        'profitability_index': fmt2, 'kcash': fmt2, 'vpub_mrub_per_year': fmt0, 'sroi': fmt2,
                        'public_core_lots': fmt_int, 't_rep': fmt3, 'c0_mrub': fmt0, 'headroom_c0_stress': fmt0},
                       family_col='lots', family_of=family_of))

    # --- pros / cons
    parts.append('<h2>Преимущества и недостатки каждого элемента</h2>')
    parts.append('<p class="prose">Текст формируется по одним и тем же правилам для всех элементов '
                 '(<code>pareto_report.pros_cons</code>). «Откат D → A» — проверка: если гипотеза D не подтвердится сметой '
                 'или спросом и лоты вернутся в A, остаётся ли портфель допустимым в BASE и STRESS.</p>')
    for r in f.itertuples():
        pros, cons = pc[r.no - 1]
        parts.append('<article class="el">')
        parts.append(f'<div class="el-title"><span class="no">№{int(r.no)}</span>'
                     f'<span class="lots"><span class="fam fam-{family_of(r.lots)}" aria-hidden="true"></span>{esc(r.lots)}</span>'
                     f'<span class="modes">{esc(r.modes)}</span></div>')
        parts.append(f'<div class="vec"><span>F_fin <b>{num(r.F_fin, 3)}</b> · ранг {r.rank_F_fin}</span>'
                     f'<span>F_public <b>{num(r.F_public, 3)}</b> · ранг {r.rank_F_public}</span>'
                     f'<span>F_quality <b>{num(r.F_quality, 3)}</b> · ранг {r.rank_F_quality}</span>'
                     f'<span>архетипы: {esc(r.archetypes)}</span></div>')
        parts.append('<div class="pc">')
        parts.append('<div class="pros"><h4>Преимущества</h4><ul>' + ''.join(f'<li>{esc(p)}</li>' for p in pros) + '</ul></div>')
        parts.append('<div class="cons"><h4>Недостатки и риски</h4><ul>' + ''.join(f'<li>{esc(q)}</li>' for q in cons) + '</ul></div>')
        parts.append('</div></article>')

    # --- clusters & representatives
    parts.append('<h2>Структура фронта по наборам лотов</h2>')
    parts.append(table(cluster_df, ['lots', 'n', 'F_fin_max', 'F_public_max', 'F_quality', 'npv_max', 'vpub_min', 'vpub_max', 'archetypes'],
                       ['лоты', 'элементов', 'F_fin max', 'F_public max', 'F_quality', 'NPV max', 'vpub min', 'vpub max', 'архетипы'],
                       {'n': fmt_int, 'F_fin_max': fmt3, 'F_public_max': fmt3, 'F_quality': fmt3, 'npv_max': fmt0,
                        'vpub_min': fmt0, 'vpub_max': fmt0}, family_col='lots', family_of=family_of))
    parts.append('<h2>Представители фронта</h2>')
    rep_rows = pd.DataFrame([{
        'role': role, 'no': int(r.no), 'lots': r.lots, 'modes': r.modes, 'F_fin': r.F_fin, 'F_public': r.F_public,
        'F_quality': r.F_quality, 'npv': r.npv_mrub, 'kcash': r.kcash, 'vpub': r.vpub_mrub_per_year,
        'core': int(r.public_core_lots),
        'fb': 'допустим' if (r.fallback_feasible_BASE and r.fallback_feasible_STRESS) else f'нарушает {r.fallback_failed}',
    } for role, r in reps])
    parts.append(table(rep_rows, ['role', 'no', 'lots', 'modes', 'F_fin', 'F_public', 'F_quality', 'npv', 'kcash', 'vpub', 'core', 'fb'],
                       ['роль', '№', 'лоты', 'режимы', 'F_fin', 'F_public', 'F_quality', 'NPV', 'kcash', 'vpub', 'ядро', 'откат D → A'],
                       {'no': fmt_int, 'F_fin': fmt3, 'F_public': fmt3, 'F_quality': fmt3, 'npv': fmt0, 'kcash': fmt2,
                        'vpub': fmt0, 'core': fmt_int}, family_col='lots', family_of=family_of))

    # --- theses
    if sum_df is not None:
        parts.append('<h2>Выбор по сумме нормированных</h2>')
        parts.append('<p class="prose">Скалярный рейтинг по всем допустимым кандидатам. <b>S</b> = F_fin + F_public + F_quality — '
                     'сумма всех десяти нормированных компонент (0…10), неявные веса блоков 3 : 3 : 4 по числу компонент; '
                     '<b>S_eq</b> = F_fin/3 + F_public/3 + F_quality/4 — равные веса блоков (0…3). Максимум положительно '
                     'взвешенной суммы всегда лежит на фронте, следующие места — не обязательно: столбец «фронт» показывает, '
                     'максимален ли элемент (№ — его номер в разделах выше).</p>')
        parts.append(table(sum_df, ['rank_S', 'S', 'rank_S_eq', 'S_eq', 'no', 'lots', 'modes', 'F_fin', 'F_public', 'F_quality',
                                    'pareto', 'npv_mrub', 'kcash', 'vpub_mrub_per_year', 'public_core_lots'],
                           ['место S', 'S', 'место S_eq', 'S_eq', '№', 'лоты', 'режимы', 'F_fin', 'F_public', 'F_quality',
                            'фронт', 'NPV', 'kcash', 'vpub', 'ядро'],
                           {'rank_S': fmt_int, 'S': fmt3, 'rank_S_eq': fmt_int, 'S_eq': fmt3, 'F_fin': fmt3, 'F_public': fmt3,
                            'F_quality': fmt3, 'no': lambda v: str(int(v)) if v else '—',
                            'pareto': lambda v: 'да' if v else 'нет', 'npv_mrub': fmt0, 'kcash': fmt2,
                            'vpub_mrub_per_year': fmt0, 'public_core_lots': fmt_int},
                           family_col='lots', family_of=family_of))
        best = sum_df.iloc[0]
        best_eq = sum_df.sort_values('S_eq', ascending=False).iloc[0]
        parts.append(f'<p class="prose">Лучший по S: <b class="mono">{esc(best.lots)} — {esc(best.modes)}</b> (S = {num(best.S, 3)}); '
                     f'лучший по S_eq: <b class="mono">{esc(best_eq.lots)} — {esc(best_eq.modes)}</b> (S_eq = {num(best_eq.S_eq, 3)}). '
                     'Сумма — это уже выбор весов; фронт показывает, что теряется при любом другом их наборе.</p>')

    parts.append('<h2>Выводы по структуре фронта</h2>')
    parts.append(f'<p class="prose note">Проверены для прогона от {date.today().isoformat()}; при смене параметров D или '
                 'блоков перегенерировать отчёт и перечитать этот раздел.</p>')
    parts.append('<ul class="theses">' + ''.join(f'<li>{inline(t)}</li>' for t in theses) + '</ul>')

    parts.append('<footer>Источник: <code>test-main/pareto_search.py</code> → <code>pareto_report.py</code> → '
                 '<code>pareto_html.py</code>. Markdown-версия: <code>PARETO_FRONT.md</code>; данные: '
                 '<code>test-main/output/pareto_front_report.csv</code>. Канонический слой <code>case_core.py</code> не изменён; '
                 f'быстрый путь перебора сверен с ним на выборке (максимальная ошибка 2e-13).</footer>')
    parts.append('</div>')
    parts.append(f'<script>{JS}</script>')
    return '\n'.join(parts)
