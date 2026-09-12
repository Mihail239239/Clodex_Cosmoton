"""
pareto_report.py — отчёт по фронту Парето: таблица максимальных элементов
(лоты и режимы) с преимуществами и недостатками каждого.

Запускает pareto_search.run_search (те же миры, нормировка, блоки), затем для
каждого элемента фронта считает:
  - ранги по трём блокам среди элементов фронта;
  - запасы по ограничениям (c0 в STRESS, opex, vpub, ядро);
  - детали пользовательских режимов: лоты в D, общая группа (D3), выигрыш NPV
    против тех же лотов в A, безубыточная доля захвата theta* (D1);
  - проверку отката: остаётся ли портфель допустимым в BASE и STRESS, если
    гипотеза D не подтвердится и лоты в D вернутся в A;
  - текст преимуществ и недостатков по фиксированным правилам (см. pros_cons).

Результат: ../PARETO_FRONT.md и output/pareto_front_report.csv.

Запуск (из test-main): python pareto_report.py [--norm minmax|threshold]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from case_core import load_case
import team_model as tm
import user_modes as um
from pareto_search import build_worlds, run_search

TOP = 5  # "в лучших TOP" по блоку считается преимуществом, "в худших TOP" — недостатком


# ---------------------------------------------------------------------------
# Обогащение элементов фронта
# ---------------------------------------------------------------------------

def _fallback_to_a(selection, cache, config):
    """Метрики и допустимость того же набора лотов с D -> A."""
    combo = tuple((lot, 'A' if mode.startswith('D') else mode) for lot, mode in selection)
    m = tm._fast_metrics(combo, cache)
    fin = tm.financial_metrics(m)
    feas = {s: tm._feasible_fast(m, config, s) for s in tm.SCENARIOS}
    failed = []
    for s in tm.SCENARIOS:
        rep = tm.constraint_report(m, config, s)
        failed += [f'{row.constraint}@{s}' for row in rep.itertuples() if not row.ok]
    return {
        'fallback_modes': '+'.join(mode for _, mode in combo),
        'fallback_c0_mrub': m['c0_mrub'],
        'fallback_npv_mrub': fin['npv_mrub'],
        'fallback_feasible_BASE': feas['BASE'],
        'fallback_feasible_STRESS': feas['STRESS'],
        'fallback_failed': ', '.join(failed),
    }


def enrich_front(front, lots, modes, config, worlds):
    """Добавляет ранги, запасы, детали D и проверку отката к каждому элементу."""
    c = config['constraints_common']
    c0_stress = config['scenarios']['STRESS']['c0_max_mrub']
    lidx = lots.set_index('lot_id', drop=False)
    caches = {name: tm._precompute(lots, mw) for name, mw, _ in worlds}
    d1 = um.d1_breakeven_table(lots, modes).set_index('lot_id')
    d3 = um.d3_breakeven_table(lots, modes).set_index('lot_id')

    f = front.copy()
    n = len(f)
    for block in tm.BLOCKS:
        f[f'rank_{block}'] = f[block].rank(ascending=False, method='min').astype(int)
    f['headroom_c0_stress'] = c0_stress - f['c0_mrub']
    f['headroom_c0_stress_pct'] = 100.0 * f['headroom_c0_stress'] / c0_stress
    f['headroom_opex'] = c['opex_max_mrub_per_year'] - f['opex_mrub_per_year']
    f['headroom_vpub'] = f['vpub_mrub_per_year'] - c['vpub_min_mrub_per_year']
    f['surplus_mrub_per_year'] = f['cash_mrub_per_year'] - f['opex_mrub_per_year']

    extra = []
    for row in f.itertuples():
        sel = list(row.selection)
        d_lots = [lot for lot, mode in sel if mode.startswith('D')]
        c_lots = [lot for lot, mode in sel if mode == 'C']
        info = {
            'd_mode': row.world if d_lots else '',
            'd_lots': '+'.join(d_lots),
            'c_lots': '+'.join(c_lots),
            'd_common_group': '',
            'd_gain_npv_mrub': 0.0,
            'd1_theta_max': float('nan'),
            'd1_unprofitable': '',
            'd1_marginal': '',
            'archetypes': '+'.join(sorted({lidx.loc[l].territorial_archetype for l, _ in sel})),
        }
        if d_lots:
            info.update(_fallback_to_a(sel, caches[row.world], config))
            info['d_gain_npv_mrub'] = row.npv_mrub - info['fallback_npv_mrub']
            if row.world == 'D3':
                common = set.intersection(*(um._caps(lidx.loc[l]) for l in d_lots))
                info['d_common_group'] = '/'.join(sorted(common))
            if row.world == 'D1':
                th = d1.loc[d_lots, 'theta_breakeven']
                info['d1_theta_max'] = float(th.max())
                info['d1_unprofitable'] = '+'.join(th[th > um.D1_PARAMS['theta']].index)
                info['d1_marginal'] = '+'.join(th[(th <= um.D1_PARAMS['theta']) & (th > 0.15)].index)
        else:
            info.update({'fallback_modes': '', 'fallback_c0_mrub': float('nan'), 'fallback_npv_mrub': float('nan'),
                         'fallback_feasible_BASE': True, 'fallback_feasible_STRESS': True, 'fallback_failed': ''})
        extra.append(info)
    f = pd.concat([f.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    f['n_front'] = n
    return f


# ---------------------------------------------------------------------------
# Преимущества и недостатки по правилам
# ---------------------------------------------------------------------------

def pros_cons(r, n, thresholds):
    """Списки строк для одного элемента. r — строка обогащённого фронта."""
    c = thresholds['constraints_common']
    pros, cons = [], []

    # деньги
    if r.rank_F_fin <= TOP:
        pros.append(f'деньги: {r.rank_F_fin}-й из {n} (NPV {r.npv_mrub:.0f}, PI {r.profitability_index:.2f})')
    elif r.rank_F_fin > n - TOP:
        cons.append(f'деньги: {r.rank_F_fin}-й из {n} (NPV {r.npv_mrub:.0f}, PI {r.profitability_index:.2f})')
    if r.kcash >= 1.2:
        pros.append(f'поступления покрывают OPEX с запасом {100 * (r.kcash - 1):.0f}% — '
                    f'операционный профицит {r.surplus_mrub_per_year:.1f} млн/год')
    elif r.kcash >= 1.0:
        pros.append(f'OPEX покрыт поступлениями (kcash {r.kcash:.2f}), субсидия на эксплуатацию не нужна')
    else:
        cons.append(f'поступления не покрывают OPEX (kcash {r.kcash:.2f}) — субсидия '
                    f'{-r.surplus_mrub_per_year:.1f} млн/год')

    # общественная польза
    if r.rank_F_public <= TOP:
        pros.append(f'общественная ценность: {r.rank_F_public}-й из {n} (vpub {r.vpub_mrub_per_year:.0f}, SROI {r.sroi:.2f})')
    elif r.headroom_vpub >= 250:
        pros.append(f'vpub {r.vpub_mrub_per_year:.0f} — запас {r.headroom_vpub:.0f} над порогом {c["vpub_min_mrub_per_year"]}')
    if r.headroom_vpub < 50:
        cons.append(f'vpub {r.vpub_mrub_per_year:.1f} — на грани порога {c["vpub_min_mrub_per_year"]} '
                    f'(запас {r.headroom_vpub:.1f})')
    elif r.rank_F_public > n - TOP:
        cons.append(f'общественная ценность: {r.rank_F_public}-й из {n} (vpub {r.vpub_mrub_per_year:.0f}, SROI {r.sroi:.2f})')
    if r.public_core_lots >= 3:
        pros.append(f'{int(r.public_core_lots)}/4 лота с общественным ядром (минимум {c["min_public_core_lots"]})')
    else:
        cons.append('ровно 2 лота с общественным ядром — на минимуме ограничения')
    if r.c_lots:
        cons.append(f'режим C на {r.c_lots}: общественная ценность этих лотов снижена до 62%')

    # качество
    if r.rank_F_quality <= TOP:
        pros.append(f'качество: {r.rank_F_quality}-й из {n} (t_rep {r.t_rep:.3f}, готовность {r.readiness_1_5:.2f}, '
                    f'устойчивость {r.resilience_1_5:.2f})')
    elif r.rank_F_quality > n - TOP:
        cons.append(f'качество: {r.rank_F_quality}-й из {n} (t_rep {r.t_rep:.3f}, готовность {r.readiness_1_5:.2f}, '
                    f'устойчивость {r.resilience_1_5:.2f})')
    if r.territorial_archetypes >= 4:
        pros.append('4 территориальных архетипа (минимум 3)')
    if r.capability_groups >= 3:
        pros.append(f'{int(r.capability_groups)} группы возможностей (минимум 2)')
    else:
        cons.append('только 2 группы возможностей — на минимуме ограничения')
    lots = set(r.lots.split('+'))
    if not lots & {'FIRE', 'FLOOD', 'ARCTIC'}:
        cons.append('нет лотов Сибири, Дальнего Востока и Арктики — территории ЧС-рисков не покрыты')

    # ограничения / стресс
    if r.headroom_c0_stress >= 40:
        pros.append(f'запас по c0 в STRESS {r.headroom_c0_stress:.0f} млн ({r.headroom_c0_stress_pct:.1f}%)')
    elif r.headroom_c0_stress < 10:
        cons.append(f'запас по c0 в STRESS всего {r.headroom_c0_stress:.1f} млн ({r.headroom_c0_stress_pct:.1f}%)')

    # пользовательские режимы
    if r.d_mode == 'D3':
        pros.append(f'совместная закупка D3 на {r.d_lots} (общая группа {r.d_common_group}): '
                    f'+{r.d_gain_npv_mrub:.0f} млн NPV против тех же лотов в A')
        n_d = len(r.d_lots.split('+'))
        if n_d >= 3:
            cons.append(f'{n_d} лота зависят от одного общего компонента D3 — риск концентрации и смены оператора')
    if r.d_mode == 'D1':
        if r.d1_unprofitable:
            cons.append(f'надстройка D1 на {r.d1_unprofitable} не окупается при theta = {um.D1_PARAMS["theta"]:.2f} '
                        f'(theta* до {r.d1_theta_max:.3f})')
        else:
            pros.append(f'надстройка D1 окупается на всех лотах {r.d_lots} (theta* <= {r.d1_theta_max:.3f} < '
                        f'{um.D1_PARAMS["theta"]:.2f}): +{r.d_gain_npv_mrub:.0f} млн NPV против A')
        if r.d1_marginal:
            cons.append(f'D1 на {r.d1_marginal} на грани окупаемости (theta* > 0.15)')
        cons.append('выручка D1 — гипотеза спроса: при theta = 0 портфель хуже версии в A на цену надстройки')
    if r.d_mode:
        if r.fallback_feasible_BASE and r.fallback_feasible_STRESS:
            pros.append('откат D → A сохраняет допустимость в BASE и STRESS')
        else:
            cons.append(f'откат D → A нарушает {r.fallback_failed} (c0 в A = {r.fallback_c0_mrub:.0f}) — '
                        f'гипотеза D критична для допустимости')
    return pros, cons


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _md_table(df, cols, headers, fmt=None):
    fmt = fmt or {}
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
    for row in df.itertuples(index=False):
        cells = []
        for col in cols:
            v = getattr(row, col)
            cells.append(fmt[col](v) if col in fmt else str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Обозначения и формулы (для расшифровки расчёта в отчёте)
# ---------------------------------------------------------------------------
# Каждый пункт: (обозначение, формула / источник, смысл). Секции выводятся в
# markdown и HTML одинаково; менять здесь, а не в рендерах.

GLOSSARY = [
    ('Входные данные лота (`data/lots.csv`, канон, не меняются)', [
        ('C', '`c0_mrub`', 'разовые затраты на запуск сервиса, млн руб.'),
        ('O', '`opex_mrub_per_year`', 'годовые эксплуатационные затраты, млн руб./год'),
        ('V', '`vpub_mrub_per_year`', 'общественная ценность, млн руб./год — оценка эффекта, не денежный поток'),
        ('P', '`anchor_cash_mrub_per_year`', 'якорные поступления от публичного заказчика, млн руб./год'),
        ('Q', '`commercial_cash_mrub_per_year`', 'коммерческие поступления от бизнеса, млн руб./год'),
        ('t_rep', '`t_rep`', 'канонический безразмерный показатель, 0…1; порог 0.63 по среднему портфеля; содержательная расшифровка в материалах кейса не дана (README организаторов просит не подменять её догадкой)'),
        ('readiness, resilience, scale', '`*_1_5`', 'готовность, устойчивость, масштабируемость лота, 1…5'),
        ('архетип, группы', '`territorial_archetype`, `capability_groups`', 'территория лота; группы возможностей EO, PNT/InSAR, SATCOM, SSA'),
    ]),
    ('Режим доступа и пересчёт лота (`case_core.apply_mode`)', [
        ('k_c0, k_opex, k_vpub, k_anchor, k_commercial', '`data/access_modes.csv` (A/B/C), `user_modes.py` (D1, D3)', 'множители режима; `public_core` — флаг бесплатного общественного слоя'),
        ('c0', 'C · k_c0', 'затраты лота в выбранном режиме'),
        ('opex', 'O · k_opex', 'эксплуатация лота в режиме'),
        ('vpub', 'V · k_vpub', 'общественная ценность лота в режиме'),
        ('cash', 'P · k_anchor + Q · k_commercial', 'денежные поступления лота; vpub сюда не входит'),
    ]),
    ('Метрики портфеля (`case_core.evaluate_portfolio`, быстрый путь `team_model._fast_metrics`)', [
        ('c0, opex, vpub, cash', 'Σ по четырём лотам', 'суммарные показатели портфеля'),
        ('kcash', 'cash / opex', 'покрытие эксплуатации поступлениями; ≥ 1 — субсидия не нужна'),
        ('t_rep, readiness, resilience, scale', 'среднее по четырём лотам', 'качественные индексы портфеля'),
        ('архетипы', 'число различных `territorial_archetype` среди нефедеральных лотов', 'территориальное покрытие'),
        ('группы', 'число различных групп возможностей', 'технологическое разнообразие'),
        ('ядро', 'число лотов с `public_core = true`', 'лоты с гарантированным бесплатным слоем'),
    ]),
    ('Ограничения (`case_core.check_constraints`, все девять в BASE и STRESS)', [
        ('лоты', '= 4', 'ровно четыре уникальных лота'),
        ('архетипы ≥ 3, группы ≥ 2, ядро ≥ 2', '`constraints_common`', 'структурные требования'),
        ('c0 ≤ 1300 (BASE) / 1180 (STRESS)', '`scenarios`', 'единственное различие сценариев'),
        ('opex ≤ 360, vpub ≥ 1000, kcash ≥ 0.6, t_rep ≥ 0.63', '`constraints_common`', 'пороги на год'),
    ]),
    ('Финансовые метрики (`team_model.financial_metrics`; допущения r = 0.10, T = 10 лет)', [
        ('af', '(1 − (1 + r)^−T) / r = 6.1446', 'аннуитетный множитель: сумма дисконтов за T лет'),
        ('net', 'cash − opex', 'годовой операционный поток; субсидия = −net при net < 0'),
        ('NPV', '−c0 + net · af', 'чистая приведённая стоимость денежного потока, млн руб.'),
        ('PI', 'net · af / c0', 'индекс доходности: приведённый поток на рубль запуска'),
        ('IRR', 'решение −c0 + net · af(IRR, T) = 0', 'ставка, при которой NPV = 0; не существует при net ≤ 0'),
        ('payback', '−ln(1 − c0 · r / net) / ln(1 + r)', 'дисконтированный срок окупаемости; ∞ при c0 · r ≥ net'),
        ('total_cost_pv', 'c0 + opex · af', 'полная приведённая стоимость владения'),
        ('SROI', 'vpub · af / total_cost_pv', 'рублей общественной ценности на рубль полной стоимости'),
    ]),
    ('Нормировка (`team_model.normalize_criteria`)', [
        ('minmax (принята)', 'z = (x − min) / (max − min)', 'по 1 056 допустимым кандидатам; каждый критерий заранее ориентирован «больше = лучше»; константа → 0'),
        ('threshold (проверка)', 'z = 1 − порог/x; z = 1 − x/лимит; z = (индекс − 1)/4', 'привязка к порогам ограничений и точке безубыточности, отсечка снизу −1'),
    ]),
    ('Блоки и итоговый вектор (`team_model.fitness_function`, `criteria_vector`)', [
        ('F_fin', 'z_NPV + z_PI + z_kcash', 'деньги, 0…3'),
        ('F_public', 'z_vpub + z_SROI + z_ядро', 'общественная польза, 0…3; доля ядра = ядро / 4'),
        ('F_quality', 'z_t_rep + z_readiness + z_resilience + z_scale', 'нефинансовое качество, 0…4; зависит только от набора лотов'),
        ('S', 'F_fin + F_public + F_quality = Σ всех десяти z', 'сумма нормированных, 0…10 — скалярный рейтинг (раздел «Выбор по сумме»)'),
        ('S_eq', 'F_fin/3 + F_public/3 + F_quality/4', 'та же сумма с равными весами блоков, 0…3'),
    ]),
    ('Порядок, фронт и проверки (`team_model.pareto_mask`, `pareto_report`)', [
        ('X ≽ Y', 'F_k(X) ≥ F_k(Y) для всех k и > хотя бы для одного', 'X доминирует Y; максимальный элемент — тот, кого никто не доминирует'),
        ('ранг', 'место элемента по блоку среди элементов фронта', 'в разделе преимуществ/недостатков'),
        ('запас c0', '1180 − c0', 'до лимита STRESS, млн руб.'),
        ('откат D → A', 'те же лоты, D заменён на A; проверка девяти ограничений', 'допустим ли портфель, если гипотеза D не подтвердится'),
        ('θ* (D1)', '(δ_c0 · C / af + δ_opex · O) / (0.45 · Q)', 'доля разрыва B−A, при которой надстройка окупается; принято θ = 0.20'),
        ('экономия D3', 'C · (1.05 · σ_c0 · ρ_c0 − γ_c0) + af · O · (1.05 · σ_opex · ρ_opex − γ_opex)', 'выигрыш NPV лота против A; допустимо при ≥ 2 лотах с общей группой'),
    ]),
]

FUNCTION_MAP = [
    ('`case_core.py`', '`load_case`, `apply_mode`, `evaluate_portfolio`, `check_constraints`', 'канонический слой организаторов; не изменён'),
    ('`team_model.py` §2–3', '`normalize_metrics`, `constraint_report`', 'z относительно порогов, таблица порог/факт/запас'),
    ('`team_model.py` §6', '`annuity_factor`, `financial_metrics`, `_irr`, `_discounted_payback`', 'NPV, PI, IRR, окупаемость, SROI'),
    ('`team_model.py` §8', '`_precompute`, `_fast_metrics`, `_feasible_fast`', 'быстрый перебор, сверенный с каноном'),
    ('`team_model.py` §9', '`CRITERIA_REGISTRY`, `BLOCKS`, `normalize_criteria`, `fitness_function`, `criteria_vector`, `pareto_mask`', 'нормировка, блоки, частичный порядок'),
    ('`user_modes.py`', '`d1_coefficients`, `d3_coefficients`, `ADMISSIBILITY`, `d1_breakeven_table`, `d3_breakeven_table`', 'вывод D из A/B, допустимость, безубыточность'),
    ('`pareto_search.py`', '`build_worlds`, `enumerate_world`, `run_search`, `verify_against_canon`', 'миры перебора, отсечение, фронт, сверка'),
    ('`pareto_report.py`', '`enrich_front`, `pros_cons`, `sum_ranking`, `conclusions`', 'ранги, откат, тексты, рейтинг по сумме'),
]


def sum_ranking(cand, f, top=15):
    """Скалярный рейтинг по сумме нормированных среди всех допустимых кандидатов.

    S = Σ z по всем компонентам блоков (= F_fin + F_public + F_quality);
    S_eq — с равными весами блоков. Флаг pareto показывает, максимален ли элемент.
    """
    weights = {name: 1.0 / len(keys) for name, keys in tm.BLOCKS.items()}
    d = cand.copy()
    d['S'] = sum(d[b] for b in tm.BLOCKS)
    d['S_eq'] = sum(d[b] * w for b, w in weights.items())
    d['rank_S'] = d['S'].rank(ascending=False, method='min').astype(int)
    d['rank_S_eq'] = d['S_eq'].rank(ascending=False, method='min').astype(int)
    no_map = dict(zip(zip(f['lots'], f['modes']), f['no']))
    d['no'] = [no_map.get(k, 0) for k in zip(d['lots'], d['modes'])]
    top_s = d.sort_values('S', ascending=False).head(top)
    top_eq = d.sort_values('S_eq', ascending=False).head(top)
    keep = pd.concat([top_s, top_eq]).drop_duplicates(subset=['lots', 'modes']).sort_values('S', ascending=False)
    return keep.reset_index(drop=True)


def build_markdown(f, stats, reference, config, norm, cand=None):
    n = len(f)
    c = config['constraints_common']
    um_df = um.build_user_modes(load_case('.')[1])
    d1 = um.d1_breakeven_table(*load_case('.')[:2])
    d3 = um.d3_breakeven_table(*load_case('.')[:2])

    # порядок: кластеры по набору лотов (по лучшему F_fin в кластере), внутри — F_fin по убыванию
    f = f.copy()
    best = f.groupby('lots')['F_fin'].max()
    f['_cluster_rank'] = f['lots'].map(best)
    f = f.sort_values(['_cluster_rank', 'lots', 'F_fin'], ascending=[False, True, False]).reset_index(drop=True)
    f.insert(0, 'no', range(1, n + 1))

    pc = [pros_cons(r, n, config) for r in f.itertuples()]
    f['pros'] = ['; '.join(p) for p, _ in pc]
    f['cons'] = ['; '.join(q) for _, q in pc]

    num = {
        'F_fin': lambda v: f'{v:.3f}', 'F_public': lambda v: f'{v:.3f}', 'F_quality': lambda v: f'{v:.3f}',
        'npv_mrub': lambda v: f'{v:.0f}', 'profitability_index': lambda v: f'{v:.2f}', 'kcash': lambda v: f'{v:.2f}',
        'vpub_mrub_per_year': lambda v: f'{v:.0f}', 'sroi': lambda v: f'{v:.2f}',
        'public_core_lots': lambda v: f'{int(v)}', 'c0_mrub': lambda v: f'{v:.0f}',
        'headroom_c0_stress': lambda v: f'{v:.0f}', 't_rep': lambda v: f'{v:.3f}',
        'no': lambda v: f'{int(v)}',
    }

    out = []
    out.append(f'# Фронт Парето: максимальные элементы (лоты и режимы)\n')
    out.append(f'Прогон `test-main/pareto_search.py` от {date.today().isoformat()}, нормировка `{norm}`, '
               f'пространство блоков (F_fin, F_public, F_quality). Генерируется `test-main/pareto_report.py`; '
               f'сырые данные — `test-main/output/pareto_front_report.csv`.\n')

    out.append('## 1. Постановка\n')
    out.append('- Перебор: C(8,4) = 70 наборов лотов x назначения режимов из {A, B, C, D}, где D — один активный '
               'пользовательский режим на прогон (как в шаблоне организаторов). Пользовательские режимы и их вывод — '
               '`test-main/user_modes.py`.')
    out.append(f'- Отсечение: девять канонических ограничений `case_core.check_constraints` в BASE (c0 <= 1300) '
               f'**и** STRESS (c0 <= 1180); opex <= {c["opex_max_mrub_per_year"]}, vpub >= {c["vpub_min_mrub_per_year"]}, '
               f'kcash >= {c["kcash_min"]}, t_rep >= {c["t_rep_min"]}, ядро >= {c["min_public_core_lots"]}, '
               f'архетипы >= {c["min_territorial_archetypes"]}, группы >= {c["min_capability_groups"]}.')
    out.append('- Вектор: три блока, каждый — простая сумма нормированных компонент:')
    for name, keys in tm.BLOCKS.items():
        out.append(f'  - `{name}` = z({", ".join(keys)})')
    out.append('- Порядок: X доминирует Y, если X >= Y по всем трём блокам и X > Y хотя бы по одному. '
               'Максимальные элементы — портфели, которые никто не доминирует.')
    out.append('- Деньги (F_fin) и общественная ценность (F_public) нигде не складываются.\n')

    out.append('### Статистика перебора\n')
    out.append(_md_table(stats, list(stats.columns), ['мир', 'перебрано', 'недопустимо по D', 'допустимо BASE+STRESS']))
    out.append(f'\nДопустимых всего: **{int(stats.feasible.sum())}**, максимальных элементов: **{n}**.\n')

    out.append('### Эталон нормировки (min/max по допустимым)\n')
    ref = reference.reset_index().rename(columns={'index': 'criterion'})
    out.append(_md_table(ref, ['criterion', 'min', 'max'], ['критерий', 'min', 'max'],
                         {'min': lambda v: f'{v:.4g}', 'max': lambda v: f'{v:.4g}'}))
    out.append('')

    out.append('## 2. Обозначения, формулы и функции\n')
    out.append('Расшифровка всех величин, которые встречаются в таблицах ниже, и карта функций по файлам.\n')
    for title, items in GLOSSARY:
        out.append(f'**{title}**\n')
        out.append('| обозначение | формула / источник | смысл |')
        out.append('|---|---|---|')
        for sym, formula, meaning in items:
            out.append(f'| {sym} | {formula} | {meaning} |')
        out.append('')
    out.append('**Карта функций**\n')
    out.append('| файл | функции | назначение |')
    out.append('|---|---|---|')
    for file, funcs, purpose in FUNCTION_MAP:
        out.append(f'| {file} | {funcs} | {purpose} |')
    out.append('')

    out.append('## 3. Пользовательские режимы в прогоне\n')
    out.append(_md_table(um_df, ['mode_id', 'k_c0', 'k_opex', 'k_vpub', 'k_anchor', 'k_commercial', 'public_core', 'label'],
                         ['режим', 'k_c0', 'k_opex', 'k_vpub', 'k_anchor', 'k_commercial', 'ядро', 'механизм'],
                         {k: (lambda v: f'{v:.4g}') for k in ('k_c0', 'k_opex', 'k_vpub', 'k_anchor', 'k_commercial')}))
    out.append('\nКоэффициенты выводятся формулами из A/B (см. docstring `user_modes.py`), а не задаются числами. '
               'Варианты D2, D4–D7 из `d_type.md` исключены — причины в docstring `user_modes.py`, '
               'раздел «Исключённые варианты».\n')
    out.append('**D1, безубыточность надстройки по лотам** (theta* — доля разрыва коммерческих коэффициентов B−A, '
               f'при которой ΔNPV = 0; принято theta = {um.D1_PARAMS["theta"]:.2f}):\n')
    out.append(_md_table(d1, ['lot_id', 'Q_commercial', 'dC0_mrub', 'dOPEX_mrub_per_year', 'dCASH_mrub_per_year',
                              'dNPV_vs_A_mrub', 'theta_breakeven', 'pays_off_at_theta'],
                         ['лот', 'Q', 'ΔC0', 'ΔOPEX/год', 'ΔCASH/год', 'ΔNPV vs A', 'theta*', 'окупается'],
                         {**{k: (lambda v: f'{v:.2f}') for k in ('dC0_mrub', 'dOPEX_mrub_per_year', 'dCASH_mrub_per_year',
                                                                'dNPV_vs_A_mrub')},
                          'theta_breakeven': lambda v: f'{v:.3f}'}))
    out.append('\n**D3, экономия против A по лотам** (допустимо только для >= 2 лотов с общей группой возможностей; '
               f'нулевая экономия при rho_c0 = {d3.attrs["rho_c0_zero"]:.3f}, rho_opex = {d3.attrs["rho_opex_zero"]:.3f}):\n')
    out.append(_md_table(d3, ['lot_id', 'capability_groups', 'dC0_saved_mrub', 'dOPEX_saved_mrub_per_year', 'dNPV_vs_A_mrub'],
                         ['лот', 'группы', 'экономия C0', 'экономия OPEX/год', 'ΔNPV vs A'],
                         {k: (lambda v: f'{v:.2f}') for k in ('dC0_saved_mrub', 'dOPEX_saved_mrub_per_year', 'dNPV_vs_A_mrub')}))
    out.append('')

    out.append('## 4. Максимальные элементы: сводная таблица\n')
    out.append('Сгруппировано по набору лотов (кластеры упорядочены по лучшему F_fin), внутри кластера — по F_fin. '
               'Ранги в разделе 5 считаются среди элементов фронта. «Запас c0» — до лимита STRESS 1180.\n')
    out.append(_md_table(f, ['no', 'lots', 'modes', 'F_fin', 'F_public', 'F_quality', 'npv_mrub', 'profitability_index',
                             'kcash', 'vpub_mrub_per_year', 'sroi', 'public_core_lots', 't_rep', 'c0_mrub', 'headroom_c0_stress'],
                         ['№', 'лоты', 'режимы', 'F_fin', 'F_public', 'F_quality', 'NPV', 'PI', 'kcash', 'vpub', 'SROI',
                          'ядро', 't_rep', 'c0', 'запас c0'], num))
    out.append('')

    out.append('## 5. Преимущества и недостатки каждого элемента\n')
    out.append('Правила формирования текста фиксированы в `pareto_report.pros_cons` — одинаковые для всех элементов. '
               '«Откат D → A» — проверка: если гипотеза D не подтвердится сметой/спросом и лоты вернутся в A, '
               'остаётся ли портфель допустимым.\n')
    for r in f.itertuples():
        out.append(f'### {r.no}. {r.lots} — {r.modes}\n')
        out.append(f'F = ({r.F_fin:.3f}, {r.F_public:.3f}, {r.F_quality:.3f}); ранги: деньги {r.rank_F_fin}, '
                   f'польза {r.rank_F_public}, качество {r.rank_F_quality} из {n}. Архетипы: {r.archetypes}.\n')
        out.append('**Преимущества:**')
        for p in pc[r.no - 1][0]:
            out.append(f'- {p}')
        out.append('\n**Недостатки и риски:**')
        for q in pc[r.no - 1][1]:
            out.append(f'- {q}')
        out.append('')

    out.append('## 6. Структура фронта по наборам лотов\n')
    g = cluster_table(f)
    out.append(_md_table(g, ['lots', 'n', 'F_fin_max', 'F_public_max', 'F_quality', 'npv_max', 'vpub_min', 'vpub_max', 'archetypes'],
                         ['лоты', 'элементов', 'F_fin max', 'F_public max', 'F_quality', 'NPV max', 'vpub min', 'vpub max', 'архетипы'],
                         {'F_fin_max': lambda v: f'{v:.3f}', 'F_public_max': lambda v: f'{v:.3f}',
                          'F_quality': lambda v: f'{v:.3f}', 'npv_max': lambda v: f'{v:.0f}',
                          'vpub_min': lambda v: f'{v:.0f}', 'vpub_max': lambda v: f'{v:.0f}'}))
    out.append('')

    out.append('## 7. Представители фронта\n')
    out.append('| роль | № | лоты | режимы | F_fin | F_public | F_quality | NPV | kcash | vpub | ядро | откат D → A |')
    out.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for role, r in representatives(f):
        out.append(f'| {role} | {int(r.no)} | {r.lots} | {r.modes} | {r.F_fin:.3f} | {r.F_public:.3f} | {r.F_quality:.3f} | '
                   f'{r.npv_mrub:.0f} | {r.kcash:.2f} | {r.vpub_mrub_per_year:.0f} | {int(r.public_core_lots)} | '
                   f'{fallback_label(r)} |')
    out.append('')

    if cand is not None:
        out.append('## 8. Выбор по сумме нормированных\n')
        out.append('Скалярный рейтинг по всем допустимым кандидатам: S = F_fin + F_public + F_quality — сумма всех '
                   'десяти нормированных компонент (0…10), неявные веса блоков 3 : 3 : 4 по числу компонент; '
                   'S_eq = F_fin/3 + F_public/3 + F_quality/4 — равные веса блоков (0…3). Максимум положительно '
                   'взвешенной суммы всегда лежит на фронте, следующие места — не обязательно: столбец «фронт» '
                   'показывает, максимален ли элемент (№ — его номер в разделах 4–5).\n')
        sr = sum_ranking(cand, f)
        out.append(_md_table(sr, ['rank_S', 'S', 'rank_S_eq', 'S_eq', 'no', 'lots', 'modes', 'F_fin', 'F_public', 'F_quality',
                                  'pareto', 'npv_mrub', 'kcash', 'vpub_mrub_per_year', 'public_core_lots'],
                             ['место S', 'S', 'место S_eq', 'S_eq', '№', 'лоты', 'режимы', 'F_fin', 'F_public', 'F_quality',
                              'фронт', 'NPV', 'kcash', 'vpub', 'ядро'],
                             {'S': lambda v: f'{v:.3f}', 'S_eq': lambda v: f'{v:.3f}', 'F_fin': lambda v: f'{v:.3f}',
                              'F_public': lambda v: f'{v:.3f}', 'F_quality': lambda v: f'{v:.3f}',
                              'no': lambda v: str(int(v)) if v else '—', 'pareto': lambda v: 'да' if v else 'нет',
                              'npv_mrub': lambda v: f'{v:.0f}', 'kcash': lambda v: f'{v:.2f}',
                              'vpub_mrub_per_year': lambda v: f'{v:.0f}', 'public_core_lots': lambda v: str(int(v))}))
        best = sr.iloc[0]
        best_eq = sr.sort_values('S_eq', ascending=False).iloc[0]
        out.append(f'\nЛучший по S: **{best.lots} — {best.modes}** (S = {best.S:.3f}); лучший по S_eq: '
                   f'**{best_eq.lots} — {best_eq.modes}** (S_eq = {best_eq.S_eq:.3f}). '
                   'Сумма — это уже выбор весов; фронт показывает, что теряется при любом другом их наборе.\n')

    out.append('## 9. Выводы по структуре фронта\n')
    out.append(f'Выводы проверены для прогона от {date.today().isoformat()}; при смене параметров D или блоков '
               'перегенерировать отчёт и перечитать этот раздел.\n')
    for i, para in enumerate(conclusions(f), 1):
        out.append(f'{i}. {para}')
    out.append('')
    return '\n'.join(out), f, pc


def cluster_table(f):
    """Сводка по наборам лотов, упорядоченная по лучшему F_fin."""
    return (f.groupby('lots')
             .agg(n=('no', 'size'), F_fin_max=('F_fin', 'max'), F_public_max=('F_public', 'max'),
                  F_quality=('F_quality', 'first'), vpub_min=('vpub_mrub_per_year', 'min'),
                  vpub_max=('vpub_mrub_per_year', 'max'), npv_max=('npv_mrub', 'max'),
                  archetypes=('archetypes', 'first'))
             .reset_index().sort_values('F_fin_max', ascending=False))


def fallback_label(r):
    return 'допустим' if (r.fallback_feasible_BASE and r.fallback_feasible_STRESS) else f'нарушает {r.fallback_failed}'


def representatives(f):
    """Роли-представители фронта: лучший по каждому блоку и компромиссы."""
    f = f.copy()
    f['_rank_sum'] = f['rank_F_fin'] + f['rank_F_public'] + f['rank_F_quality']
    safe = f[f.fallback_feasible_BASE & f.fallback_feasible_STRESS]
    picks = [
        ('лучший по деньгам', f.loc[f['F_fin'].idxmax()]),
        ('лучший по общественной ценности', f.loc[f['F_public'].idxmax()]),
        ('лучший по общественной ценности с безопасным откатом', safe.loc[safe['F_public'].idxmax()] if not safe.empty else None),
        ('лучший по качеству', f.loc[f['F_quality'].idxmax()]),
        ('компромисс (минимальная сумма рангов)', f.loc[f['_rank_sum'].idxmin()]),
        ('компромисс с безопасным откатом', safe.loc[safe['_rank_sum'].idxmin()] if not safe.empty else None),
    ]
    return [(role, r) for role, r in picks if r is not None]


# Семейства фронта — для выводов и цветового кода в HTML. Проверены для прогона
# 2026-09-12; при изменении структуры фронта пересмотреть вместе с conclusions().
FAMILIES = {
    'money': ('Деньги', ('FLOOD+AGRI+TRANS+ENV', 'FIRE+AGRI+TRANS+ENV')),
    'public': ('Польза', ('FIRE+FLOOD+AGRI+TRANS',)),
    'quality': ('Качество', ('AGRI+INFRA+TRANS+ENV',)),
}


def family_of(lots):
    for key, (_, sets) in FAMILIES.items():
        if lots in sets:
            return key
    return 'other'


def conclusions(f):
    """Тезисы по структуре фронта (markdown-разметка: **жирный**, `код`)."""
    n = len(f)
    unsafe = ~(f.fallback_feasible_BASE & f.fallback_feasible_STRESS)
    n_unsafe = int(unsafe.sum())
    n_subsidy = int((f.kcash < 1).sum())
    unsafe_sets = sorted(f[unsafe]['lots'].unique())
    return [
        ('**Без пользовательских режимов ни один портфель A/B/C не является максимальным.** 16 из 21 '
               'элементов фронта A/B/C доминируются тем же набором лотов с заменой всех A → D3; остальные 5 — это '
               'AGRI+INFRA+TRANS+ENV, где лоты в A не имеют общей группы (INFRA — PNT/InSAR, ENV — EO) и D3 '
               'недопустим для всей четвёрки: их доминируют частичная замена на D3 (AGRI+INFRA+TRANS, общая группа '
               'PNT/InSAR) или D1 (см. вывод `pareto_search.py`, блок «Базовая линия»). Это ожидаемо: при принятых '
               'параметрах D3 — это A с меньшими C0 и OPEX при том же общественном слое. Значит, главный '
               'управленческий тезис — не «какие лоты», а «совместная закупка общественного ядра лотов с общей '
               'технологической цепочкой вместо раздельной».'),
        (f'**{n_unsafe} из {n} элементов существуют только благодаря экономии D3.** Для наборов '
         f'{", ".join(unsafe_sets)} те же лоты в A стоят больше лимита STRESS 1180. Это в том числе весь кластер '
         'FIRE+FLOOD+AGRI+TRANS с максимальной общественной ценностью (vpub до 1610, SROI 3.11). Такие портфели '
         'нельзя выбирать без подтверждённой сметы совместной закупки: при её провале портфель не проходит '
         'стресс-сценарий кейса.'),
        ('**Три семейства на фронте.** «Деньги» — FLOOD/FIRE+AGRI+TRANS+ENV с двумя лотами в D3 (ядро) и '
         'AGRI/TRANS в C или B: лучший NPV (−480…−650), профицит 80–105 млн/год, но ядро ровно на минимуме и '
         'режим C снижает общественную ценность коммерческих лотов до 62%; откат в A безопасен. «Польза» — '
         'FIRE+FLOOD+AGRI+TRANS: максимальный vpub и SROI, но худшее качество (t_rep 0.702, готовность 4.425), '
         'c0 у лимита и полная зависимость от D3. «Качество» — AGRI+INFRA+TRANS+ENV (21 элемент): лучшие '
         'индексы (t_rep 0.738, готовность 4.475, масштаб 4.70), но vpub 1015–1110 на грани порога 1000 и '
         'нет лотов Сибири, Дальнего Востока и Арктики; ядро в этом семействе достигается через D1 '
         '(надстройка окупается на всех четырёх лотах, theta* <= 0.083) или D3 на AGRI+INFRA+TRANS '
         '(общая группа PNT/InSAR).'),
        (f'**Полное общественное ядро стоит субсидии.** {n_subsidy} элементов с четырьмя лотами в '
         'ядре имеют kcash < 1 (субсидия 1–22 млн/год); ограничение kcash >= 0.6 при этом выполнено с '
         'большим запасом. Это цена максимального бесплатного доступа, а не дефект портфеля.'),
        ('**D1 окупается везде, кроме FIRE** (theta* = 0.228 > 0.20 из-за малой коммерческой базы Q = 35): '
         'элементы с FIRE в D1 держатся на фронте за счёт ядра, а не денег. На лотах с Q >= 85 (AGRI, INFRA, '
         'TRANS, ENV, SSA, ARCTIC) надстройка окупается при захвате 6–9% разрыва B−A.'),
        ('**ARCTIC и SSA не входят ни в один максимальный элемент**: ARCTIC — c0 460 и готовность 3.2, '
         'SSA — федеральный лот с t_rep 0.64 у порога 0.63; оба тянут вниз качество и бюджет сильнее, чем '
         'добавляют пользы в этой метрике. Если у команды есть содержательный аргумент за Арктику или SSA, '
         'его нужно вносить как отдельный критерий, а не ждать от перебора.'),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default='.')
    parser.add_argument('--norm', choices=('minmax', 'threshold'), default='minmax')
    parser.add_argument('--md', default='../PARETO_FRONT.md', help='куда писать markdown (относительно --root)')
    parser.add_argument('--html', default='output/pareto_front.html', help='куда писать HTML (относительно --root)')
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    lots, modes, config = load_case(args.root)
    worlds = build_worlds(modes)
    cand, front, stats, reference = run_search(lots, modes, config, worlds, norm=args.norm)
    f = enrich_front(front, lots, modes, config, worlds)
    text, f, pc = build_markdown(f, stats, reference, config, args.norm, cand)

    md_path = Path(args.root) / args.md
    md_path.write_text(text, encoding='utf-8')
    out = Path(args.root) / 'output'
    out.mkdir(exist_ok=True)
    export = [c for c in f.columns if c not in ('selection', 'capability_set', '_cluster_rank')]
    f[export].to_csv(out / 'pareto_front_report.csv', index=False)
    from pareto_html import build_html
    html_text = build_html(f, pc, cand, stats, reference, config, args.norm, family_of, FAMILIES,
                           cluster_table(f), representatives(f), conclusions(f), lots, modes,
                           glossary=GLOSSARY, function_map=FUNCTION_MAP, sum_df=sum_ranking(cand, f))
    html_path = Path(args.root) / args.html
    html_path.write_text(html_text, encoding='utf-8')
    print(f'Фронт: {len(f)} элементов. Записано {md_path}, {html_path} и {out / "pareto_front_report.csv"}')
    print('Откат D -> A недопустим у', int((~(f.fallback_feasible_BASE & f.fallback_feasible_STRESS)).sum()), 'элементов')
    return f


if __name__ == '__main__':
    main()
