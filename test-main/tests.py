"""
tests.py — контрольные примеры для Т1 (корректность расчётов) и Т3 (проверка
ограничений). Без внешних зависимостей кроме pandas/numpy, уже используемых
проектом. Запуск: `python tests.py` (из test-main). Падает с AssertionError и
ненулевым кодом возврата при первом нарушении; при успехе печатает сводку.

Ничего в case_core.py не подменяется и не мокается — тесты гоняют реальный
канонический слой и надстройку team_model.py/user_modes.py поверх него.
"""
from __future__ import annotations

import itertools
import math
import sys

import numpy as np
import pandas as pd

from case_core import apply_mode, check_constraints, evaluate_portfolio, load_case
import team_model as tm
import user_modes as um

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


# ---------------------------------------------------------------------------
# Т1: быстрый путь перебора сверен с каноническим слоем на ВСЕХ комбинациях
# ---------------------------------------------------------------------------

@check('быстрый путь совпадает с case_core на всех 5670 комбинациях A/B/C')
def test_fast_path_matches_canon_full_abc(ctx):
    lots, modes, config = ctx['lots'], ctx['modes'], ctx['config']
    cache = tm._precompute(lots, modes)
    max_err = 0.0
    n = 0
    for lots4 in itertools.combinations(lots['lot_id'], 4):
        for modes4 in itertools.product(modes['mode_id'], repeat=4):
            combo = tuple(zip(lots4, modes4))
            fast = tm._fast_metrics(combo, cache)
            _, canon = evaluate_portfolio(list(combo), lots, modes, config)
            for key in ('c0_mrub', 'opex_mrub_per_year', 'vpub_mrub_per_year', 'cash_mrub_per_year',
                       'kcash', 't_rep', 'readiness_1_5', 'resilience_1_5', 'scale_1_5'):
                max_err = max(max_err, abs(float(fast[key]) - float(canon[key])))
            for key in ('selected_lots', 'territorial_archetypes', 'capability_groups', 'public_core_lots'):
                assert int(fast[key]) == int(canon[key]), f'{combo}: {key} {fast[key]} != {canon[key]}'
            assert fast['capability_set'] == canon['capability_set'], combo
            for scenario in tm.SCENARIOS:
                fast_ok = tm._feasible_fast(fast, config, scenario)
                canon_ok = bool(check_constraints(canon, config, scenario)['ok'].all())
                assert fast_ok == canon_ok, f'{combo} {scenario}: feasibility {fast_ok} != {canon_ok}'
            n += 1
    assert n == 5670, f'ожидалось 5670 комбинаций, получено {n}'
    assert max_err < 1e-9, f'максимальная ошибка метрик {max_err}'
    return f'{n} комбинаций, max |ошибка| = {max_err:.2e}'


@check('constraint_report.ok совпадает с check_constraints.ok на выборке D1/D3')
def test_constraint_report_matches_check_constraints(ctx):
    lots, modes, config = ctx['lots'], ctx['modes'], ctx['config']
    modes_d = pd.concat([modes, pd.DataFrame([um.d1_coefficients(modes), um.d3_coefficients(modes)])],
                        ignore_index=True)
    rng = np.random.default_rng(0)
    lot_ids = list(lots['lot_id'])
    mode_ids = list(modes_d['mode_id'])
    n = 0
    for _ in range(300):
        sel = list(zip(rng.choice(lot_ids, 4, replace=False), rng.choice(mode_ids, 4)))
        _, m = evaluate_portfolio(sel, lots, modes_d, config)
        for scenario in tm.SCENARIOS:
            a = check_constraints(m, config, scenario).set_index('constraint')['ok']
            b = tm.constraint_report(m, config, scenario).set_index('constraint')['ok']
            assert (a == b).all(), f'{sel} {scenario}: {a[a != b].to_dict()}'
        n += 1
    return f'{n} случайных портфелей x {len(tm.SCENARIOS)} сценария'


@check('граница t_rep = 0.63 проходит (>=, не >)')
def test_t_rep_boundary(ctx):
    config = ctx['config']
    m = {'selected_lots': 4, 'c0_mrub': 1, 'opex_mrub_per_year': 1, 'vpub_mrub_per_year': 10000,
         'kcash': 100, 't_rep': 0.63, 'territorial_archetypes': 3, 'capability_groups': 2, 'public_core_lots': 2}
    assert bool(check_constraints(m, config, 'BASE')['ok'].all())
    m['t_rep'] = 0.6299999
    assert not bool(check_constraints(m, config, 'BASE')['ok'].all())
    return 't_rep=0.63 -> PASS, t_rep=0.6299999 -> FAIL'


@check('c0 = лимит сценария проходит; лимит + eps не проходит')
def test_c0_boundary_both_scenarios(ctx):
    config = ctx['config']
    base = {'selected_lots': 4, 'opex_mrub_per_year': 1, 'vpub_mrub_per_year': 10000, 'kcash': 100,
            't_rep': 1.0, 'territorial_archetypes': 3, 'capability_groups': 2, 'public_core_lots': 2}
    for scenario in tm.SCENARIOS:
        limit = config['scenarios'][scenario]['c0_max_mrub']
        m = dict(base, c0_mrub=limit)
        assert bool(check_constraints(m, config, scenario)['ok'].all()), f'{scenario} на границе должен пройти'
        m2 = dict(base, c0_mrub=limit + 1.0)
        assert not bool(check_constraints(m2, config, scenario)['ok'].all()), f'{scenario} за границей не должен пройти'
    return 'BASE 1300 PASS / 1301 FAIL; STRESS 1180 PASS / 1181 FAIL'


# ---------------------------------------------------------------------------
# Т1: внутренняя согласованность финансовых формул
# ---------------------------------------------------------------------------

@check('NPV в точке IRR равен нулю')
def test_npv_zero_at_irr(ctx):
    lots, modes, config = ctx['lots'], ctx['modes'], ctx['config']
    cases = 0
    for lots4 in itertools.islice(itertools.combinations(lots['lot_id'], 4), 0, None):
        for modes4 in itertools.product(modes['mode_id'], repeat=4):
            sel = list(zip(lots4, modes4))
            _, m = evaluate_portfolio(sel, lots, modes, config)
            fin = tm.financial_metrics(m)
            if not math.isfinite(fin['irr']):
                continue
            af = tm.annuity_factor(fin['irr'], fin['horizon_years'])
            npv_at_irr = -m['c0_mrub'] + fin['net_cash_flow_mrub_per_year'] * af
            assert abs(npv_at_irr) < 1e-6, f'{sel}: NPV(IRR)={npv_at_irr}'
            cases += 1
            if cases >= 200:
                return f'{cases} портфелей с определённым IRR, |NPV(IRR)| < 1e-6'
    return f'{cases} портфелей с определённым IRR'


@check('дисконтированная окупаемость восстанавливает c0')
def test_discounted_payback_recovers_c0(ctx):
    lots, modes, config = ctx['lots'], ctx['modes'], ctx['config']
    cases = 0
    for lots4 in itertools.islice(itertools.combinations(lots['lot_id'], 4), 0, 30):
        for modes4 in itertools.product(modes['mode_id'], repeat=4):
            sel = list(zip(lots4, modes4))
            _, m = evaluate_portfolio(sel, lots, modes, config)
            fin = tm.financial_metrics(m)
            T = fin['payback_discounted_years']
            if not math.isfinite(T):
                continue
            net, r, c0 = fin['net_cash_flow_mrub_per_year'], fin['discount_rate'], m['c0_mrub']
            recovered = net * tm.annuity_factor(r, T)
            assert abs(recovered - c0) < 1e-6, f'{sel}: net*af(T)={recovered} != c0={c0}'
            cases += 1
    assert cases > 0, 'не нашлось ни одного портфеля с конечной окупаемостью в первых 30 наборах лотов'
    return f'{cases} портфелей, net·af(T_payback) == c0 с точностью 1e-6'


@check('SROI не зависит от cash (деньги и общественная ценность не смешаны)')
def test_sroi_independent_of_cash(ctx):
    lots, modes, config = ctx['lots'], ctx['modes'], ctx['config']
    sel = [('FIRE', 'A'), ('AGRI', 'B'), ('TRANS', 'C'), ('ENV', 'A')]
    _, m = evaluate_portfolio(sel, lots, modes, config)
    sroi_a = tm.financial_metrics(m)['sroi']
    m2 = dict(m)
    m2['cash_mrub_per_year'] = m['cash_mrub_per_year'] * 5  # варьируем cash впятеро
    sroi_b = tm.financial_metrics(m2)['sroi']
    assert abs(sroi_a - sroi_b) < 1e-9, f'{sroi_a} != {sroi_b}'
    return f'SROI = {sroi_a:.4f} неизменен при 5x cash'


@check('score_drop >= 0 и score_robust == min(base, stress) на выборке')
def test_score_robust(ctx):
    lots, modes, config = ctx['lots'], ctx['modes'], ctx['config']
    rng = np.random.default_rng(1)
    lot_ids = list(lots['lot_id'])
    mode_ids = list(modes['mode_id'])
    n = 0
    for _ in range(200):
        sel = list(zip(rng.choice(lot_ids, 4, replace=False), rng.choice(mode_ids, 4)))
        _, m = evaluate_portfolio(sel, lots, modes, config)
        sc = tm.score_scenarios(m, config)
        assert sc['score_drop'] >= -1e-12, f'{sel}: score_drop={sc["score_drop"]}'
        assert abs(sc['score_robust'] - min(sc['score_base'], sc['score_stress'])) < 1e-12
        n += 1
    return f'{n} случайных портфелей, score_drop >= 0 и score_robust == min(base, stress)'


# ---------------------------------------------------------------------------
# Раздел 9: нормировка, блоки, фронт Парето
# ---------------------------------------------------------------------------

@check('minmax-нормировка: [0,1] на данных, 0 при равных значениях столбца')
def test_normalize_minmax_bounds(ctx):
    raw = pd.DataFrame({'a': [1.0, 5.0, 3.0], 'b': [2.0, 2.0, 2.0], 'c': [float('nan'), 1.0, 3.0]})
    z = tm.normalize_criteria(raw, method='minmax')
    assert z['a'].min() == 0.0 and z['a'].max() == 1.0
    assert (z['b'] == 0.0).all(), 'константный столбец должен давать 0'
    assert z['c'].iloc[0] == 0.0, 'nan должен давать 0 (нет информации = не лучшее)'
    return 'диапазон [0,1], константа -> 0, nan -> 0'


@check('pareto_mask: находит ровно точки без строгого доминатора')
def test_pareto_mask_known_case(ctx):
    V = np.array([
        [1.0, 1.0],  # доминирует (0.5,0.5)
        [0.5, 0.5],  # доминируется (1,1)
        [2.0, 0.0],  # максимален (никто не >= по обеим)
        [0.0, 2.0],  # максимален
        [1.0, 1.0],  # дубликат первой — обе остаются (не строго доминируют друг друга)
    ])
    mask = tm.pareto_mask(V)
    assert list(mask) == [True, False, True, True, True], mask
    return f'ожидаемая маска {[True, False, True, True, True]} получена'


@check('D3 допустим только для >=2 лотов с общей группой возможностей')
def test_d3_admissibility(ctx):
    lots = ctx['lots'].set_index('lot_id', drop=False)
    assert um.d3_admissible([lots.loc['FIRE'], lots.loc['ENV']])  # оба EO
    assert not um.d3_admissible([lots.loc['FIRE']])  # один лот
    assert not um.d3_admissible([lots.loc['ARCTIC'], lots.loc['SSA']])  # SATCOM;PNT/InSAR vs SSA — нет общей
    return 'FIRE+ENV (общая EO) — да; один лот — нет; ARCTIC+SSA (нет общей группы) — нет'


@check('нормировка весов: сумма 1, нулевая сумма — ошибка')
def test_normalize_weights(ctx):
    w = tm.normalize_weights({'vpub': 2.0, 'capex': 2.0})
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert abs(w['vpub'] - 0.5) < 1e-12
    try:
        tm.normalize_weights({k: 0.0 for k in tm.CRITERIA})
        raised = False
    except ValueError:
        raised = True
    assert raised, 'нулевая сумма весов должна поднимать ValueError'
    return 'сумма нормируется к 1; нулевая сумма весов -> ValueError'


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    lots, modes, config = load_case('.')
    ctx = {'lots': lots, 'modes': modes, 'config': config}

    print(f'=== {len(CHECKS)} контрольных примеров ===\n')
    failed = 0
    for name, fn in CHECKS:
        try:
            detail = fn(ctx)
            print(f'[OK]   {name}\n       {detail}')
        except AssertionError as e:
            failed += 1
            print(f'[FAIL] {name}\n       {e}')
    print(f'\n{len(CHECKS) - failed}/{len(CHECKS)} пройдено')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
