"""
team_model.py — расчетный слой команды поверх канонического case_core.py.

Канонический слой не изменяется. Здесь только надстройка:

  1. производные метрики, не требующие новых входных данных;
  2. финансовые метрики, требующие допущений команды (ставка, горизонт);
  3. нормализация показателей и композитная метрика team_score;
  4. сценарные оценки BASE / STRESS и робастный критерий min(BASE, STRESS);
  5. полный перебор допустимых портфелей и поиск оптимума под заданные веса;
  6. анализ чувствительности: меняется ли победитель при сдвиге весов.

Все допущения команды собраны в ASSUMPTIONS и передаются явно.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from case_core import (
    apply_mode,
    check_constraints,
    evaluate_portfolio,
    normalize_capability,
)

# ---------------------------------------------------------------------------
# 0. Допущения команды. НЕ часть исходных данных кейса, требуют обоснования.
# ---------------------------------------------------------------------------

ASSUMPTIONS = {
    # Ставка дисконтирования, доля в год. В case_config.json её нет.
    'discount_rate': 0.10,
    # Горизонт расчёта, лет. В case_config.json его нет.
    'horizon_years': 10,
    # Годовой объём услуги для удельной стоимости. Нет в данных кейса.
    'service_volume_per_year': None,
    # Вероятность нереализации: скаляр или dict lot_id -> p. Нет в данных кейса.
    'risk_prob': None,
}

CRITERIA = ('vpub', 'capex', 'opex', 'kcash', 't_rep', 'readiness', 'resilience', 'scale')
DEFAULT_WEIGHTS = {k: 1.0 / len(CRITERIA) for k in CRITERIA}
SCENARIOS = ('BASE', 'STRESS')
EPS = 1e-9


# ---------------------------------------------------------------------------
# 1. Веса
# ---------------------------------------------------------------------------

def normalize_weights(weights=None):
    """Приводит веса к сумме 1. Пустые значения трактуются как 0."""
    weights = weights or DEFAULT_WEIGHTS
    w = {k: float(weights.get(k) or 0.0) for k in CRITERIA}
    total = sum(w.values())
    if total <= 0:
        raise ValueError('Сумма весов должна быть положительной')
    return {k: v / total for k, v in w.items()}


# ---------------------------------------------------------------------------
# 2. Нормализация показателей относительно порогов ограничений
# ---------------------------------------------------------------------------
# Все восемь z-компонент приведены к виду "больше = лучше" и ограничены сверху
# единицей. z = 0 означает "ровно на пороге", z < 0 — нарушение ограничения.
#
#   затраты (c0, opex):        z = 1 - value / limit
#   пороги снизу (vpub, ...):  z = 1 - floor / value
#   индексы 1..5:              z = (value - 1) / 4
#
# Ограниченность сверху принципиальна: отношение вида limit/value не ограничено
# и при малом value задавило бы остальные слагаемые независимо от весов.

def _z_cost(value, limit):
    if limit <= 0:
        return float('-inf')
    return 1.0 - float(value) / float(limit)


def _z_floor(value, floor):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        return float('-inf')
    if floor <= 0:
        return 1.0
    return 1.0 - float(floor) / value


def _z_index(value):
    return (float(value) - 1.0) / 4.0


def normalize_metrics(metrics, config, scenario='BASE'):
    """Возвращает восемь безразмерных z-компонент для сценария."""
    c = config['constraints_common']
    sc = config['scenarios'][scenario]
    return {
        'vpub': _z_floor(metrics['vpub_mrub_per_year'], c['vpub_min_mrub_per_year']),
        'capex': _z_cost(metrics['c0_mrub'], sc['c0_max_mrub']),
        'opex': _z_cost(metrics['opex_mrub_per_year'], c['opex_max_mrub_per_year']),
        'kcash': _z_floor(metrics['kcash'], c['kcash_min']),
        't_rep': _z_floor(metrics['t_rep'], c['t_rep_min']),
        'readiness': _z_index(metrics['readiness_1_5']),
        'resilience': _z_index(metrics['resilience_1_5']),
        'scale': _z_index(metrics['scale_1_5']),
    }


def team_score(metrics, config, weights=None, scenario='BASE'):
    """Композитная метрика: взвешенная сумма нормированных показателей."""
    w = normalize_weights(weights)
    z = normalize_metrics(metrics, config, scenario)
    return float(sum(w[k] * z[k] for k in CRITERIA if w[k] != 0.0))


def score_scenarios(metrics, config, weights=None):
    """score в обоих сценариях, просадка и робастный критерий min(BASE, STRESS)."""
    base = team_score(metrics, config, weights, 'BASE')
    stress = team_score(metrics, config, weights, 'STRESS')
    drop = base - stress
    return {
        'score_base': base,
        'score_stress': stress,
        'score_drop': drop,
        'score_drop_pct': (100.0 * drop / base) if (base != 0.0 and math.isfinite(base)) else float('nan'),
        'score_robust': min(base, stress),
    }


# ---------------------------------------------------------------------------
# 3. Проверка ограничений с порогами и запасом (порог / факт / запас / PASS)
# ---------------------------------------------------------------------------

def constraint_report(metrics, config, scenario='BASE'):
    """check_constraints + пороговое значение, фактическое значение и запас."""
    c = config['constraints_common']
    sc = config['scenarios'][scenario]
    spec = [
        ('exact_lot_count', '==', c['selected_lots_exactly'], metrics.get('selected_lots')),
        ('territorial_archetypes', '>=', c['min_territorial_archetypes'], metrics.get('territorial_archetypes')),
        ('capability_groups', '>=', c['min_capability_groups'], metrics.get('capability_groups')),
        ('public_core_lots', '>=', c['min_public_core_lots'], metrics.get('public_core_lots')),
        ('c0_limit', '<=', sc['c0_max_mrub'], metrics.get('c0_mrub')),
        ('opex_limit', '<=', c['opex_max_mrub_per_year'], metrics.get('opex_mrub_per_year')),
        ('vpub_floor', '>=', c['vpub_min_mrub_per_year'], metrics.get('vpub_mrub_per_year')),
        ('kcash_floor', '>=', c['kcash_min'], metrics.get('kcash')),
        ('t_rep_floor', '>=', c['t_rep_min'], metrics.get('t_rep')),
    ]
    rows = []
    for name, op, threshold, actual in spec:
        if actual is None:
            ok, margin = False, float('nan')
        elif op == '==':
            ok, margin = (actual == threshold), float(actual - threshold)
        elif op == '>=':
            ok, margin = (actual >= threshold - EPS), float(actual - threshold)
        else:
            ok, margin = (actual <= threshold + EPS), float(threshold - actual)
        rows.append({
            'constraint': name, 'op': op, 'threshold': threshold,
            'actual': actual, 'margin': margin, 'ok': bool(ok),
        })
    return pd.DataFrame(rows)


def is_feasible(metrics, config, scenario='BASE'):
    return bool(check_constraints(metrics, config, scenario)['ok'].all())


# ---------------------------------------------------------------------------
# 4. Детализация портфеля с разложением cash на якорную и коммерческую части
# ---------------------------------------------------------------------------

def portfolio_detail(selection, lots, modes):
    """Канонический apply_mode + разложение cash на две компоненты."""
    lidx = lots.set_index('lot_id', drop=False)
    midx = modes.set_index('mode_id', drop=False)
    rows = []
    for lot_id, mode_id in selection:
        lot, mode = lidx.loc[lot_id], midx.loc[mode_id]
        row = apply_mode(lot, mode)
        row['anchor_cash_mrub_per_year'] = float(lot.anchor_cash_mrub_per_year) * float(mode.k_anchor)
        row['commercial_cash_mrub_per_year'] = float(lot.commercial_cash_mrub_per_year) * float(mode.k_commercial)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Производные метрики, не требующие новых входных данных
# ---------------------------------------------------------------------------

def _safe_div(a, b):
    a, b = float(a), float(b)
    return a / b if b != 0.0 else float('nan')


def portfolio_extras(detail, metrics):
    """Удельные показатели, структура денежных потоков, диверсификация."""
    c0 = metrics['c0_mrub']
    opex = metrics['opex_mrub_per_year']
    vpub = metrics['vpub_mrub_per_year']
    cash = metrics['cash_mrub_per_year']
    n = metrics['selected_lots']

    anchor = float(detail['anchor_cash_mrub_per_year'].sum()) if 'anchor_cash_mrub_per_year' in detail else float('nan')
    commercial = float(detail['commercial_cash_mrub_per_year'].sum()) if 'commercial_cash_mrub_per_year' in detail else float('nan')

    counts = {}
    for s in detail['capability_groups']:
        for token in str(s).split(';'):
            for group in normalize_capability(token):
                counts[group] = counts.get(group, 0) + 1
    total_tokens = sum(counts.values())
    hhi = sum((v / total_tokens) ** 2 for v in counts.values()) if total_tokens else float('nan')

    def cv(column):
        values = detail[column].to_numpy(dtype=float)
        mean = values.mean()
        return float(values.std(ddof=0) / mean) if mean else float('nan')

    return {
        # удельная отдача
        'vpub_per_c0': _safe_div(vpub, c0),
        'vpub_per_opex': _safe_div(vpub, opex),
        'cash_per_c0': _safe_div(cash, c0),
        # структура финансирования
        'subsidy_need_mrub_per_year': float(opex - cash),
        'subsidy_share_of_opex': _safe_div(opex - cash, opex),
        'anchor_cash_mrub_per_year': anchor,
        'commercial_cash_mrub_per_year': commercial,
        'anchor_share_of_cash': _safe_div(anchor, cash),
        'commercial_share_of_cash': _safe_div(commercial, cash),
        # структура портфеля
        'public_core_share': _safe_div(metrics['public_core_lots'], n),
        'c0_per_lot_mrub': _safe_div(c0, n),
        'opex_per_lot_mrub': _safe_div(opex, n),
        # диверсификация и разброс
        'hhi_capability': hhi,
        'cv_c0': cv('c0_mrub'),
        'cv_opex': cv('opex_mrub_per_year'),
        'cv_vpub': cv('vpub_mrub_per_year'),
    }


def scenario_extras(metrics, config, scenario='BASE'):
    """Запас по каждому ограничению и загрузка лимитов для конкретного сценария."""
    c = config['constraints_common']
    sc = config['scenarios'][scenario]
    c0 = metrics['c0_mrub']
    opex = metrics['opex_mrub_per_year']
    return {
        'headroom_c0_mrub': float(sc['c0_max_mrub'] - c0),
        'headroom_opex_mrub': float(c['opex_max_mrub_per_year'] - opex),
        'headroom_vpub_mrub': float(metrics['vpub_mrub_per_year'] - c['vpub_min_mrub_per_year']),
        'headroom_kcash': float(metrics['kcash'] - c['kcash_min']),
        'headroom_t_rep': float(metrics['t_rep'] - c['t_rep_min']),
        'c0_utilization_pct': 100.0 * _safe_div(c0, sc['c0_max_mrub']),
        'opex_utilization_pct': 100.0 * _safe_div(opex, c['opex_max_mrub_per_year']),
        'feasible': is_feasible(metrics, config, scenario),
    }


# ---------------------------------------------------------------------------
# 6. Финансовые метрики. Требуют допущений команды (ставка, горизонт).
# ---------------------------------------------------------------------------
# Модель денежного потока:
#   t = 0        : -c0
#   t = 1..T     : cash - opex (постоянный аннуитет)
# Поток vpub считается отдельно и НИКОГДА не складывается с cash.

def annuity_factor(rate, years):
    """Сумма 1/(1+r)^t по t = 1..T."""
    if years <= 0:
        return 0.0
    if abs(rate) < 1e-12:
        return float(years)
    return (1.0 - (1.0 + rate) ** (-years)) / rate


def _irr(c0, net, years, lo=-0.9999, hi=10.0, iters=400, tol=1e-10):
    """IRR методом бисекции. nan, если знак не меняется на интервале."""
    if net <= 0 or years <= 0 or c0 <= 0:
        return float('nan')
    f = lambda r: -c0 + net * annuity_factor(r, years)
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        return float('nan')
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))


def _discounted_payback(c0, net, rate):
    """Дисконтированный срок окупаемости, непрерывная интерполяция по годам."""
    if net <= 0 or c0 <= 0:
        return float('nan')
    if abs(rate) < 1e-12:
        return float(c0 / net)
    k = c0 * rate / net
    if k >= 1.0:
        return float('inf')  # дисконтированный поток никогда не покроет c0
    return float(-math.log(1.0 - k) / math.log(1.0 + rate))


def financial_metrics(metrics, discount_rate=None, horizon_years=None, service_volume_per_year=None):
    """NPV, IRR, окупаемость, PI, ROI, SROI и удельная стоимость услуги."""
    rate = ASSUMPTIONS['discount_rate'] if discount_rate is None else float(discount_rate)
    years = ASSUMPTIONS['horizon_years'] if horizon_years is None else int(horizon_years)
    volume = ASSUMPTIONS['service_volume_per_year'] if service_volume_per_year is None else service_volume_per_year

    c0 = float(metrics['c0_mrub'])
    opex = float(metrics['opex_mrub_per_year'])
    cash = float(metrics['cash_mrub_per_year'])
    vpub = float(metrics['vpub_mrub_per_year'])
    net = cash - opex

    af = annuity_factor(rate, years)
    pv_cash, pv_opex, pv_vpub, pv_net = cash * af, opex * af, vpub * af, net * af
    total_cost_pv = c0 + pv_opex
    npv = -c0 + pv_net
    payback_simple = float(c0 / net) if net > 0 else float('inf')
    payback_disc = _discounted_payback(c0, net, rate)

    return {
        'discount_rate': rate,
        'horizon_years': years,
        'net_cash_flow_mrub_per_year': net,
        'pv_cash_mrub': pv_cash,
        'pv_opex_mrub': pv_opex,
        'pv_vpub_mrub': pv_vpub,
        'total_cost_pv_mrub': total_cost_pv,
        'npv_mrub': npv,
        'npv_per_c0': _safe_div(npv, c0),
        'irr': _irr(c0, net, years),
        'profitability_index': _safe_div(pv_net, c0),
        'payback_simple_years': payback_simple,
        'payback_discounted_years': payback_disc,
        'payback_within_horizon': bool(payback_disc <= years),
        'roi_horizon': _safe_div(net * years - c0, c0),
        'sroi': _safe_div(pv_vpub, total_cost_pv),
        'lcos_per_unit': _safe_div(total_cost_pv, volume * af) if volume else float('nan'),
    }


def risk_adjusted_metrics(detail, metrics, risk_prob=None):
    """Ожидаемые значения vpub и cash при вероятности нереализации по лоту.

    risk_prob: скаляр для всего портфеля или dict lot_id -> p.
    Допущение: риск бьёт по результату и поступлениям, но не снимает OPEX.
    """
    risk_prob = ASSUMPTIONS['risk_prob'] if risk_prob is None else risk_prob
    if risk_prob is None:
        return {}
    if isinstance(risk_prob, dict):
        probs = detail['lot_id'].map(lambda x: float(risk_prob.get(x, 0.0))).to_numpy(dtype=float)
    else:
        probs = np.full(len(detail), float(risk_prob))
    survive = 1.0 - probs
    vpub_ra = float((detail['vpub_mrub_per_year'].to_numpy(dtype=float) * survive).sum())
    cash_ra = float((detail['cash_mrub_per_year'].to_numpy(dtype=float) * survive).sum())
    opex = float(metrics['opex_mrub_per_year'])
    return {
        'vpub_risk_adjusted_mrub_per_year': vpub_ra,
        'cash_risk_adjusted_mrub_per_year': cash_ra,
        'kcash_risk_adjusted': _safe_div(cash_ra, opex),
        'vpub_risk_loss_mrub_per_year': float(metrics['vpub_mrub_per_year']) - vpub_ra,
    }


# ---------------------------------------------------------------------------
# 7. Полный расчёт одного портфеля
# ---------------------------------------------------------------------------

def evaluate_full(selection, lots, modes, config, weights=None, assumptions=None):
    """Канонические метрики + все надстройки + score по обоим сценариям."""
    params = {**ASSUMPTIONS, **(assumptions or {})}
    selection = [tuple(x) for x in selection]
    detail = portfolio_detail(selection, lots, modes)
    _, metrics = evaluate_portfolio(selection, lots, modes, config)

    result = {
        'selection': selection,
        'detail': detail,
        'metrics': dict(metrics),
        'extras': portfolio_extras(detail, metrics),
        'financial': financial_metrics(
            metrics,
            discount_rate=params['discount_rate'],
            horizon_years=params['horizon_years'],
            service_volume_per_year=params['service_volume_per_year'],
        ),
        'risk': risk_adjusted_metrics(detail, metrics, params['risk_prob']),
        'score': score_scenarios(metrics, config, weights),
        'weights': normalize_weights(weights),
        'assumptions': params,
        'scenarios': {},
        'constraints': {},
        'z': {},
    }
    for scenario in SCENARIOS:
        result['scenarios'][scenario] = scenario_extras(metrics, config, scenario)
        result['constraints'][scenario] = constraint_report(metrics, config, scenario)
        result['z'][scenario] = normalize_metrics(metrics, config, scenario)
    return result


def flatten_result(result):
    """Плоский словарь всех числовых выходов — для таблиц и экспорта."""
    flat = {
        'selection': '+'.join(f'{lot}:{mode}' for lot, mode in result['selection']),
    }
    for key, value in result['metrics'].items():
        flat[key] = ';'.join(value) if isinstance(value, list) else value
    flat.update(result['extras'])
    flat.update(result['financial'])
    flat.update(result['risk'])
    flat.update(result['score'])
    for scenario in SCENARIOS:
        for key, value in result['scenarios'][scenario].items():
            flat[f'{key}_{scenario}'] = value
        for key, value in result['z'][scenario].items():
            flat[f'z_{key}_{scenario}'] = value
    return flat


# ---------------------------------------------------------------------------
# 8. Полный перебор допустимых портфелей и поиск оптимума
# ---------------------------------------------------------------------------
# Пространство поиска конечно и мало: C(8,4) сочетаний лотов на число режимов
# в четвёртой степени. Перебор точный, эвристики не нужны.

def _precompute(lots, modes):
    lidx = lots.set_index('lot_id', drop=False)
    midx = modes.set_index('mode_id', drop=False)
    cache = {}
    for lot_id in lidx.index:
        for mode_id in midx.index:
            row = apply_mode(lidx.loc[lot_id], midx.loc[mode_id])
            caps = set()
            for token in str(row['capability_groups']).split(';'):
                caps |= normalize_capability(token)
            row['_caps'] = frozenset(caps)
            cache[(lot_id, mode_id)] = row
    return cache


def _fast_metrics(combo, cache):
    """Повторяет агрегацию evaluate_portfolio без пересборки DataFrame."""
    rows = [cache[pair] for pair in combo]
    n = len(rows)
    opex = sum(r['opex_mrub_per_year'] for r in rows)
    cash = sum(r['cash_mrub_per_year'] for r in rows)
    caps = set().union(*(r['_caps'] for r in rows))
    territorial = {r['territorial_archetype'] for r in rows if not r['federal']}
    return {
        'selected_lots': len({r['lot_id'] for r in rows}),
        'c0_mrub': float(sum(r['c0_mrub'] for r in rows)),
        'opex_mrub_per_year': float(opex),
        'vpub_mrub_per_year': float(sum(r['vpub_mrub_per_year'] for r in rows)),
        'cash_mrub_per_year': float(cash),
        'kcash': float(cash / opex) if opex else float('nan'),
        't_rep': float(sum(r['t_rep'] for r in rows) / n),
        'readiness_1_5': float(sum(r['readiness_1_5'] for r in rows) / n),
        'resilience_1_5': float(sum(r['resilience_1_5'] for r in rows) / n),
        'scale_1_5': float(sum(r['scale_1_5'] for r in rows) / n),
        'territorial_archetypes': len(territorial),
        'capability_groups': len(caps),
        'capability_set': sorted(caps),
        'public_core_lots': int(sum(1 for r in rows if r['public_core'])),
    }


def _feasible_fast(m, config, scenario):
    c = config['constraints_common']
    sc = config['scenarios'][scenario]
    return (
        m['selected_lots'] == c['selected_lots_exactly']
        and m['territorial_archetypes'] >= c['min_territorial_archetypes']
        and m['capability_groups'] >= c['min_capability_groups']
        and m['public_core_lots'] >= c['min_public_core_lots']
        and m['c0_mrub'] <= sc['c0_max_mrub'] + EPS
        and m['opex_mrub_per_year'] <= c['opex_max_mrub_per_year'] + EPS
        and m['vpub_mrub_per_year'] >= c['vpub_min_mrub_per_year'] - EPS
        and m['kcash'] >= c['kcash_min'] - EPS
        and m['t_rep'] >= c['t_rep_min'] - EPS
    )


def enumerate_portfolios(lots, modes, config, weights=None, mode_ids=None,
                         require_feasible=('BASE', 'STRESS'), include_infeasible=False):
    """Перебирает все сочетания 4 лотов и режимов, считает score для каждого."""
    cache = _precompute(lots, modes)
    lot_ids = list(lots['lot_id'])
    mode_ids = list(mode_ids) if mode_ids is not None else list(modes['mode_id'])
    w = normalize_weights(weights)
    require_feasible = tuple(require_feasible or ())

    rows = []
    for lots4 in itertools.combinations(lot_ids, 4):
        for modes4 in itertools.product(mode_ids, repeat=4):
            combo = tuple(zip(lots4, modes4))
            m = _fast_metrics(combo, cache)
            feasible = {s: _feasible_fast(m, config, s) for s in SCENARIOS}
            if not include_infeasible and not all(feasible[s] for s in require_feasible):
                continue
            row = {
                'selection': combo,
                'lots': '+'.join(lots4),
                'modes': ''.join(modes4),
                'feasible_BASE': feasible['BASE'],
                'feasible_STRESS': feasible['STRESS'],
            }
            row.update(score_scenarios(m, config, w))
            row.update({k: m[k] for k in (
                'c0_mrub', 'opex_mrub_per_year', 'vpub_mrub_per_year', 'cash_mrub_per_year',
                'kcash', 't_rep', 'readiness_1_5', 'resilience_1_5', 'scale_1_5',
                'territorial_archetypes', 'capability_groups', 'public_core_lots')})
            rows.append(row)
    return pd.DataFrame(rows)


def optimize(lots, modes, config, weights=None, objective='score_robust', top_n=10, **kwargs):
    """Возвращает top_n лучших допустимых портфелей под заданные веса.

    objective: score_robust (min по сценариям), score_base или score_stress.
    """
    df = enumerate_portfolios(lots, modes, config, weights=weights, **kwargs)
    if df.empty:
        return df
    return df.sort_values(objective, ascending=False).head(top_n).reset_index(drop=True)


def weight_sensitivity(lots, modes, config, weights=None, key='vpub', span=0.2, steps=9,
                       objective='score_robust', **kwargs):
    """Сдвигает один вес на +-span и смотрит, меняется ли победитель.

    После сдвига веса нормируются заново, поэтому сумма остаётся равной 1.
    """
    base = normalize_weights(weights)
    if key not in CRITERIA:
        raise ValueError(f'Неизвестный критерий: {key}')
    rows = []
    winner_0 = None
    for factor in np.linspace(1.0 - span, 1.0 + span, steps):
        w = dict(base)
        w[key] = base[key] * float(factor)
        w = normalize_weights(w)
        best = optimize(lots, modes, config, weights=w, objective=objective, top_n=1, **kwargs)
        if best.empty:
            rows.append({'factor': float(factor), 'weight_value': w[key], 'winner': None,
                         objective: float('nan'), 'winner_changed': None})
            continue
        winner = f"{best.loc[0, 'lots']}|{best.loc[0, 'modes']}"
        if winner_0 is None and abs(factor - 1.0) < 1e-12:
            winner_0 = winner
        rows.append({'factor': float(factor), 'weight_value': w[key], 'winner': winner,
                     objective: float(best.loc[0, objective]), 'winner_changed': None})
    if winner_0 is None and rows:
        winner_0 = rows[len(rows) // 2]['winner']
    for row in rows:
        row['winner_changed'] = (row['winner'] != winner_0) if row['winner'] else None
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 9. Демонстрационный прогон
# ---------------------------------------------------------------------------

def _demo():
    from case_core import load_case
    lots, modes, config = load_case('.')
    weights = DEFAULT_WEIGHTS

    print('=== Полный перебор допустимых портфелей ===')
    allp = enumerate_portfolios(lots, modes, config, weights)
    print('Допустимых в BASE и STRESS:', len(allp))

    top = optimize(lots, modes, config, weights, top_n=5)
    print(top[['lots', 'modes', 'score_base', 'score_stress', 'score_drop',
               'score_robust', 'c0_mrub', 'vpub_mrub_per_year', 'kcash']].round(4).to_string(index=False))

    best = list(top.loc[0, 'selection'])
    print('\n=== Полный расчёт оптимума:', best, '===')
    result = evaluate_full(best, lots, modes, config, weights)
    flat = flatten_result(result)
    for key, value in flat.items():
        print(f'{key:40s} {value}')

    print('\n=== Ограничения BASE ===')
    print(result['constraints']['BASE'].to_string(index=False))
    print('\n=== Ограничения STRESS ===')
    print(result['constraints']['STRESS'].to_string(index=False))

    print('\n=== Чувствительность к весу vpub ===')
    print(weight_sensitivity(lots, modes, config, weights, key='vpub').to_string(index=False))


if __name__ == '__main__':
    _demo()
