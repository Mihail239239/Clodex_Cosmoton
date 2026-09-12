"""
pareto_search.py — полный перебор портфелей по всем режимам доступа,
отсечение по каноническим ограничениям и поиск фронта Парето.

Что делает:
  1. строит миры перебора: канонические A/B/C и по одному пользовательскому
     режиму из user_modes.py (D1, D3 — коэффициенты выводятся формулами из A/B;
     в шаблоне организаторов активна ровно одна строка D, поэтому в одном
     портфеле не смешиваются разные D); для каждого D проверяется формальное
     условие допустимости (user_modes.ADMISSIBILITY);
  2. перебирает C(8,4) сочетаний лотов x назначения режимов в каждом мире;
  3. отсекает портфели, не проходящие девять канонических ограничений в BASE
     и STRESS (case_core.check_constraints через быстрый путь team_model);
  4. считает финансовые метрики, нормирует критерии и собирает итоговый вектор
     (F_fin, F_public, F_quality) — см. team_model, раздел 9;
  5. находит максимальные элементы по частичному порядку "больше = лучше по
     всем координатам" и сохраняет таблицы в output/.

Запуск (из test-main):
  python pareto_search.py                      # все миры, minmax, блоки
  python pareto_search.py --norm threshold     # проверка устойчивости фронта
  python pareto_search.py --space full         # Парето по всем 10 компонентам
  python pareto_search.py --worlds D1          # только выбранные D
  python pareto_search.py --no-d               # только A/B/C
  python pareto_search.py --verify             # сверка с каноническим evaluate_portfolio
"""
from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from case_core import check_constraints, evaluate_portfolio, load_case
import team_model as tm
from user_modes import ADMISSIBILITY, build_user_modes

# ---------------------------------------------------------------------------
# Перебор
# ---------------------------------------------------------------------------

def build_worlds(modes, user_modes=None, selected=None, include_d=True):
    """Список (имя мира, DataFrame режимов, id активного D или None)."""
    canon = modes[['mode_id', 'k_c0', 'k_opex', 'k_vpub', 'k_anchor', 'k_commercial', 'public_core']].copy()
    worlds = [('ABC', canon, None)]
    if not include_d:
        return worlds
    if user_modes is None:
        user_modes = build_user_modes(modes)
    for _, um in user_modes.iterrows():
        if selected and um.mode_id not in selected:
            continue
        extra = pd.DataFrame([um[canon.columns]])
        worlds.append((um.mode_id, pd.concat([canon, extra], ignore_index=True), um.mode_id))
    return worlds


def enumerate_world(lots, modes_world, config, d_id=None, require=('BASE', 'STRESS'), admissible=None):
    """Все сочетания 4 лотов x режимы мира. В D-мирах — только портфели с D.

    admissible: функция от списка строк лотов, назначенных в D; False — портфель
    отбрасывается до расчёта (условие допустимости механизма D).
    Возвращает (feasible_rows, stats). Каждая строка — criteria_row + метки.
    """
    cache = tm._precompute(lots, modes_world)
    lidx = lots.set_index('lot_id', drop=False)
    lot_ids = list(lots['lot_id'])
    mode_ids = list(modes_world['mode_id'])
    world = d_id or 'ABC'
    if admissible is None and d_id is not None:
        admissible = ADMISSIBILITY.get(d_id)
    rows, total, inadmissible = [], 0, 0
    for lots4 in itertools.combinations(lot_ids, 4):
        for modes4 in itertools.product(mode_ids, repeat=4):
            if d_id is not None and d_id not in modes4:
                continue  # уже учтено в мире ABC
            total += 1
            combo = tuple(zip(lots4, modes4))
            if admissible is not None:
                d_lots = [lidx.loc[lot] for lot, mode in combo if mode == d_id]
                if not admissible(d_lots):
                    inadmissible += 1
                    continue
            m = tm._fast_metrics(combo, cache)
            feasible = {s: tm._feasible_fast(m, config, s) for s in tm.SCENARIOS}
            if not all(feasible[s] for s in require):
                continue
            row = tm.criteria_row(m, config)
            row.update({
                'world': world,
                'lots': '+'.join(lots4),
                'modes': '+'.join(modes4),
                'selection': combo,
                'n_D': modes4.count(d_id) if d_id else 0,
                'feasible_BASE': feasible['BASE'],
                'feasible_STRESS': feasible['STRESS'],
            })
            rows.append(row)
    return rows, {'world': world, 'enumerated': total, 'inadmissible': inadmissible, 'feasible': len(rows)}


def run_search(lots, modes, config, worlds, norm='minmax', space='blocks', blocks=None):
    """Перебор по мирам -> нормировка -> вектор -> фронт Парето."""
    blocks = blocks or tm.BLOCKS
    keys = tm.block_components(blocks)
    all_rows, stats = [], []
    for name, modes_world, d_id in worlds:
        rows, st = enumerate_world(lots, modes_world, config, d_id)
        all_rows.extend(rows)
        stats.append(st)
    stats = pd.DataFrame(stats)
    if not all_rows:
        return pd.DataFrame(), pd.DataFrame(), stats, pd.DataFrame()

    cand = pd.DataFrame(all_rows)
    raw = pd.DataFrame([tm.criteria_raw(r, config, keys) for r in all_rows])
    zthr = pd.DataFrame([tm.criteria_threshold_z(r, config, keys) for r in all_rows])
    z = tm.normalize_criteria(raw, zthr, method=norm)
    V = tm.criteria_vector(z, blocks)

    if space == 'blocks':
        mask = tm.pareto_mask(V.to_numpy())
    elif space == 'full':
        mask = tm.pareto_mask(z.to_numpy())
    else:
        raise ValueError(f'Неизвестное пространство: {space}')

    for col in raw.columns:
        cand[f'raw_{col}'] = raw[col].to_numpy()
        cand[f'z_{col}'] = z[col].to_numpy()
    for col in V.columns:
        cand[col] = V[col].to_numpy()
    cand['pareto'] = mask

    # Базовая линия: фронт только среди канонических A/B/C (та же нормировка),
    # и кто из портфелей с D доминирует каждый его элемент.
    space_matrix = V.to_numpy() if space == 'blocks' else z.to_numpy()
    abc = (cand['world'] == 'ABC').to_numpy()
    cand['pareto_ABC'] = False
    cand['dominated_by'] = ''
    if abc.any():
        abc_idx = np.flatnonzero(abc)
        cand.loc[abc_idx, 'pareto_ABC'] = tm.pareto_mask(space_matrix[abc_idx])
        for i in np.flatnonzero(cand['pareto_ABC'].to_numpy() & ~mask):
            dom = np.all(space_matrix >= space_matrix[i], axis=1) & np.any(space_matrix > space_matrix[i], axis=1)
            j = int(np.flatnonzero(dom)[np.argmax(space_matrix[dom].sum(axis=1))])
            cand.loc[i, 'dominated_by'] = f'{cand.loc[j, "lots"]} {cand.loc[j, "modes"]}'

    reference = pd.DataFrame({
        'min': raw.replace([np.inf, -np.inf], np.nan).min(),
        'max': raw.replace([np.inf, -np.inf], np.nan).max(),
        'undefined': raw.isna().sum() + np.isinf(raw).sum(),
    })
    front = cand[cand['pareto']].sort_values(list(blocks), ascending=False).reset_index(drop=True)
    return cand, front, stats, reference


# ---------------------------------------------------------------------------
# Сверка быстрого пути с каноническим слоем (Т1)
# ---------------------------------------------------------------------------

def verify_against_canon(lots, worlds, config, cand, sample=200, seed=0):
    """Случайная выборка кандидатов: метрики и вердикты совпадают с case_core."""
    rng = random.Random(seed)
    world_modes = {name: mw for name, mw, _ in worlds}
    idx = list(cand.index)
    rng.shuffle(idx)
    checked, max_err = 0, 0.0
    for i in idx[:sample]:
        row = cand.loc[i]
        _, m = evaluate_portfolio(list(row['selection']), lots, world_modes[row['world']], config)
        for key in ('c0_mrub', 'opex_mrub_per_year', 'vpub_mrub_per_year', 'cash_mrub_per_year',
                    'kcash', 't_rep', 'readiness_1_5', 'resilience_1_5', 'scale_1_5'):
            max_err = max(max_err, abs(float(m[key]) - float(row[key])))
        for key in ('territorial_archetypes', 'capability_groups', 'public_core_lots', 'selected_lots'):
            if int(m[key]) != int(row[key]):
                raise AssertionError(f'{row["lots"]} {row["modes"]}: {key} {m[key]} != {row[key]}')
        for s in tm.SCENARIOS:
            ok = bool(check_constraints(m, config, s)['ok'].all())
            if ok != bool(row[f'feasible_{s}']):
                raise AssertionError(f'{row["lots"]} {row["modes"]}: feasibility {s} mismatch')
        checked += 1
    if max_err > 1e-9:
        raise AssertionError(f'Расхождение метрик с каноном: {max_err}')
    return checked, max_err


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------

REPORT_COLS = [
    'world', 'lots', 'modes', 'F_fin', 'F_public', 'F_quality',
    'npv_mrub', 'profitability_index', 'kcash', 'vpub_mrub_per_year', 'sroi', 'public_core_lots',
    't_rep', 'readiness_1_5', 'resilience_1_5', 'scale_1_5', 'c0_mrub', 'opex_mrub_per_year',
]


def print_report(cand, front, stats, reference, args):
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print('=== Миры перебора ===')
    print(stats.to_string(index=False))
    print(f'\nВсего перебрано: {int(stats.enumerated.sum())}, допустимых в BASE и STRESS: {len(cand)}')
    if cand.empty:
        return
    print('\n=== Допустимые по мирам ===')
    print(cand.groupby('world').size().rename('feasible').to_string())

    print(f'\n=== Эталон нормировки ({args.norm}): min/max сырых критериев по допустимым ===')
    print(reference.round(4).to_string())

    print('\n=== Состав итогового вектора ===')
    for name, keys in tm.BLOCKS.items():
        print(f'  {name:10s} = сумма z({", ".join(keys)})')

    print(f'\n=== Фронт Парето в пространстве "{args.space}": {len(front)} максимальных элементов из {len(cand)} ===')
    cols = [c for c in REPORT_COLS if c in front.columns]
    print(front[cols].round(3).to_string(index=False))

    print('\n=== Использование режимов на фронте ===')
    print(front.groupby('world').size().rename('on_front').to_string())

    base = cand[cand['pareto_ABC']].sort_values(list(tm.BLOCKS), ascending=False)
    if not base.empty and (cand['world'] != 'ABC').any():
        survived = int(base['pareto'].sum())
        print(f'\n=== Базовая линия: фронт только по A/B/C — {len(base)} элементов, '
              f'в общем фронте с D остаются {survived} ===')
        base_cols = ['lots', 'modes', 'F_fin', 'F_public', 'F_quality', 'pareto', 'dominated_by']
        print(base[base_cols].round(3).to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default='.', help='каталог с data/ и config/')
    parser.add_argument('--norm', choices=('minmax', 'threshold'), default='minmax')
    parser.add_argument('--space', choices=('blocks', 'full'), default='blocks',
                        help='blocks: Парето по (F_fin, F_public, F_quality); full: по всем компонентам')
    parser.add_argument('--worlds', default=None, help='список D через запятую, например D1,D2')
    parser.add_argument('--no-d', action='store_true', help='только канонические A/B/C')
    parser.add_argument('--verify', action='store_true', help='сверить выборку с case_core')
    parser.add_argument('--out', default='output', help='каталог для CSV (относительно --root)')
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    lots, modes, config = load_case(args.root)
    selected = set(args.worlds.split(',')) if args.worlds else None
    worlds = build_worlds(modes, selected=selected, include_d=not args.no_d)

    cand, front, stats, reference = run_search(lots, modes, config, worlds, norm=args.norm, space=args.space)
    print_report(cand, front, stats, reference, args)

    if args.verify and not cand.empty:
        checked, err = verify_against_canon(lots, worlds, config, cand)
        print(f'\n=== Сверка с case_core: {checked} кандидатов, max |ошибка| = {err:.2e}, вердикты совпали ===')

    out = Path(args.root) / args.out
    out.mkdir(parents=True, exist_ok=True)
    suffix = f'{args.norm}_{args.space}'
    export_cols = [c for c in cand.columns if c not in ('selection', 'capability_set')]
    cand[export_cols].to_csv(out / f'candidates_{suffix}.csv', index=False)
    front[export_cols].to_csv(out / f'pareto_front_{suffix}.csv', index=False)
    build_user_modes(modes).to_csv(out / 'user_modes.csv', index=False)
    print(f'\nСохранено: {out / f"candidates_{suffix}.csv"}, {out / f"pareto_front_{suffix}.csv"}')
    return cand, front


if __name__ == '__main__':
    main()
