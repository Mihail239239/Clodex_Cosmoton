"""
evaluate.py — интерактивный CLI для одного портфеля: смена состава, режимов и
параметров без правки кода (Т2), проверка ограничений с порогом/фактом/запасом
(Т3), сопоставление BASE и STRESS в одном выводе (Т4).

Использует team_model.evaluate_full поверх нетронутого case_core.py. Режимы
A/B/C — из data/access_modes.csv; D1/D3, если запрошены, добавляются из
user_modes.py (коэффициенты выводятся формулами из A/B, см. её docstring).

Примеры:
  # рекомендуемый портфель записки, оба сценария, ограничения, финансы
  python evaluate.py FIRE:D3 AGRI:C TRANS:C ENV:D3

  # тот же состав в чистом A — показать эффект гипотезы D3 (откат)
  python evaluate.py FIRE:A AGRI:C TRANS:C ENV:A

  # свои веса критериев (восемь слотов, доли не обязаны суммироваться в 1 —
  # нормируются автоматически) и свои финансовые допущения
  python evaluate.py FIRE:D3 AGRI:C TRANS:C ENV:D3 \\
      --weights vpub=0.3,capex=0.1,opex=0.1,kcash=0.1,t_rep=0.1,readiness=0.1,resilience=0.1,scale=0.1 \\
      --discount-rate 0.08 --horizon 15

  # заведомо недопустимый портфель — Т3 показывает конкретное нарушение
  python evaluate.py ARCTIC:A SSA:A FIRE:A FLOOD:A

  # только сводка ограничений в виде таблицы порог/факт/запас/PASS
  python evaluate.py FIRE:D3 AGRI:C TRANS:C ENV:D3 --constraints-only

Полный автоматический перебор и поиск оптимума/фронта — pareto_search.py и
pareto_report.py; этот скрипт — для точечной проверки одного варианта.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from case_core import load_case
import team_model as tm
import user_modes as um

pd.set_option('display.width', 200)


def build_mode_table(modes, enable_d=('D1', 'D3')):
    """Канонические A/B/C + запрошенные пользовательские режимы."""
    out = modes[['mode_id', 'k_c0', 'k_opex', 'k_vpub', 'k_anchor', 'k_commercial', 'public_core']].copy()
    builders = {'D1': um.d1_coefficients, 'D3': um.d3_coefficients}
    for d in enable_d:
        if d in builders:
            out = pd.concat([out, pd.DataFrame([builders[d](modes)])[out.columns]], ignore_index=True)
    return out


def parse_selection(items):
    sel = []
    for item in items:
        if ':' not in item:
            raise SystemExit(f'Ожидается LOT:MODE, получено: {item!r}')
        lot, mode = item.split(':', 1)
        sel.append((lot.strip().upper(), mode.strip().upper()))
    return sel


def parse_weights(spec):
    if not spec:
        return None
    w = {}
    for part in spec.split(','):
        k, v = part.split('=')
        w[k.strip()] = float(v)
    unknown = set(w) - set(tm.CRITERIA)
    if unknown:
        raise SystemExit(f'Неизвестные критерии весов: {sorted(unknown)}. Допустимы: {tm.CRITERIA}')
    return w


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('selection', nargs='+', help='LOT:MODE, ровно 4 (например FIRE:D3 AGRI:C TRANS:C ENV:D3)')
    parser.add_argument('--root', default='.')
    parser.add_argument('--weights', default=None, help='crit=val,crit=val,... из {%s}' % ','.join(tm.CRITERIA))
    parser.add_argument('--discount-rate', type=float, default=None, help='ставка дисконтирования, доля в год (по умолчанию 0.10)')
    parser.add_argument('--horizon', type=int, default=None, help='горизонт расчёта, лет (по умолчанию 10)')
    parser.add_argument('--enable-d', default='D1,D3', help='какие пользовательские режимы доступны, через запятую; пусто — только A/B/C')
    parser.add_argument('--constraints-only', action='store_true', help='вывести только таблицы ограничений BASE/STRESS')
    parser.add_argument('--csv', default=None, help='сохранить плоский результат (flatten_result) в CSV')
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    lots, modes, config = load_case(args.root)
    enable_d = [d for d in args.enable_d.split(',') if d] if args.enable_d else []
    mode_table = build_mode_table(modes, enable_d)

    selection = parse_selection(args.selection)
    if len(selection) != config['constraints_common']['selected_lots_exactly']:
        raise SystemExit(f'Нужно ровно {config["constraints_common"]["selected_lots_exactly"]} лота, получено {len(selection)}')
    unknown_lots = set(l for l, _ in selection) - set(lots['lot_id'])
    unknown_modes = set(m for _, m in selection) - set(mode_table['mode_id'])
    if unknown_lots:
        raise SystemExit(f'Неизвестные лоты: {sorted(unknown_lots)}. Доступны: {sorted(lots["lot_id"])}')
    if unknown_modes:
        raise SystemExit(f'Неизвестные режимы: {sorted(unknown_modes)}. Доступны: {sorted(mode_table["mode_id"])} '
                         f'(добавьте нужный D через --enable-d)')

    weights = parse_weights(args.weights)
    assumptions = {}
    if args.discount_rate is not None:
        assumptions['discount_rate'] = args.discount_rate
    if args.horizon is not None:
        assumptions['horizon_years'] = args.horizon

    result = tm.evaluate_full(selection, lots, mode_table, config, weights=weights, assumptions=assumptions)

    print(f'=== Портфель: {" + ".join(f"{l}:{m}" for l, m in selection)} ===\n')

    if not args.constraints_only:
        print('--- Состав (после apply_mode) ---')
        cols = ['lot_id', 'mode_id', 'c0_mrub', 'opex_mrub_per_year', 'vpub_mrub_per_year',
                'cash_mrub_per_year', 'public_core', 'territorial_archetype']
        print(result['detail'][cols].round(2).to_string(index=False))

        print('\n--- Метрики портфеля ---')
        for k, v in result['metrics'].items():
            print(f'  {k:26s} {v}')

        print('\n--- Финансовые метрики (r=%.3f, T=%d лет) ---' % (
            result['assumptions']['discount_rate'], result['assumptions']['horizon_years']))
        for k, v in result['financial'].items():
            print(f'  {k:26s} {v}')

        print('\n--- Производные показатели ---')
        for k, v in result['extras'].items():
            print(f'  {k:26s} {v}')

    print('\n--- Ограничения: BASE (c0 <= %.0f) ---' % config['scenarios']['BASE']['c0_max_mrub'])
    print(result['constraints']['BASE'].to_string(index=False))
    base_ok = bool(result['constraints']['BASE']['ok'].all())
    print(f'  => {"PASS — портфель допустим в BASE" if base_ok else "FAIL — портфель НЕ допустим в BASE"}')

    print('\n--- Ограничения: STRESS (c0 <= %.0f) ---' % config['scenarios']['STRESS']['c0_max_mrub'])
    print(result['constraints']['STRESS'].to_string(index=False))
    stress_ok = bool(result['constraints']['STRESS']['ok'].all())
    print(f'  => {"PASS — портфель допустим в STRESS" if stress_ok else "FAIL — портфель НЕ допустим в STRESS"}')

    if not args.constraints_only:
        print('\n--- Сравнение сценариев ---')
        for scenario in tm.SCENARIOS:
            sc = result['scenarios'][scenario]
            print(f'  {scenario}: c0={sc["c0_utilization_pct"]:.1f}% лимита, opex={sc["opex_utilization_pct"]:.1f}% лимита, '
                 f'запас c0={sc["headroom_c0_mrub"]:.1f} млн, feasible={sc["feasible"]}')
        print('\n--- Композитный score (веса: %s) ---' % (weights or 'равные 1/8'))
        for k, v in result['score'].items():
            print(f'  {k:16s} {v:.4f}' if isinstance(v, float) else f'  {k:16s} {v}')

    if args.csv:
        flat = tm.flatten_result(result)
        pd.DataFrame([flat]).to_csv(args.csv, index=False)
        print(f'\nСохранено: {args.csv}')


if __name__ == '__main__':
    main()
