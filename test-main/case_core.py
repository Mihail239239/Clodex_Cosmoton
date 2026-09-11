"""
Do not silently change source inputs. Teams may build any decision model on top.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


def load_case(root='.'):
    root = Path(root)
    lots = pd.read_csv(root/'data'/'lots.csv')
    modes = pd.read_csv(root/'data'/'access_modes.csv')
    with open(root/'config'/'case_config.json', encoding='utf-8') as f:
        config = json.load(f)
    return lots, modes, config


def normalize_capability(token: str) -> set[str]:
    token = token.strip()
    if token in {'PNT','PNT/InSAR','InSAR'}:
        return {'PNT/InSAR'}
    if token == 'EO': return {'EO'}
    if token == 'SATCOM': return {'SATCOM'}
    if token == 'SSA': return {'SSA'}
    return {token} if token else set()


def apply_mode(lot, mode):
    c0 = float(lot.c0_mrub) * float(mode.k_c0)
    opex = float(lot.opex_mrub_per_year) * float(mode.k_opex)
    vpub = float(lot.vpub_mrub_per_year) * float(mode.k_vpub)
    cash = float(lot.anchor_cash_mrub_per_year) * float(mode.k_anchor) + float(lot.commercial_cash_mrub_per_year) * float(mode.k_commercial)
    return {'lot_id': lot.lot_id, 'mode_id': mode.mode_id, 'c0_mrub': c0,
            'opex_mrub_per_year': opex, 'vpub_mrub_per_year': vpub,
            'cash_mrub_per_year': cash, 't_rep': float(lot.t_rep),
            'readiness_1_5': float(lot.readiness_1_5),
            'resilience_1_5': float(lot.resilience_1_5), 'scale_1_5': float(lot.scale_1_5),
            'territorial_archetype': lot.territorial_archetype,
            'federal': bool(lot.federal), 'capability_groups': lot.capability_groups,
            'public_core': bool(mode.public_core)}


def evaluate_portfolio(selection, lots, modes, config):
    """selection: list of (lot_id, mode_id), exactly 4 for a complete submission."""
    lidx = lots.set_index('lot_id', drop=False)
    midx = modes.set_index('mode_id', drop=False)
    rows=[]
    for lot_id, mode_id in selection:
        if lot_id not in lidx.index: raise ValueError(f'Unknown lot: {lot_id}')
        if mode_id not in midx.index: raise ValueError(f'Unknown mode: {mode_id}')
        rows.append(apply_mode(lidx.loc[lot_id], midx.loc[mode_id]))
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, {'selected_lots':0}
    capset=set()
    for s in detail.capability_groups:
        for token in str(s).split(';'):
            capset |= normalize_capability(token)
    territorial = set(detail.loc[~detail.federal, 'territorial_archetype'])
    opex = detail.opex_mrub_per_year.sum()
    cash = detail.cash_mrub_per_year.sum()
    metrics = {
        'selected_lots': int(detail.lot_id.nunique()),
        'c0_mrub': float(detail.c0_mrub.sum()),
        'opex_mrub_per_year': float(opex),
        'vpub_mrub_per_year': float(detail.vpub_mrub_per_year.sum()),
        'cash_mrub_per_year': float(cash),
        'kcash': float(cash/opex) if opex else float('nan'),
        't_rep': float(detail.t_rep.mean()),
        'readiness_1_5': float(detail.readiness_1_5.mean()),
        'resilience_1_5': float(detail.resilience_1_5.mean()),
        'scale_1_5': float(detail.scale_1_5.mean()),
        'territorial_archetypes': len(territorial),
        'capability_groups': len(capset),
        'capability_set': sorted(capset),
        'public_core_lots': int(detail.public_core.sum()),
    }
    return detail, metrics


def check_constraints(metrics, config, scenario='BASE'):
    c = config['constraints_common']; sc=config['scenarios'][scenario]
    checks = {
      'exact_lot_count': metrics.get('selected_lots') == c['selected_lots_exactly'],
      'territorial_archetypes': metrics.get('territorial_archetypes',0) >= c['min_territorial_archetypes'],
      'capability_groups': metrics.get('capability_groups',0) >= c['min_capability_groups'],
      'public_core_lots': metrics.get('public_core_lots',0) >= c['min_public_core_lots'],
      'c0_limit': metrics.get('c0_mrub', float('inf')) <= sc['c0_max_mrub'] + 1e-9,
      'opex_limit': metrics.get('opex_mrub_per_year', float('inf')) <= c['opex_max_mrub_per_year'] + 1e-9,
      'vpub_floor': metrics.get('vpub_mrub_per_year', -float('inf')) >= c['vpub_min_mrub_per_year'] - 1e-9,
      'kcash_floor': metrics.get('kcash', -float('inf')) >= c['kcash_min'] - 1e-9,
      't_rep_floor': metrics.get('t_rep', -float('inf')) >= c['t_rep_min'] - 1e-9,
    }
    return pd.DataFrame([{'constraint':k,'ok':bool(v)} for k,v in checks.items()])
