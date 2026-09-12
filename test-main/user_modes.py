"""
user_modes.py — пользовательские режимы доступа D с формальным выводом коэффициентов.

Каждый режим D задаётся не пятью числами, а набором именованных параметров
механизма и формулами, по которым пять коэффициентов выводятся из канонических
режимов A/B/C (data/access_modes.csv). Числа в таблице — результат этих формул.
Так параметры можно менять и пересчитывать, а не подбирать.

Обозначения (для одного лота, исходные значения из data/lots.csv):
    C  = c0_mrub                      разовые затраты
    O  = opex_mrub_per_year           годовые затраты
    V  = vpub_mrub_per_year           общественная ценность в год
    P  = anchor_cash_mrub_per_year    якорные поступления в год
    Q  = commercial_cash_mrub_per_year коммерческие поступления в год
    af = annuity_factor(r, T)         сумма дисконтов, r = 0.10, T = 10 -> 6.1446

Канонические режимы (k_c0, k_opex, k_vpub, k_anchor, k_commercial, public_core):
    A = (1.05, 1.05, 1.00, 1.00, 0.25, true)
    B = (1.00, 1.00, 0.82, 0.80, 0.70, false)
    C = (0.98, 0.95, 0.62, 0.45, 0.95, false)

Принцип отбора. Оставлены только режимы, у которых:
  (а) общественный слой A сохраняется физически, поэтому public_core, k_vpub и
      k_anchor наследуются от A, а не назначаются;
  (б) каждый изменённый коэффициент имеет один названный механизм и формулу;
  (в) есть формальное условие допустимости, проверяемое по данным лота/портфеля;
  (г) есть точка безубыточности, которую можно посчитать по данным кейса.

================================================================================
D1. Общественное ядро A + платная надстройка для бизнеса
================================================================================
Механизм. Сервис поставляется ровно как в A: тот же бесплатный общественный слой,
тот же якорный договор, та же общественная ценность. Дополнительно строится и
эксплуатируется коммерческая надстройка (интеграция в ИС заказчика, индивидуальные
отчёты, обработка дополнительных запросов). Её затраты входят в C0 и OPEX, её
выручка — только в коммерческие поступления.

Параметры:
    delta_c0   = 0.02  разработка надстройки, доля исходного C0
    delta_opex = 0.03  сопровождение, продажи, биллинг, доля исходного OPEX
    theta      = 0.20  доля разрыва коммерческих коэффициентов B - A, которую
                       надстройка забирает без ослабления общественного доступа

Формулы:
    k_c0         = k_c0^A   + delta_c0                    = 1.05 + 0.02 = 1.07
    k_opex       = k_opex^A + delta_opex                  = 1.05 + 0.03 = 1.08
    k_vpub       = k_vpub^A                               = 1.00   (слой A не тронут)
    k_anchor     = k_anchor^A                             = 1.00   (договор A не тронут)
    k_commercial = k_com^A + theta * (k_com^B - k_com^A)  = 0.25 + 0.20 * 0.45 = 0.34
    public_core  = public_core^A                          = true   (наследуется)

Экономика относительно A для одного лота:
    dCASH  = theta * (k_com^B - k_com^A) * Q = 0.09 * Q
    dOPEX  = delta_opex * O                  = 0.03 * O
    dC0    = delta_c0 * C                    = 0.02 * C
    dNPV   = -dC0 + af * (dCASH - dOPEX)
Годовой операционный эффект положителен при Q > O * delta_opex / 0.09 = O / 3.
Безубыточная доля захвата (dNPV = 0):
    theta* = (delta_c0 * C / af + delta_opex * O) / ((k_com^B - k_com^A) * Q)
Лот, у которого theta* > theta, в D1 хуже, чем в A: надстройка не окупается.
Проверка по данным: d1_breakeven_table().

Допустимость: любой лот с Q > 0 (у всех восьми лотов Q > 0). Надстройка не
трогает базовый слой, поэтому в одном портфеле D1 может стоять у любого числа
лотов независимо друг от друга.

Чувствительность (для записки): theta in {0, 0.10, 0.20}; delta_c0 in [0.02, 0.04];
delta_opex in [0.03, 0.06]. При theta = 0 D1 строго хуже A на всех лотах —
это цена гипотезы спроса, и она видна в расчёте.

================================================================================
D3. Повторно используемое ядро и совместная закупка (на базе A)
================================================================================
Механизм. Общественный пакет остаётся на уровне A. Часть работ, которая при
раздельной закупке повторяется у каждого лота (типовой интерфейс, каталог данных,
общие лицензии, эксплуатация общего компонента), выполняется один раз для группы
лотов с общей технологической цепочкой. Локальные работы (региональные пороги,
карты, обучение служб) остаются в смете каждого лота. Координация группы стоит
денег и добавляется явно.

Параметры:
    sigma_c0   = 0.20  доля стартовых работ, затронутая повторным использованием
    rho_c0     = 0.30  экономия на этой доле (строится один раз, а не n раз)
    gamma_c0   = 0.02  координация и адаптация, доля исходного C0
    sigma_opex = 0.25  доля эксплуатации, затронутая общим компонентом
    rho_opex   = 0.20  экономия на этой доле
    gamma_opex = 0.02  координация, доля исходного OPEX

Формулы:
    k_c0         = k_c0^A   * (1 - sigma_c0 * rho_c0)     + gamma_c0
                 = 1.05 * (1 - 0.06) + 0.02 = 1.007
    k_opex       = k_opex^A * (1 - sigma_opex * rho_opex) + gamma_opex
                 = 1.05 * (1 - 0.05) + 0.02 = 1.0175
    k_vpub       = k_vpub^A   = 1.00
    k_anchor     = k_anchor^A = 1.00
    k_commercial = k_com^A    = 0.25
    public_core  = public_core^A = true

Экономия положительна тогда и только тогда, когда sigma * rho > gamma / k^A:
    C0:   0.20 * 0.30 = 0.060 > 0.02 / 1.05 = 0.0190   (запас 3.2x)
    OPEX: 0.25 * 0.20 = 0.050 > 0.0190                 (запас 2.6x)
Точка нулевой экономии по rho при фиксированной sigma:
    rho_c0*   = gamma_c0   / (k_c0^A   * sigma_c0)   = 0.0952
    rho_opex* = gamma_opex / (k_opex^A * sigma_opex) = 0.0762
Выигрыш относительно A для одного лота:
    dC0   = C * (k_c0^A * sigma_c0 * rho_c0 - gamma_c0)         = 0.0430 * C
    dOPEX = O * (k_opex^A * sigma_opex * rho_opex - gamma_opex) = 0.0325 * O
    dNPV  = dC0 + af * dOPEX  (> 0 всегда при принятых параметрах)

Допустимость (формально, проверяется в переборе):
    (1) в портфеле не меньше двух лотов в D3 — общий компонент с одним
        участником не существует, экономия не возникает;
    (2) у всех лотов в D3 есть хотя бы одна общая группа возможностей
        (capability_groups после normalize_capability) — иначе нет общей
        технологической цепочки, которую можно построить один раз.
Экономия принята постоянной на лот при n >= 2 (консервативно: при большем n
доля каждого участника в общем компоненте меньше, но это не заявляется).
Один и тот же общий компонент считается один раз: формула задаёт долю каждого
лота в общем компоненте (70% своей повторно используемой части + координация),
а не «полную экономию каждому».

Чувствительность: rho_c0 in {0, 0.15, 0.30}; rho_opex in {0, 0.10, 0.20};
gamma in {0.02, 0.04}. При rho = 0 D3 хуже A ровно на координацию — цена
гипотезы видна.

================================================================================
Исключённые варианты из d_type.md и причины
================================================================================
D2  «Публичный заказчик оплачивает базовый доступ» — public_core = true
    назначается на базе B, у которой по канону бесплатного слоя нет; k_vpub = 0.91
    противоречит «полному общественному результату»; k_c0 = 1.04 < k_c0^A = 1.05
    означает, что организация массового доступа поверх B дешевле встроенного
    доступа A — механизм этого не объясняет. В переборе D2 вытеснял 19 из 21
    элементов фронта A/B/C именно за счёт флага. Консервативная переформулировка
    (k_c0 >= 1.05, k_vpub = 1.00 как условие флага) совпадает с D1.
D4  «Взвешенный пакет A/B» — доля alpha = 0.5 не выводится из данных; линейная
    смесь пяти разнородных баз — допущение без сметы; на фронт не попадал.
D5  «Дополнительный якорный заказ» — два независимых «10%» с разными базами без
    связи между ними; public_core = false; на фронт не попадал.
D6  «Переносимый сервис» — по построению дороже C при тех же поступлениях и
    vpub, доминируется C лот за лотом; его ценность (сменяемость поставщика) в
    метрическом пространстве отсутствует.
D7  «Равное среднее A/B/C» — доминируется B лот за лотом (доказано в d_type.md).
"""
from __future__ import annotations

import sys

import pandas as pd

from case_core import normalize_capability
from team_model import ASSUMPTIONS, annuity_factor

MODE_COLUMNS = ['mode_id', 'k_c0', 'k_opex', 'k_vpub', 'k_anchor', 'k_commercial', 'public_core']

# ---------------------------------------------------------------------------
# Параметры механизмов. Менять здесь; коэффициенты пересчитываются формулами.
# ---------------------------------------------------------------------------

D1_PARAMS = {
    'delta_c0': 0.02,    # разработка платной надстройки, доля исходного C0
    'delta_opex': 0.03,  # сопровождение, продажи, биллинг, доля исходного OPEX
    'theta': 0.20,       # доля разрыва коммерческих коэффициентов B - A, захватываемая надстройкой
}

D3_PARAMS = {
    'sigma_c0': 0.20,    # доля стартовых работ, затронутая повторным использованием
    'rho_c0': 0.30,      # экономия на этой доле
    'gamma_c0': 0.02,    # координация и адаптация, доля исходного C0
    'sigma_opex': 0.25,  # доля эксплуатации, затронутая общим компонентом
    'rho_opex': 0.20,    # экономия на этой доле
    'gamma_opex': 0.02,  # координация, доля исходного OPEX
    'min_lots': 2,       # общий компонент существует минимум для двух лотов
}

LABELS = {
    'D1': 'Общественное ядро A + платная надстройка для бизнеса',
    'D3': 'Повторно используемое ядро и совместная закупка (на базе A)',
}


# ---------------------------------------------------------------------------
# Вывод коэффициентов из канонических режимов
# ---------------------------------------------------------------------------

def _canon(modes):
    return modes.set_index('mode_id', drop=False)


def d1_coefficients(modes, p=None):
    p = {**D1_PARAMS, **(p or {})}
    m = _canon(modes)
    a, b = m.loc['A'], m.loc['B']
    return {
        'mode_id': 'D1',
        'k_c0': float(a.k_c0) + p['delta_c0'],
        'k_opex': float(a.k_opex) + p['delta_opex'],
        'k_vpub': float(a.k_vpub),
        'k_anchor': float(a.k_anchor),
        'k_commercial': float(a.k_commercial) + p['theta'] * (float(b.k_commercial) - float(a.k_commercial)),
        'public_core': bool(a.public_core),
    }


def d3_coefficients(modes, p=None):
    p = {**D3_PARAMS, **(p or {})}
    a = _canon(modes).loc['A']
    return {
        'mode_id': 'D3',
        'k_c0': float(a.k_c0) * (1.0 - p['sigma_c0'] * p['rho_c0']) + p['gamma_c0'],
        'k_opex': float(a.k_opex) * (1.0 - p['sigma_opex'] * p['rho_opex']) + p['gamma_opex'],
        'k_vpub': float(a.k_vpub),
        'k_anchor': float(a.k_anchor),
        'k_commercial': float(a.k_commercial),
        'public_core': bool(a.public_core),
    }


def build_user_modes(modes, d1=None, d3=None):
    """DataFrame пользовательских режимов в формате access_modes.csv + label."""
    rows = [d1_coefficients(modes, d1), d3_coefficients(modes, d3)]
    df = pd.DataFrame(rows, columns=MODE_COLUMNS)
    df['label'] = df['mode_id'].map(LABELS)
    return df


# ---------------------------------------------------------------------------
# Допустимость режима для набора лотов (проверяется в переборе)
# ---------------------------------------------------------------------------

def _caps(lot_row):
    caps = set()
    for token in str(lot_row.capability_groups).split(';'):
        caps |= normalize_capability(token)
    return caps


def d1_admissible(lot_rows):
    """Любой лот с коммерческой базой; надстройки лотов независимы."""
    return all(float(r.commercial_cash_mrub_per_year) > 0 for r in lot_rows)


def d3_admissible(lot_rows, p=None):
    """>= min_lots лотов и общая группа возможностей у всех лотов в D3."""
    p = {**D3_PARAMS, **(p or {})}
    rows = list(lot_rows)
    if len(rows) < p['min_lots']:
        return False
    common = set.intersection(*(_caps(r) for r in rows))
    return len(common) > 0


ADMISSIBILITY = {'D1': d1_admissible, 'D3': d3_admissible}


# ---------------------------------------------------------------------------
# Точки безубыточности по данным кейса (для записки и проверки гипотез)
# ---------------------------------------------------------------------------

def d1_breakeven_table(lots, modes, p=None, rate=None, years=None):
    """Для каждого лота: dNPV D1 относительно A и безубыточная доля захвата theta*."""
    p = {**D1_PARAMS, **(p or {})}
    r = ASSUMPTIONS['discount_rate'] if rate is None else rate
    T = ASSUMPTIONS['horizon_years'] if years is None else years
    af = annuity_factor(r, T)
    m = _canon(modes)
    gap = float(m.loc['B'].k_commercial) - float(m.loc['A'].k_commercial)
    rows = []
    for lot in lots.itertuples():
        C, O, Q = float(lot.c0_mrub), float(lot.opex_mrub_per_year), float(lot.commercial_cash_mrub_per_year)
        d_cash = p['theta'] * gap * Q
        d_opex = p['delta_opex'] * O
        d_c0 = p['delta_c0'] * C
        rows.append({
            'lot_id': lot.lot_id,
            'Q_commercial': Q,
            'dC0_mrub': d_c0,
            'dOPEX_mrub_per_year': d_opex,
            'dCASH_mrub_per_year': d_cash,
            'd_net_per_year': d_cash - d_opex,
            'dNPV_vs_A_mrub': -d_c0 + af * (d_cash - d_opex),
            'theta_breakeven': (d_c0 / af + d_opex) / (gap * Q) if Q > 0 else float('inf'),
            'pays_off_at_theta': (-d_c0 + af * (d_cash - d_opex)) > 0,
        })
    return pd.DataFrame(rows)


def d3_breakeven_table(lots, modes, p=None, rate=None, years=None):
    """Для каждого лота: экономия D3 относительно A по C0, OPEX и NPV."""
    p = {**D3_PARAMS, **(p or {})}
    r = ASSUMPTIONS['discount_rate'] if rate is None else rate
    T = ASSUMPTIONS['horizon_years'] if years is None else years
    af = annuity_factor(r, T)
    a = _canon(modes).loc['A']
    s_c0 = float(a.k_c0) * p['sigma_c0'] * p['rho_c0'] - p['gamma_c0']
    s_op = float(a.k_opex) * p['sigma_opex'] * p['rho_opex'] - p['gamma_opex']
    rows = []
    for lot in lots.itertuples():
        C, O = float(lot.c0_mrub), float(lot.opex_mrub_per_year)
        rows.append({
            'lot_id': lot.lot_id,
            'capability_groups': ';'.join(sorted(_caps(lot))),
            'dC0_saved_mrub': s_c0 * C,
            'dOPEX_saved_mrub_per_year': s_op * O,
            'dNPV_vs_A_mrub': s_c0 * C + af * s_op * O,
        })
    df = pd.DataFrame(rows)
    df.attrs['rho_c0_zero'] = p['gamma_c0'] / (float(a.k_c0) * p['sigma_c0'])
    df.attrs['rho_opex_zero'] = p['gamma_opex'] / (float(a.k_opex) * p['sigma_opex'])
    return df


def _print_tables():
    from case_core import load_case
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    pd.set_option('display.width', 200)
    lots, modes, _ = load_case('.')
    um = build_user_modes(modes)
    print('=== Пользовательские режимы (выведены из A/B формулами) ===')
    print(um.to_string(index=False))
    print('\n=== D1: безубыточность надстройки по лотам (theta = %.2f) ===' % D1_PARAMS['theta'])
    print(d1_breakeven_table(lots, modes).round(3).to_string(index=False))
    d3 = d3_breakeven_table(lots, modes)
    print('\n=== D3: экономия относительно A по лотам ===')
    print(d3.round(3).to_string(index=False))
    print(f"rho_c0* = {d3.attrs['rho_c0_zero']:.4f}, rho_opex* = {d3.attrs['rho_opex_zero']:.4f}")


if __name__ == '__main__':
    _print_tables()
