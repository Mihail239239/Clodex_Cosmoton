# Запуск и воспроизводимость

Всё работает офлайн, без внешних API, ключей и регистрации: только Python 3.10+ и
`pandas`, `numpy` (протестировано на Python 3.14.2, pandas 3.0.5, numpy 2.3.5;
код не использует специфичных для версии возможностей и должен работать на
Python 3.9+/pandas 1.5+/numpy 1.23+).

```bash
cd test-main
pip install pandas numpy   # если их ещё нет
python tests.py            # 12 контрольных примеров, Т1/Т3 — должно быть 12/12
python team_model.py       # демонстрационный прогон, перебор + оптимум + чувствительность
```

## Файлы

| Файл | Назначение |
|---|---|
| `case_core.py` | канонический слой организаторов — **не изменён** |
| `team_model.py` | надстройка команды: финансы, нормировка, блоки, фронт Парето (раздел 9) |
| `user_modes.py` | пользовательские режимы D1/D3: вывод коэффициентов, допустимость, безубыточность |
| `evaluate.py` | **CLI для одного портфеля** — смена состава/режимов/весов/параметров без правки кода |
| `pareto_search.py` | полный перебор всех режимов, отсечение по ограничениям, фронт Парето |
| `pareto_report.py` / `pareto_html.py` | генерация `../PARETO_FRONT.md` и `output/pareto_front.html` |
| `tests.py` | контрольные примеры (Т1, Т3) |
| `data/`, `config/` | входные данные и ограничения кейса — не изменены |

## Т2: смена портфеля и параметров без правки кода

```bash
# рекомендуемый портфель записки — состав, финансы, ограничения, оба сценария
python evaluate.py FIRE:D3 AGRI:C TRANS:C ENV:D3

# любой другой состав / режимы — просто другие аргументы командной строки
python evaluate.py FLOOD:A AGRI:C TRANS:B ENV:A

# заведомо недопустимый портфель — Т3 показывает конкретное нарушение
python evaluate.py ARCTIC:A SSA:A FIRE:A FLOOD:A --constraints-only

# свои веса восьми критериев (доли, нормируются автоматически)
python evaluate.py FIRE:D3 AGRI:C TRANS:C ENV:D3 \
  --weights vpub=0.3,capex=0.1,opex=0.1,kcash=0.1,t_rep=0.1,readiness=0.1,resilience=0.1,scale=0.1

# своя ставка дисконтирования и горизонт (по умолчанию 0.10 и 10 лет)
python evaluate.py FIRE:D3 AGRI:C TRANS:C ENV:D3 --discount-rate 0.07 --horizon 15
```

## Т3: проверка ограничений

`evaluate.py` печатает для каждого из девяти ограничений колонку
порог/факт/запас/PASS-FAIL, отдельно для BASE и STRESS (`team_model.constraint_report`).
Пример нарушения — `ARCTIC:A SSA:A FIRE:A FLOOD:A` выше: `c0_limit` и `opex_limit`
дают отрицательный запас и `False`, `t_rep_floor` — `False` при факте 0.6225 против
порога 0.63.

## Т4: сопоставление BASE и STRESS

`evaluate.py` без `--constraints-only` печатает блок «Сравнение сценариев» —
загрузку лимитов в процентах и запас по c0 для BASE и STRESS рядом. Полное
сопоставление всех допустимых вариантов в обоих сценариях сразу —
`pareto_search.py` / `../PARETO_FRONT.md` (раздел 4, столбец «запас c0» уже в
STRESS-метрике).

## Полный перебор и фронт Парето

```bash
python pareto_search.py --verify   # перебор + сверка быстрого пути с case_core
python pareto_report.py            # ../PARETO_FRONT.md + output/pareto_front.html + CSV
```

Занимает секунды: пространство перебора конечно (C(8,4) × режимы³/⁴), полный
перебор без эвристик.
