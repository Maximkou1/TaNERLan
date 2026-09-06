# Аудит конвенций границ

Документов: 13000 | сущностей: 66083
Скрипт документов: {'cyrillic': 2207, 'latin': 7840, 'mixed': 2952, 'other': 1}
Классы: {'GEO': 21445, 'ORG': 23420, 'NAME': 21218}

## Ошибки данных (это баги, не конвенции)

- `err_out_of_range`: **0**   0.0%
- `err_start_ge_end`: **0**   0.0%
- `err_leading_ws`: **0**   0.0%
- `err_trailing_ws`: **0**   0.0%
- `err_overlap`: **0**   0.0%
- `err_nested`: **0**   0.0%
- `err_dup`: **0**   0.0%

## Конвенции разметки (это модель должна выучить)

| Признак | ORG | NAME | GEO | всего | доля |
|---|---|---|---|---|---|
| suffix_included | 7369 | 3968 | 9400 | 20737 |  31.4% |
| quote_included | 511 | 40 | 249 | 800 |   1.2% |
| orgform_included | 48 | 0 | 0 | 48 |   0.1% |
| geotail_included | 909 | 0 | 2742 | 3651 |   5.5% |
| orgtail_included | 2483 | 1 | 112 | 2596 |   3.9% |
| title_included | 58 | 0 | 0 | 58 |   0.1% |
| trail_punct | 16 | 49 | 14 | 79 |   0.1% |
| lead_punct | 0 | 0 | 0 | 0 |   0.0% |
| span_starts_midword | 0 | 0 | 0 | 0 |   0.0% |
| span_ends_midword | 0 | 0 | 0 | 0 |   0.0% |

## Топ аффиксов внутри спанов

- `-i` : 3349
- `-da` : 2379
- `-si` : 1830
- `-ning` : 1745
- `-и` : 1310
- `-ga` : 1152
- `-да` : 971
- `-ni` : 951
- `-dagi` : 783
- `-нинг` : 636
- `-си` : 635
- `-га` : 552
- `-ining` : 474
- `-ни` : 418
- `-dan` : 394
- `-даги` : 335
- `-iga` : 279
- `-u` : 272
- `-li` : 233
- `-ининг` : 210

## Варианты апострофа внутри спанов

- U+02BB MODIFIER LETTER TURNED COMMA : 4100
- U+0027 APOSTROPHE : 567
- U+2018 LEFT SINGLE QUOTATION MARK : 522
- U+2019 RIGHT SINGLE QUOTATION MARK : 71
- U+02BC MODIFIER LETTER APOSTROPHE : 15
- U+0060 GRAVE ACCENT : 9

## Неоднозначные поверхностные формы (169)

- `Oʻzbekiston` -> {'GEO': 640, 'ORG': 7}
- `Eron` -> {'GEO': 277, 'ORG': 2}
- `Murad Buildings` -> {'ORG': 253, 'GEO': 3}
- `Ўзбекистон` -> {'GEO': 230, 'ORG': 2}
- `Toshkent` -> {'GEO': 228, 'ORG': 1}
- `Исроил` -> {'GEO': 140, 'ORG': 1}
- `Andijon` -> {'ORG': 20, 'GEO': 39}
- `Samarqand` -> {'GEO': 55, 'ORG': 1}
- `Мурод билдинг` -> {'ORG': 46, 'GEO': 2}
- `Paxtakor` -> {'ORG': 41, 'GEO': 5}
- `Qatar` -> {'GEO': 44, 'ORG': 1}
- `Bunyodkor` -> {'ORG': 24, 'GEO': 16}
- `Buxoro` -> {'ORG': 8, 'GEO': 30}
- `Андижон` -> {'ORG': 18, 'GEO': 15}
- `Yaponiya` -> {'ORG': 4, 'GEO': 29}
- `Belarus` -> {'GEO': 30, 'ORG': 1}
- `Madina` -> {'NAME': 27, 'GEO': 3}
- `Iroq` -> {'GEO': 29, 'ORG': 1}
- `Бухоро` -> {'ORG': 20, 'GEO': 10}
- `Бунёдкор` -> {'ORG': 13, 'GEO': 15}
- `Navbahor` -> {'NAME': 1, 'ORG': 23, 'GEO': 1}
- `Хоразм` -> {'GEO': 12, 'ORG': 11}
- `Пахтакор` -> {'ORG': 22, 'GEO': 1}
- `Oqtepa` -> {'GEO': 12, 'ORG': 11}
- `Murod building` -> {'GEO': 1, 'ORG': 20}
- `Avstraliya` -> {'GEO': 18, 'ORG': 2}
- `Nest One` -> {'GEO': 17, 'ORG': 3}
- `Janubiy Koreya` -> {'GEO': 17, 'ORG': 2}
- `Milan` -> {'ORG': 17, 'GEO': 2}
- `Barcelona` -> {'ORG': 16, 'GEO': 2}