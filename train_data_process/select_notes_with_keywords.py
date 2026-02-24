import re
import csv
import pandas as pd

MAX_NOTES = 8000

keywords = ['Pijn', 'Pijnscore', 'VAS', 'Gevoelig', 'Pijnlijk',
            'Slaap', 'Geslapen', 'Ingeslapen', 'Doorgeslapen', 'Wakker', 'Uitgerust', 'Diep', 'Oppervlakkig', 'Powernap',
            'Broer', 'Broers', 'Zuster', 'Zusters', 'Moeder', 'Vader', 'Dochter', 'Zoon', 'Kleinkind', 'Kleinkinderen', 'Pleegkind', 'Stiefkind', 'Stiefmoeder', 'Stiefbroer', 'Stiefzus', 'Schoonzus',
            'Zwager', 'Schoonmoeder', 'Schoonvader', 'Oom', 'Tante', 'Nicht', 'Neef', 'Bonuskind', 'Bonusvader', 'Bonusmoeder',
            'Vergeten', 'Herkennen', 'Herinneren', 'Verward', 'Verwarder', 'Helder', 'Cognitief', 'Scherp', 'Rekenen', 'Terughalen',
            'Rollator', 'Infuuspaal', 'Mobilyzer', 'Stok', 'Rek', 'Rekje', 'Eifeltje', 'Steun', 'Armsteun', 'Handgreep', 'Kruk', 'Krukken', 'Rolstoel', 'Step', 'Vierpoot', 'Looprek', 'Handvat',
            'Gaan Liggen', 'Verplaatsen', 'Draaien', 'Rollen', 'Opstaan', 'Zitten', 'Rechtop', 'Transfer', 'Hogerop', 'Bed-stoel', 'Buigen', 'Omhoogkomen', 'Gaan staan', 'Knielen', 'Gaan zitten',
            'Roteren', 'Veranderen', 'Verandereno',
            'Horen', 'Doof', 'Slechthorend', 'Verstaan', 'Begrijpen',
            'Ontspannen', 'Gestressed', 'Verdrieting', 'Blij', 'Opgewonden', 'Gespannen', 'Stressbestendig', 'Opgejaagd', 'Controle',
            'Inspanningsvermogen', 'Uithoudingsvermogen', 'Uitgeput', 'Inspanning', 'Sporten', 'Gezeten', 'Gewandeld', 'Gefietst', 'Gelopen', 'Moe', 'Kapot', 'Vermoeid'
            ]

pattern = re.compile(r'\b(?:' + '|'.join(re.escape(w) for w in keywords) + r')\b', flags=re.IGNORECASE)

out_f = open('newcats_notities_2023.csv', 'w', encoding='utf-8', newline='')
writer = csv.writer(out_f, delimiter=';')

written = 0
stop_now = False

CHUNKSIZE = 2000
NOTE_COL = 8

for chunk in pd.read_csv(
    'VUMC notities jan-jun 2023.csv',
    sep=';',
    header=None,
    engine = 'python',
    quoting = csv.QUOTE_NONE,
    encoding = 'utf-8',
    on_bad_lines = 'warn',
    chunksize = CHUNKSIZE
):

    mask = chunk[NOTECOL].str.contains(pattern, na=False)
    for row in chunk[mask].itertuples(index=False, name=None):
        writer.writerow(row)
        written += 1
        if written >= MAX_NOTES:
            stop_now = True
            break
    if stop_now:
        break

out_f.close()
