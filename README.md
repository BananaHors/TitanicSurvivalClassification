# Titanic Survival Classification

Predmetni projekat (SAUSAU). Cilj je da se za svakog putnika Titanika predvidi da
li je preživeo (`Survived` 0/1) na osnovu njegovih podataka, i da se vidi koji
faktori najviše utiču na ishod. Pored same predikcije, trudio sam se da kroz
grafike i kratko tumačenje pokažem i šta se zapravo dešava u podacima.

Sve radi nad jednim fajlom, `data/data.csv` (Titanik train.csv, 891 putnik).

Ako te zanima kako sve radi i zašto sam neke stvari uradio baš tako, to je u
[DOKUMENTACIJA.md](DOKUMENTACIJA.md).

## Šta dobijamo kad pokrenemo

Kad pokrenemo `python main.py`, sve završi u folderu `outputs/`:

- metrike modela (tačnost, preciznost, odziv, F1, ROC-AUC),
- 13 grafika u `outputs/figures/`, i EDA i grafici evaluacije,
- `outputs/metrics/izvestaj.txt` sa metrikama i tumačenjem,
- `outputs/metrics/metrics.json`, iste metrike samo mašinski čitljivo,
- `outputs/models/best_model.joblib`, sačuvan najbolji model.

## Zahtevi

Python 3.10 ili noviji (ja sam radio na 3.14) i paketi iz
[requirements.txt](requirements.txt).

## Pokretanje

Iz rootaprojekta:

```powershell
cd "c:\Projects\Titanic Survival"
```

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```powershell
pip install -r requirements.txt
python main.py
```

Skripta ispisuje napredak kroz faze i za par sekundi napuni `outputs/`.

## Organizacija

```
Titanic Survival/
├── data/
│   └── data.csv                 # ulazni podaci
├── src/
│   ├── config.py                # putanje i konstante (seed, test_size, prag...)
│   ├── data_loader.py           # učitavanje CSV i pregled praznih polja
│   ├── features.py              # izvođenje atributa (Title, FamilySize, Deck...)
│   ├── preprocessing.py         # sklearn transformeri
│   ├── model.py                 # treniranje, poređenje modela, evaluacija
│   └── visualization.py         # svi grafici
├── outputs/                     # pravi se pri pokretanju
│   ├── figures/
│   ├── metrics/
│   └── models/
├── main.py                      # pokreće ceo pipeline
├── explore.py                   # pregled podataka
├── requirements.txt
├── README.md
└── DOKUMENTACIJA.md
```

## Korišćenje sačuvanog modela

U `best_model.joblib` je ceo pipeline, i obrada i model, pa može direktno na
sirove kolone bez ikakve pripreme:

```python
import joblib, pandas as pd

model = joblib.load("outputs/models/best_model.joblib")
novi = pd.read_csv("data/data.csv").drop(columns=["Survived"])
predikcije = model.predict(novi)            # 0 = nije preživeo, 1 = preživeo
verovatnoce = model.predict_proba(novi)[:, 1]
```

## Reproduktivnost

Seed je fiksiran (`RANDOM_STATE = 42` u [src/config.py](src/config.py)), tako da
svako pokretanje daje iste rezultate.
