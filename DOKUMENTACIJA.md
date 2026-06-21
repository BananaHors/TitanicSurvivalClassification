# Dokumentacija — Titanic Survival Classification

Ovde sam zapisao kako projekat radi i, što mi je bilo bitnije, zašto sam pojedine
stvari uradio baš tako. Pratim manje-više redosled iz specifikacije: opis
problema, obrada podataka, modelovanje, rezultati i tumačenje.

## 1. Problem

Treba predvideti da li je putnik preživeo (`Survived` = 1) ili nije (= 0), samo na
osnovu fajla `data/data.csv` (Titanik train.csv, 891 putnik). Pored predikcije,
zanima me i koji atributi najviše utiču na ishod.

Jedna stvar koju sam odmah primetio: klase nisu izbalansirane — preživelo je
38.4% putnika. Zbog toga za izbor modela ne gledam tačnost (koja zna da zavara kad
je jedna klasa veća), nego F1-skor, koji uzima u obzir i preciznost i odziv na
manjinskoj klasi (preživeli).

## 2. Kako je sve povezano

Podelio sam kod na module, svaki radi jednu stvar, a `main.py` ih spaja u
pipeline. Tok je otprilike:

1. `data_loader` učita CSV i izvuče pregled praznih polja.
2. `features` pravi nove atribute (Title, FamilySize, IsAlone, Deck).
3. `visualizations` crta EDA grafici.
4. Podela na train/test (stratifikovano, 80/20).
5. `preprocessing` puni prazna polja, skalira i kodira.
6. `model` poredi tri modela kroz unakrsnu validaciju, izabere najbolji i oceni
   ga na test skupu.
7. Snime se metrike, grafici i sam model.


## 3. Kolone

| Kolona | Tip |
|---|---|---|
| `PassengerId` | id |
| `Survived` | cilj |
| `Pclass` | ordinalna |
| `Name` | tekst |
| `Sex` | kategorijska |
| `Age` | numerička | 
| `SibSp` | numerička 
| `Parch` | numerička |
| `Ticket` | tekst |
| `Fare` | numerička |
| `Cabin` | tekst |
| `Embarked` | kategorijska |

## 4. Prazna polja

Tri kolone imaju prazna polja:

| Kolona | Nedostaje | % |
|---|---|---|
| `Cabin` | 687 | 77.10% |
| `Age` | 177 | 19.87% |
| `Embarked` | 2 | 0.22% |

**Age** popunjavam medijanom po tituli, a ne jednom globalnom vrednošću. Razlog je
prost: titula dosta govori o uzrastu — "Master" su dečaci pa imaju mnogo nižu
tipičnu starost od "Mr". Uzeo sam medijanu, a ne prosek, jer je otpornija na
ekstreme. Ako se u testu pojavi titula koju nisam video u treningu, padam na
globalnu medijanu da ne pukne.

**Embarked** ima samo 2 prazna polja, pa ih popunjavam najčešćom lukom (S =
Southampton). Uticaj je zanemarljiv.

**Fare** nema praznih u trening skupu, ali sam imputaciju (medijanom) svejedno
ostavio za svaki slučaj na novim podacima.

**Cabin** je prazna u 77% slučajeva, pa je kao takva beskorisna. Ali prvo slovo
broja kabine je oznaka palube (`C85 → C`), što nešto govori o položaju na brodu.
Zato iz nje izvlačim `Deck`, a sve prazno ide u `U` (unknown). Retke palube (manje
od `MIN_CATEGORY = 5` putnika, npr. `T` sa 1 i `G` sa 4) takođe spajam u `U` —
više o tom pragu u sledećem odeljku.

Umesto da ručno nabrajam koje su palube retke, koristim jedan prag
(`config.MIN_CATEGORY`). Tako je pravilo dosledno i mogu da ga potkrepim brojevima
preko `python explore.py`, umesto da prolazim kroz CSV peške.

## 5. Novi atributi

Sve u [src/features.py](src/features.py) je "po redu" (gleda jedan red, ne uči iz
raspodele), pa može i pre podele na train/test bez problema s curenjem.

- **`Title`** iz `Name` — vadim titulu između zareza i tačke (`"Braund, Mr. Owen"
  → Mr`). Prvo normalizujem jezičke varijante (`Mlle → Miss`, `Mme → Mrs`,
  `Ms → Miss`), a tek presavijanje retkih titula (< 5) u `Rare` radim kasnije, u
  preprocessing-u, jer to zavisi od učestanosti pa mora da se uči na treningu.
  Puno ime je beskorisno (svako je jedinstveno), a titula lepo sažima pol, status
  i grubo uzrast.
- **`FamilySize`** = `SibSp + Parch + 1` — jasnija slika veličine porodice nego
  dve odvojene kolone.
- **`IsAlone`** = 1 ako je `FamilySize == 1` — samci se ponašaju drugačije.
- **`Deck`** iz `Cabin` — objašnjeno gore.

Izbacio sam `PassengerId` (identifikator), `Name` (zamenjena titulom), `Ticket`
(unique je za svaku osobu) i `Cabin` (zamenjena palubom).

## 6. Kodiranje

U [src/preprocessing.py](src/preprocessing.py) `ColumnTransformer` radi sledeće:

- numeričke kolone (`Age, Fare, SibSp, Parch, FamilySize, Pclass`) prolaze kroz
  `StandardScaler`. To je bitno za logističku regresiju (da se kolone različitih
  opsega porede pošteno); stablima ne smeta.
- kategorijske (`Sex, Embarked, Title, Deck`) idu na `OneHotEncoder`. One-hot zato
  što tu nema prirodnog redosleda, pa bi obično brojčano kodiranje uvelo lažan
  poredak. Sa `handle_unknown="ignore"` ne puca ako se u testu pojavi nepoznata
  kategorija.
- `Pclass` namerno ne kodiram one-hot — ostavljam je kao broj, jer 1 < 2 < 3
  stvarno ima smisao (socio-ekonomski rang).
- `IsAlone` je već 0/1, pa prolazi netaknut.

## 7. Modeli i izbor

Poredim tri prilično različita modela (u [src/model.py](src/model.py)), svaki u
istom pipeline-u:

1. logistička regresija — jednostavan, linearan, lako se tumači;
2. Random Forest — šuma stabala, hvata nelinearnosti;
3. Gradient Boosting — često najjači na ovakvim tabelarnim podacima.

Najboljeg biram stratifikovanom 5-fold unakrsnom validacijom po F1, na trening
skupu. Onda ga istreniram na celom treningu i tek na kraju ocenim na test skupu
(20%, koji model dotad nije video).

| Model | F1 (CV) |
|---|---|
| LogisticRegression | 0.7677 ± 0.0334 |
| RandomForest | 0.7442 ± 0.0180 |
| GradientBoosting | 0.7535 ± 0.0297 |

Pobedila je logistička regresija na ovom skupu podataka.

## 8. Rezultati na test skupu

| Metrika | Vrednost | Šta znači |
|---|---|---|
| tačnost | 0.8492 | ukupan udeo tačnih predikcija |
| preciznost | 0.8182 | kad kaže "preživeo", koliko je puta u pravu |
| odziv | 0.7826 | od stvarno preživelih, koliko ih je našao |
| F1 | 0.800 | balans preciznosti i odziva |
| ROC-AUC | 0.8726 | koliko dobro razdvaja klase preko svih pragova |

Pripadajući grafici su matrica konfuzije
([10](outputs/figures/10_matrica_konfuzije.png)) i ROC kriva
([11](outputs/figures/11_roc_kriva.png)).

## 9. Važnost atributa

Za važnost koristim permutacionu važnost (u `model.get_feature_importance`):
nasumično izmešam vrednosti jednog atributa i vidim koliko F1 padne. Radim to na
test skupu i radi za bilo koji model, pa mi je delovalo poštenije od gledanja
koeficijenata samog modela. Računam je na nivou izvedenih atributa (a ne
pojedinačnih one-hot kolona) da bude čitljivije.

Najjači atributi (grafik [12](outputs/figures/12_vaznost_atributa.png)):

1. `Title` (~0.20) — sažima pol, uzrast i status;
2. `Sex` (~0.19) — pol kao samostalan, dominantan faktor;
3. `Age` (~0.10) — uzrast.

Za njima idu `Pclass`, `Fare` i `FamilySize`.

### 9.1. Zašto su neki atributi niski

Ovo me je u početku zbunilo, pa sam proverio. Permutaciona važnost meri koliko
atribut dodaje *preko ostalih*, a ne koliko je sam po sebi povezan sa ishodom.
Zato nešto može da izgleda važno na EDA grafiku, a da ima nisku važnost.

`Parch` je nizak jer je redundantan — pošto je `FamilySize = SibSp + Parch + 1`,
kad pokvarim `Parch` model istu informaciju i dalje ima u `FamilySize`, pa F1
jedva mrdne.

`Embarked` je zanimljiviji slučaj. Grafik
[09](outputs/figures/09_prezivljavanje_po_luci.png) pokazuje jasan skok (C ≈ 55%,
Q ≈ 39%, S ≈ 34%), pa deluje da luka mnogo znači. Ali kad sam pogledao sastav
putnika, ispalo je da je Cherbourg bio pun prve klase:

| Luka | % 1. klase | prosečna karta | preživljavanje |
|---|---|---|---|
| C | 51% | 60 £ | 55% |
| Q | 3% | 13 £ | 39% |
| S | 20% | 27 £ | 34% |

Kad fiksiram klasu, razlika između luka se uglavnom izgubi (grafik
[13](outputs/figures/13_luka_unutar_klase.png)) — u 3. klasi su C i Q oba ~38%:

| Luka | 1. klasa | 2. klasa | 3. klasa |
|---|---|---|---|
| C | 0.69 | 0.53 | 0.38 |
| Q | 0.50 | 0.67 | 0.38 |
| S | 0.58 | 0.46 | 0.19 |

Pošto model već koristi `Pclass` i `Fare`, kvarenje `Embarked` skoro ništa ne
menja — signal luke je u stvari klasa prerušena u luku. (Ima i mali pravi ostatak:
treća klasa iz Southamptona preživljava upadljivo ređe, 19%, ali je to slab i
bučan efekat.)

Zaključak koji sam izvukao: EDA grafici pokazuju obrasce, a važnost atributa
pokazuje koliko je taj obrazac jedinstven — vredi gledati oba, jer niska važnost
ne znači da je atribut bezvredan, nego često da je ista informacija već negde
drugde.

## 10. Šta sam zaključio

Glavni nalazi (ovo se i automatski upisuje u
[outputs/metrics/izvestaj.txt](outputs/metrics/izvestaj.txt)):

- Pol je ubedljivo najjači faktor: žene preživljavaju ~74%, muškarci ~19% — klasično
  "žene i deca prvi".
- Klasa jako utiče: 1. klasa ~63%, 2. klasa ~47%, 3. klasa ~24%.
- Titula spaja pol i uzrast; "Master" (dečaci) i "Mrs/Miss" imaju visoke stope.
- Cena karte prati klasu — skuplja karta, veća šansa.
- Veličina porodice deluje nelinearno: samci i jako velike porodice preživljavaju
  ređe od malih (2–4 člana).

## 11. Šta bi moglo bolje

- Podešavanje hiperparametara (GridSearch/RandomizedSearch) i pomeranje praga
  odlučivanja ako bih hteo da naglasim preciznost ili odziv.
- Pametnija imputacija (model-based) i analiza grešaka — koje tipove putnika model
  najčešće promaši.

## 12. Šta se sve generiše

| Putanja | Sadržaj |
|---|---|
| `outputs/figures/01–09_*.png` | EDA grafici |
| `outputs/figures/10–13_*.png` | matrica konfuzije, ROC, važnost atributa, luka-po-klasi |
| `outputs/metrics/metrics.json` | sve metrike + CV + važnost (mašinski čitljivo) |
| `outputs/metrics/izvestaj.txt` | tumačenje + klasifikacioni izveštaj |
| `outputs/models/best_model.joblib` | ceo pipeline (obrada + model) |
