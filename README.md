# Social-Media-Usage-and-Emotional-Well-Being
Data set pro natrénování ML modelu predikce pro zaměření příspěvku na konkrétního uživatele.

Data set (train.csv) bude rozdělen do 3 částí - na trénovací, testovací a validační část.
Vstupní features budou age, gender, platform, daily_usage_time_spent, likes_received, comments_received, post_per_day a message_sent.
Výstupní hodnota bude "The dominant emotional state of the user during the day (e.g., Happiness, Sadness, Anger, Anxiety, Boredom, Neutral)".

zdroj: https://doi.org/10.34740/kaggle/dsv/8460631 

## Prediktor emocionálního stavu na základě chování na sociálních sítích

Ke spuštění projektu jsou potřeba soubory:

```
MLModel.py, Test.py, train.csv
```

Tento projekt načítá a předzpracovává data o uživatelích sociálních sítí, trénuje Adaptive liner neuron pro klasifikaci emocionálního stavu (pozitivní - 1 vs. negativní - 0) a vyhodnocuje výsledky pomocí metrik a grafů.

## Funkcionalita

Projekt obsahuje následující části:

- Načtení CSV dat z lokálního souboru.
- Čištění dat a oprava chyb (např. záměněné hodnoty mezi sloupci).
- Převedení dat do formátu `pandas.DataFrame`, včetně převodu kategorických proměnných pomocí one-hot encodingu.
- Trénování lineárního neronu pomocí vlastního algoritmu:
  - Rozdělení dat (train.csv) na trénovací, validační a testovací množiny.
  - Normalizace vstupních dat (`StandardScaler`).
  - Využití metody early stopping ke snížení přeučení.
  - Výpočet přesnosti a ztrátové funkce
- Predikce emocionálního stavu na nově vygenerovaných datech z API Mockaroo uložených do .json file.
- Vizualizace výsledků pomocí koláčových grafů (skutečné dle souboru train.csv vs. predikce emoce z API dat).

## Struktura projektu

Projekt se skládá z hlavního skriptu a několika pomocných tříd a funkcí:

- `DataLoader`: třída pro načítání a zpracování CSV souboru.
- `DataFrameBuilder`: třída pro převod dat na `DataFrame`.
- `sigmoid`: aktivační funkce používaná v neuronu.
- `train_neuron`: hlavní funkce pro trénování Adaline.
- `Mockaroo API`: generování testovacích dat ze vzdáleného API.


## Instalace a požadavky

Projekt je psán pro Python 3.8+ a využívá následující knihovny:  
`pandas`, `numpy`, `matplotlib`, `scikit-learn`, `requests`.

Knihovny lze nainstalovat příkazem:
```
pip install -r requirements.txt
```
nebo jednotlivě:
```
pip install pandas numpy matplotlib scikit-learn requests
```

## Spuštění

Nejprve je potřeba upravit cestu ke vstupnímu CSV souboru ve skriptu (např. `train.csv`).

Po úpravě spusť skript:
```
MLModel.py
```

Skript načte a vyčistí data, natrénuje model a vypíše klasifikační výsledky.

## Výstupy

Po úspěšném spuštění skriptu získáš:

- Přesnost klasifikace na testovacích datech.
- Klasifikační report (precision, recall, f1-score).
- Vizualizaci pomocí dvou koláčových grafů (skutečné dle souboru train.csv vs. predikce emoce z API dat).
- Zobrazení naučených vah neuronové sítě pro každou vstupní proměnnou.

## Generování nových dat

Projekt využívá veřejné API [Mockaroo](https://mockaroo.com) pro vygenerování testovacích dat s realistickými hodnotami. Tato data jsou následně převedena a použita k novým predikcím modelu.

## Autor

Aneta Strnadová  

