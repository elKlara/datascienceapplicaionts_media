import os
import spacy
import pandas as pd

# =========================
# Modell laden (deutsch)
# =========================
nlp = spacy.load("de_core_news_md")

# =========================
# Pfad zu deinen Artikeln
# =========================
ordner = r"C:\Users\Klara Ellensohn\Documents\GitHub\datascienceapplicaionts_media\sentiment_project"

# =========================
# Listen mit typischen Vornamen
# (kannst du jederzeit erweitern)
# =========================
maennlich = {"marco", "thomas", "carlos", "gregoritsch", "morant", "jokic", "sarrazin"}
weiblich = {"michelle", "gisin", "emma", "mikutina", "aicher"}

# =========================
# Funktion zur Geschlechtererkennung
# =========================
def detect_gender(text):
    doc = nlp(text.lower())

    found_m = False
    found_w = False

    # 1) Named Entities (Personen)
    for ent in doc.ents:
        if ent.label_ == "PER":
            name = ent.text.split()[0]
            if name in maennlich:
                found_m = True
            if name in weiblich:
                found_w = True

    # 2) Pronomen
    if " er " in text.lower():
        found_m = True
    if " sie " in text.lower():
        found_w = True

    # Ergebnislogik
    if found_m and not found_w:
        return "männlich"
    if found_w and not found_m:
        return "weiblich"
    if found_m and found_w:
        return "gemischt"
    return "unklar"

# =========================
# Analyse aller Artikel
# =========================
ergebnisse = []

for datei in os.listdir(ordner):
    if not datei.endswith(".txt"):
        continue

    pfad = os.path.join(ordner, datei)

    with open(pfad, "r", encoding="utf-8") as f:
        text = f.read()

    gender = detect_gender(text)

    ergebnisse.append({
        "datei": datei,
        "geschlecht": gender
    })

# =========================
# Ergebnisse speichern
# =========================
df = pd.DataFrame(ergebnisse)
print(df["geschlecht"].value_counts())
print(df)

df.to_csv("geschlechteranalyse.csv", index=False, encoding="utf-8")
