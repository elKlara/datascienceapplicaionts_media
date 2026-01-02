import os
import pandas as pd
from transformers import pipeline

# Pfad zum Ordner
ordner = r"C:\Users\Klara Ellensohn\Documents\GitHub\datascienceapplicaionts_media\sentiment_project"

# Modell laden
sentiment = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")

# Funktion: Text in kleine Stücke teilen (max. 200 Wörter)
def split_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

daten = []

for dateiname in os.listdir(ordner):
    if dateiname.endswith(".txt"):
        pfad = os.path.join(ordner, dateiname)
        with open(pfad, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = split_text(text)

        chunk_results = []
        for chunk in chunks:
            result = sentiment(chunk)[0]
            chunk_results.append(result)

        # Mehrheitsvoting
        labels = [r["label"] for r in chunk_results]
        final_label = max(set(labels), key=labels.count)

        # Durchschnittlicher Score
        avg_score = sum(r["score"] for r in chunk_results) / len(chunk_results)

        daten.append({
            "datei": dateiname,
            "label": final_label,
            "score": avg_score
        })

df = pd.DataFrame(daten)

print(df["label"].value_counts())
print(df.head())

df.to_csv("sentiment_ergebnisse.csv", index=False)
