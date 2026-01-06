import os
import pandas as pd
from transformers import pipeline, AutoTokenizer

# =========================
# Pfade
# =========================
eingabe_ordner = r"C:\Users\Klara Ellensohn\Documents\GitHub\datascienceapplicaionts_media\sentiment_project"
ergebnis_ordner = r"C:\Users\Klara Ellensohn\Documents\GitHub\datascienceapplicaionts_media\sentiment_ergebnisse2.0"

os.makedirs(ergebnis_ordner, exist_ok=True)

# =========================
# Modell & Tokenizer laden
# =========================
modell_name = "oliverguhr/german-sentiment-bert"

sentiment = pipeline(
    "sentiment-analysis",
    model=modell_name,
    tokenizer=modell_name
)

tokenizer = AutoTokenizer.from_pretrained(modell_name)

# =========================
# Token-basiertes Chunking
# =========================
def split_text_tokens(text, max_tokens=450):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk_tokens))
    return chunks

# =========================
# Sentiment Aggregation
# =========================
label_map = {
    "negative": -1,
    "neutral": 0,
    "positive": 1
}

def label_weight(label):
    if label == "neutral":
        return 1.2  # Neutral etwas stärker gewichten
    return 1.0

# =========================
# Analyse
# =========================
ergebnisse = []

for dateiname in os.listdir(eingabe_ordner):
    if not dateiname.endswith(".txt"):
        continue

    pfad = os.path.join(eingabe_ordner, dateiname)

    with open(pfad, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = split_text_tokens(text)

    if len(chunks) == 0:
        continue

    # Batch-Analyse (schneller!)
    chunk_results = sentiment(chunks, batch_size=8)

    labels = [r["label"] for r in chunk_results]

    weighted_scores = [
        label_map[r["label"]] * r["score"] * label_weight(r["label"])
        for r in chunk_results
    ]

    final_score = sum(weighted_scores) / len(weighted_scores)

    # Finales Label anhand Schwellenwert
    if final_score > 0.2:
        final_label = "positive"
    elif final_score < -0.2:
        final_label = "negative"
    else:
        final_label = "neutral"

    ergebnisse.append({
        "datei": dateiname,
        "final_label": final_label,
        "sentiment_score": round(final_score, 3),
        "positive_chunks": labels.count("positive"),
        "neutral_chunks": labels.count("neutral"),
        "negative_chunks": labels.count("negative"),
        "anzahl_chunks": len(chunks)
    })

# =========================
# Ergebnisse speichern
# =========================
df = pd.DataFrame(ergebnisse)

csv_pfad = os.path.join(ergebnis_ordner, "sentiment_ergebnisse.csv")
df.to_csv(csv_pfad, index=False, encoding="utf-8")

# Übersicht ausgeben
print("Sentiment-Verteilung:")
print(df["final_label"].value_counts())
print("\nBeispiel-Ergebnisse:")
print(df.head())

print(f"\nErgebnisse gespeichert in: {csv_pfad}")
