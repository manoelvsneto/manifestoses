import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# === 1. Carregar as frases ===
df = pd.read_csv("ORDEM.csv", sep='|', engine='python', on_bad_lines='skip', header=0)
df.columns = ["FileName", "ResearchQuestion", "Theme", "Subject", "Sentence", "Drop"]
df = df.drop(columns=["Drop"], errors="ignore").dropna(subset=["Sentence"])

# === 2. Carregar as stopwords customizadas ===
stop_df = pd.read_csv("stopwords_1000.csv")
stopwords = set(stop_df['StopWord'].dropna().str.lower())

# === 3. Juntar todas as frases ===
texto = ' '.join(df['Sentence'].dropna().astype(str)).lower()

# === 4. Gerar a WordCloud com as stopwords ===
wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color='white',
    stopwords=stopwords,
    max_words=200
).generate(texto)

# === 5. Exibir a nuvem de palavras ===
plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Wordcloud com Stopwords Customizadas", fontsize=20)
plt.axis('off')
plt.tight_layout()
plt.show()

# === 6. Salvar opcional ===
wordcloud.to_file("wordcloud_filtrada.png")

