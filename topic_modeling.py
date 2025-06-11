import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import nltk
nltk.download('stopwords')
import re

# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# === 1. Carregar e limpar as sentenças ===
df = pd.read_csv("ORDEM.csv", sep='|', engine='python', on_bad_lines='skip', header=0)
df.columns = ["FileName", "ResearchQuestion", "Theme", "Subject", "Sentence", "Drop"]
df = df.drop(columns=["Drop"], errors="ignore").dropna(subset=["Sentence"])
sentences = df["Sentence"].astype(str).tolist()

# Função de pré-processamento simples
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # remove pontuação e lowercase
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

sentences_clean = [preprocess(s) for s in sentences]

# === 2. Vetorização ===
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
tfidf = tfidf_vectorizer.fit_transform(sentences_clean)

count_vectorizer = CountVectorizer(max_df=0.9, min_df=2)
count = count_vectorizer.fit_transform(sentences_clean)

# === 3. Aplicar Modelos ===
n_topics = 7

# LDA
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(count)

# NMF
nmf = NMF(n_components=n_topics, random_state=42)
nmf.fit(tfidf)

# === 4. Mostrar os tópicos ===
def extract_topics(model, feature_names, no_top_words=8):
    topic_data = []
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.expand_frame_repr", False)
    for topic_idx, topic in enumerate(model.components_):
        top_terms = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_data.append({
            "Topic": f"Tópico {topic_idx + 1}",
            "Top Terms": ", ".join(top_terms)
        })
    return pd.DataFrame(topic_data)

print("\n=====  LDA =====")
lda_topics_df = extract_topics(lda, count_vectorizer.get_feature_names_out())
print(lda_topics_df)

print("\n===== 🔍  NMF =====")
nmf_topics_df = extract_topics(nmf, tfidf_vectorizer.get_feature_names_out())
print(nmf_topics_df)