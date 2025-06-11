import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# === 1. Carregar sentenças ===
df = pd.read_csv("ORDEM.csv", sep='|', engine='python', on_bad_lines='skip', header=0)
df.columns = ["FileName", "ResearchQuestion", "Theme", "Subject", "Sentence", "Drop"]
df = df.drop(columns=["Drop"], errors="ignore").dropna(subset=["Sentence"])
sentences = df["Sentence"].astype(str).tolist()


# === 2. Pré-processamento simples ===
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)


sentences_clean = [preprocess(s) for s in sentences]

# === 3. Vetorização TF-IDF ===
vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
tfidf = vectorizer.fit_transform(sentences_clean)

# === 4. Topic Modeling (NMF) ===
n_topics = 5
nmf_model = NMF(n_components=n_topics, random_state=42)
W = nmf_model.fit_transform(tfidf)
H = nmf_model.components_

# === 5. Associar frases ao tópico dominante ===
dominant_topic = W.argmax(axis=1)
df['DominantTopic'] = dominant_topic

# === 6. Mostrar tópicos e palavras-chave ===
feature_names = vectorizer.get_feature_names_out()
top_words = []
for topic_idx, topic in enumerate(H):
    words = ", ".join([feature_names[i] for i in topic.argsort()[:-9:-1]])
    top_words.append(words)
    print(f"\n🔹 Tópico {topic_idx + 1}: {words}")

df['TopicKeywords'] = df['DominantTopic'].apply(lambda i: top_words[i])

# === 7. Salvar frases com tópicos ===
df.to_csv("frases_com_topicos.csv", index=False)

# === 8. WordCloud por tópico ===
for topic_idx in range(n_topics):
    topic_sentences = df[df['DominantTopic'] == topic_idx]['Sentence']
    text = ' '.join(topic_sentences).lower()
    text = preprocess(text)

    wordcloud = WordCloud(stopwords=stop_words, background_color='white', max_words=100).generate(text)

    plt.figure(figsize=(6, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud - Tópico {topic_idx + 1}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"wordcloud_topico_{topic_idx + 1}.png")
    plt.close()
