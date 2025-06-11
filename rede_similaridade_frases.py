import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === 1. Carregar dados ===
df = pd.read_csv("ORDEM.csv", sep='|', engine='python', on_bad_lines='skip', header=0)
df.columns = ["FileName", "ResearchQuestion", "Theme", "Subject", "Sentence", "Drop"]
df = df.drop(columns=["Drop"], errors="ignore").dropna(subset=["Sentence"])

# === 2. Selecionar amostra de frases ===
sentences = df['Sentence'].dropna().unique()
sample_size = min(len(sentences), len(sentences))  # Limita para não sobrecarregar o grafo
sentences_sample = sentences[:sample_size]

# === 3. Vetorização com TF-IDF ===
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(sentences_sample)

# === 4. Matriz de similaridade ===
sim_matrix = cosine_similarity(X)

# === 5. Construção do grafo ===
G_sim = nx.Graph()
for i, frase in enumerate(sentences_sample):
    G_sim.add_node(i, label=f"F{i+1}", sentence=frase)

threshold = 0.5  # Define similaridade mínima
for i in range(len(sentences_sample)):
    for j in range(i + 1, len(sentences_sample)):
        if sim_matrix[i, j] >= threshold:
            G_sim.add_edge(i, j, weight=sim_matrix[i, j])

# === 6. Remover nós isolados ===
G_sim.remove_nodes_from(list(nx.isolates(G_sim)))

# === 7. Visualizar rede ===
pos = nx.spring_layout(G_sim, k=0.8, seed=42)
plt.figure(figsize=(16, 12))
nx.draw_networkx_nodes(G_sim, pos, node_color='lightyellow', node_size=500)
nx.draw_networkx_edges(G_sim, pos, width=[G_sim[u][v]['weight']*5 for u, v in G_sim.edges()], alpha=0.6)
nx.draw_networkx_labels(G_sim, pos, labels={i: f"F{i+1}" for i in G_sim.nodes()}, font_size=8)
plt.title("Rede de Similaridade entre Frases (TF-IDF + Cosine, sem nós isolados)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()
