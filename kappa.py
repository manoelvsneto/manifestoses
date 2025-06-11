from sre_constants import AT_END
import numpy as np
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from statsmodels.stats.inter_rater import fleiss_kappa, cohens_kappa
import pandas as pd
# Função de interpretação permanece a mesma
def interpretar_kappa(valor):
    if valor is None or np.isnan(valor):
        return "Indefinido"
    # Ajuste para incluir o limite inferior
    elif valor < 0.20:
        return "Pouco acordo"
    elif valor < 0.40:
        return "Acordo razoável"
    elif valor < 0.60:
        return "Acordo moderado"
    elif valor < 0.80:
        return "Acordo bom"
    else:
        return "Acordo muito bom"

def calcular_metricas_corrigido(total_avaliados, intersecao, exclusivos_p1, exclusivos_p2,rejeitados_ambos,encontrados_p1,encontrados_p2):
    # Vetores binários para métricas do sklearn (considerando todos os itens)
    p1_vector = [1] * intersecao + [1] * exclusivos_p1 + [0] * exclusivos_p2 + [0] * rejeitados_ambos
    p2_vector = [1] * intersecao + [0] * exclusivos_p1 + [1] * exclusivos_p2 + [0] * rejeitados_ambos

    jaccard = jaccard_score(p1_vector, p2_vector)
    precision = precision_score(p1_vector, p2_vector)
    recall = recall_score(p1_vector, p2_vector)
    f1 = f1_score(p1_vector, p2_vector)

    # Tabela de contingência para Cohen's Kappa (estava correta)
    table = np.array([
        [intersecao, exclusivos_p1],
        [exclusivos_p2, rejeitados_ambos]
    ])

    # O parâmetro 'return_results' não existe, a função já retorna um objeto de resultados
    kappa_cohen_obj = cohens_kappa(table)
    kappa_cohen = kappa_cohen_obj.kappa

    # --- CORREÇÃO PRINCIPAL: Matriz para Fleiss/Randolph Kappa ---
    # Colunas: [Votos para 'Selecionou', Votos para 'Não Selecionou']
    ratings_matrix = []
    # Acordo em 'Sim': ambos votaram 'Selecionou'
    ratings_matrix.extend([[2, 0]] * intersecao)
    # Desacordo: um votou 'Selecionou', o outro 'Não'
    ratings_matrix.extend([[1, 1]] * (exclusivos_p1 + exclusivos_p2))
    # Acordo em 'Não': ambos votaram 'Não Selecionou'
    ratings_matrix.extend([[0, 2]] * rejeitados_ambos)

    ratings_matrix = np.array(ratings_matrix)

    # O resultado agora será idêntico ao de Cohen
    kappa_fleiss = fleiss_kappa(ratings_matrix, method='fleiss')

    return {
        "Cohen's Kappa": round(kappa_cohen, 4),
        "Descrição Kappa": interpretar_kappa(kappa_cohen),
        "Fleiss' Kappa": round(kappa_fleiss, 4),
        "Descrição Fleiss": interpretar_kappa(kappa_fleiss),
        "Jaccard": round(jaccard, 4),
        "F1-score": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Avaliados":total_avaliados,
        "Interseção":intersecao,
        "Exclusivos P1":exclusivos_p1,
        "Exclusivos P2":exclusivos_p2,
        "Rejeitados":rejeitados_ambos,
        "Considerados": total_avaliados - rejeitados_ambos,
        "Encontrados P1":encontrados_p1,
        "Encontrados P2":encontrados_p2
    }

if __name__ == "__main__":
    total_avaliados = 564
    rejeitados_ambos = 420
    validos = total_avaliados - rejeitados_ambos
    intersecao = 107
    encontrados_p1 = 140
    encontrados_p2 = 111
    exclusivos_p1 = encontrados_p1 -  intersecao
    exclusivos_p2 = encontrados_p2 -  intersecao
    resultados = []
    try :
      metricas = calcular_metricas_corrigido(total_avaliados, intersecao, exclusivos_p1, exclusivos_p2,rejeitados_ambos,encontrados_p1,encontrados_p2)
      resultados.append(metricas)
    except:
      print("Erro ao calcular as métricas")

    df_resultados = pd.DataFrame(resultados)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.expand_frame_repr", False)
    print(df_resultados)


