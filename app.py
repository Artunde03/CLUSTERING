"""
Estudio Comparativo de Modelos de Aprendizaje No Supervisado.
Algoritmos: K-Medias, Clustering Jerárquico, DBSCAN.

Autor: [PAZ LOAIZA ARTURO JOSUE / QUISPE BRAVO KELVIN RONNY]
Fecha: Diciembre 2025
Descripción: Este script genera datos sintéticos de clientes, aplica tres modelos
de clustering y visualiza los resultados comparativos.
"""

# 1. IMPORTACIÓN DE LIBRERÍAS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Configuración de estilo para las gráficas
sns.set(style="whitegrid")

def generar_datos():
    """
    Genera un conjunto de datos sintético simulando 'Mall Customers'.
    Retorna: DataFrame de pandas.
    """
    np.random.seed(42) # Semilla para reproducibilidad (siempre saldrán los mismos datos)
    
    # Simulación de clusters (Ingresos vs Gasto)
    # Formato: [Media_Ingreso, Media_Gasto], [Desviacion, Desviacion], cantidad
    c1 = np.random.normal([25, 20], [5, 5], (30, 2))  
    c2 = np.random.normal([25, 80], [5, 5], (30, 2))  
    c3 = np.random.normal([85, 20], [10, 5], (30, 2)) 
    c4 = np.random.normal([85, 80], [10, 5], (30, 2)) 
    c5 = np.random.normal([55, 50], [10, 10], (60, 2)) 
    ruido = np.random.uniform([15, 0], [130, 100], (10, 2)) # Datos aleatorios (ruido)

    data = np.concatenate([c1, c2, c3, c4, c5, ruido])
    df = pd.DataFrame(data, columns=['Ingresos', 'Puntuacion_Gasto'])
    return df

def main():
    # --- PASO 1: PREPROCESAMIENTO ---
    print("Generando datos y preprocesando...")
    df = generar_datos()
    
    # Estandarización: Vital para que las distancias sean justas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Configuración del panel de gráficos
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3)

    # --- PASO 2: MODELO K-MEDIAS ---
    print("Ejecutando K-Medias...")
    k = 5
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    y_kmeans = kmeans.fit_predict(X_scaled)
    sil_kmeans = silhouette_score(X_scaled, y_kmeans)

    # Visualización K-Medias
    axes[0, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis', s=50)
    axes[0, 0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                       s=200, c='red', marker='X', label='Centroides')
    axes[0, 0].set_title(f'K-Medias (K={k})\nSilueta: {sil_kmeans:.3f}')
    axes[0, 0].legend()

    # --- PASO 3: CLUSTERING JERÁRQUICO ---
    print("Ejecutando Clustering Jerárquico...")
    hc = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
    y_hc = hc.fit_predict(X_scaled)
    sil_hc = silhouette_score(X_scaled, y_hc)

    # Visualización Jerárquico
    axes[0, 1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_hc, cmap='plasma', s=50)
    axes[0, 1].set_title(f'Jerárquico (Ward, K={k})\nSilueta: {sil_hc:.3f}')

    # Dendrograma (Visualización auxiliar)
    Z = linkage(X_scaled, method='ward')
    dendrogram(Z, ax=axes[1, 0], truncate_mode='lastp', p=15)
    axes[1, 0].set_title('Dendrograma (Estructura del Árbol)')

    # --- PASO 4: DBSCAN ---
    print("Ejecutando DBSCAN...")
    # Eps y min_samples ajustados experimentalmente para estos datos
    dbscan = DBSCAN(eps=0.35, min_samples=4)
    y_dbscan = dbscan.fit_predict(X_scaled)

    # Visualización DBSCAN (Manejando el ruido visualmente)
    unique_labels = set(y_dbscan)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k_lbl, col in zip(unique_labels, colors):
        if k_lbl == -1:
            col = [0, 0, 0, 1] # Color negro para ruido
            label = 'Ruido'
        else:
            label = f'Cluster {k_lbl}'
        
        class_member_mask = (y_dbscan == k_lbl)
        xy = X_scaled[class_member_mask]
        axes[1, 1].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), 
                        markeredgecolor='k', markersize=8, label=label)

    axes[1, 1].set_title(f'DBSCAN (Detección de Ruido)')
    axes[1, 1].legend(loc='upper right', fontsize='x-small')

    print("Generando gráficos finales...")
    plt.show()

if __name__ == "__main__":
    main()
