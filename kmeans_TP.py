import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# GÉNÉRATION DES DONNÉES
# ============================================================================

def generer_donnees_etudiants(n=100):
    """Génère des données d'étudiants avec 3 profils distincts"""
    np.random.seed(42)
    
    print("\n" + "="*80)
    print("GÉNÉRATION DES DONNÉES - Profils d'étudiants")
    print("="*80)
    
    # Profil 1: Étudiants excellents
    print("\n Profil 1: Étudiants excellents (33 individus)")
    excellents = pd.DataFrame({
        'age': np.random.randint(18, 21, 33),
        'notes': np.random.uniform(15, 20, 33),
        'absences': np.random.randint(0, 5, 33),
        'heures_etude': np.random.uniform(15, 25, 33)
    })
    
    # Profil 2: Étudiants moyens
    print(" Profil 2: Étudiants moyens (34 individus)")
    moyens = pd.DataFrame({
        'age': np.random.randint(20, 23, 34),
        'notes': np.random.uniform(10, 15, 34),
        'absences': np.random.randint(5, 15, 34),
        'heures_etude': np.random.uniform(5, 15, 34)
    })
    
    # Profil 3: Étudiants en difficulté
    print(" Profil 3: Étudiants en difficulté (33 individus)")
    difficulte = pd.DataFrame({
        'age': np.random.randint(21, 25, 33),
        'notes': np.random.uniform(5, 10, 33),
        'absences': np.random.randint(15, 30, 33),
        'heures_etude': np.random.uniform(0, 5, 33)
    })
    
    # Combiner
    data = pd.concat([excellents, moyens, difficulte], ignore_index=True)
    
    print(f"\n Total: {len(data)} étudiants générés")
    print("\n Aperçu des données:")
    print(data.head(10))
    print(f"\n Statistiques descriptives:")
    print(data.describe())
    
    return data

# ============================================================================
# NORMALISATION
# ============================================================================

def normaliser_donnees(data, features):
    """Normalise les données avec StandardScaler"""
    print("\n" + "="*80)
    print("NORMALISATION DES DONNÉES")
    print("="*80)
    
    X = data[features].copy()
    
    print(f"\n Variables sélectionnées: {features}")
    print(f" Dimensions: {X.shape[0]} lignes × {X.shape[1]} colonnes")
    
    print("\n Avant normalisation:")
    print(X.describe())
    
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    X_norm = pd.DataFrame(X_norm, columns=features)
    
    print("\n Après normalisation:")
    print(X_norm.describe())
    
    print("\n Normalisation terminée!")
    
    return X_norm, scaler

# ============================================================================
# K-MEANS
# ============================================================================

def appliquer_kmeans(X, k):
    """Applique K-Means avec k clusters"""
    print(f"\n{'='*80}")
    print(f"APPLICATION DE K-MEANS avec k={k}")
    print("="*80)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    clusters = kmeans.fit_predict(X) 
    print(f"\n Clustering terminé!")
    print(f"   • Nombre d'itérations: {kmeans.n_iter_}")
    print(f"   • Inertie (WCSS): {kmeans.inertia_:.2f}")
    print(f"   • Nombre de clusters: {k}")
    
    return kmeans, clusters

# ============================================================================
# MÉTHODE DU COUDE
# ============================================================================

def methode_coude(X, k_min=2, k_max=10):
    """Calcule et affiche la méthode du coude"""
    print("\n" + "="*80)
    print("MÉTHODE DU COUDE (ELBOW METHOD)")
    print("="*80)
    
    inertias = []
    k_range = range(k_min, k_max + 1)
    
    print("\n Calcul en cours...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        print(f"   k={k:2d} → Inertie = {kmeans.inertia_:8.2f}")
    
    # Visualisation
    plt.figure(figsize=(12, 6))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=10)
    plt.xlabel('Nombre de clusters (k)', fontsize=14, fontweight='bold')
    plt.ylabel('Inertie (WCSS)', fontsize=14, fontweight='bold')
    plt.title('Méthode du Coude - Détermination du k optimal', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_range)
    
    # Annotations
    for k, inertia in zip(k_range, inertias):
        plt.annotate(f'{inertia:.0f}', 
                    xy=(k, inertia), 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('methode_coude.png', dpi=300, bbox_inches='tight')
    print("\nGraphique sauvegardé: methode_coude.png")
    plt.show()
    
    return inertias

# ============================================================================
# ANALYSE DES CLUSTERS
# ============================================================================

def analyser_clusters(data, features, clusters, k):
    """Analyse détaillée des clusters"""
    print("\n" + "="*80)
    print(f"ANALYSE DES {k} CLUSTERS")
    print("="*80)
    
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = clusters
    
    # Analyse par cluster
    for i in range(k):
        cluster_data = data_with_clusters[data_with_clusters['cluster'] == i]
        n = len(cluster_data)
        pct = (n / len(data)) * 100
        
        print(f"\n{'─'*80}")
        print(f"🔹 CLUSTER {i} - {n} individus ({pct:.1f}%)")
        print(f"{'─'*80}")
        
        for feature in features:
            mean = cluster_data[feature].mean()
            std = cluster_data[feature].std()
            min_val = cluster_data[feature].min()
            max_val = cluster_data[feature].max()
            print(f"   {feature:15s}: μ={mean:6.2f} | σ={std:5.2f} | min={min_val:6.2f} | max={max_val:6.2f}")
    
    print("\n" + "="*80)
    
    return data_with_clusters

# ============================================================================
# VISUALISATIONS
# ============================================================================

def visualiser_2d(data, features, clusters, k):
    """Visualisation 2D des clusters"""
    print("\n Création des visualisations 2D...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    scatter = axes[0].scatter(data[features[0]], data[features[1]], 
                             c=clusters, cmap='viridis', 
                             s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel(features[0], fontsize=12, fontweight='bold')
    axes[0].set_ylabel(features[1], fontsize=12, fontweight='bold')
    axes[0].set_title(f'Clusters K-Means (k={k})', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='Cluster')
    
    # Distribution
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    colors = plt.cm.viridis(np.linspace(0, 1, k))
    bars = axes[1].bar(cluster_counts.index, cluster_counts.values, 
                       color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('Cluster', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Nombre d\'individus', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribution des individus par cluster', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(k))
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Annotations
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('clusters_2d.png', dpi=300, bbox_inches='tight')
    print(" Graphique sauvegardé: clusters_2d.png")
    plt.show()

def visualiser_3d(data, features, clusters, k):
    """Visualisation 3D des clusters"""
    if len(features) < 3:
        print("\n  Visualisation 3D nécessite au moins 3 variables")
        return
    
    print("\n Création de la visualisation 3D...")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Créer le scatter plot 3D avec des couleurs pour chaque cluster
    colors = plt.cm.viridis(np.linspace(0, 1, k))
    
    for i in range(k):
        cluster_data = data[clusters == i]
        ax.scatter(cluster_data[features[0]], 
                  cluster_data[features[1]], 
                  cluster_data[features[2]],
                  c=[colors[i]], 
                  s=100, 
                  alpha=0.6, 
                  edgecolors='black', 
                  linewidth=0.5,
                  label=f'Cluster {i}')
    
    ax.set_xlabel(features[0], fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel(features[1], fontsize=14, fontweight='bold', labelpad=10)
    ax.set_zlabel(features[2], fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(f'Clusters K-Means 3D (k={k})', fontsize=16, fontweight='bold', pad=20)
    
    # Ajouter une légende
    ax.legend(loc='upper right', fontsize=10)
    
    # Améliorer la vue
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('clusters_3d.png', dpi=300, bbox_inches='tight')
    print(" Graphique sauvegardé: clusters_3d.png")
    plt.show()

# ============================================================================
# EXPORT
# ============================================================================

def exporter_resultats(data_with_clusters, filename='resultats_kmeans.csv'):
    """Exporte les résultats en CSV"""
    data_with_clusters.to_csv(filename, index=False)
    print(f"\n Résultats exportés: {filename}")
    print(f"   • Lignes: {len(data_with_clusters)}")
    print(f"   • Colonnes: {list(data_with_clusters.columns)}")

# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================

def main():
    """Programme principal - Workflow complet"""
    
    print("\n" + "="*80)
    print(" "*25 + "K-MEANS CLUSTERING - TP")
    print(" "*20 + "Application de l'algorithme K-Means")
    print("="*80)
    
    # Étape 1: Générer les données
    data = generer_donnees_etudiants(n=100)
    
    # Étape 2: Sélectionner et normaliser
    features = ['age', 'notes', 'absences', 'heures_etude']
    X_norm, scaler = normaliser_donnees(data, features)
    
    # Étape 3: Méthode du coude
    inertias = methode_coude(X_norm, k_min=2, k_max=10)
    
    # Étape 4: Appliquer K-Means avec k optimal
    print("\n" + "="*80)
    print("Choix du k optimal d'après la méthode du coude")
    print("="*80)
    k_optimal = int(input("\nEntrez le nombre de clusters optimal (par défaut 3): ") or 3)
    
    kmeans, clusters = appliquer_kmeans(X_norm, k=k_optimal)
    
    # Étape 5: Analyser les clusters
    data_with_clusters = analyser_clusters(data, features, clusters, k_optimal)
    
    # Étape 6: Visualisations
    print("\n" + "="*80)
    print("VISUALISATIONS")
    print("="*80)
    
    visualiser_2d(data, features, clusters, k_optimal)
    visualiser_3d(data, features, clusters, k_optimal)
    
    # Étape 7: Export
    exporter_resultats(data_with_clusters)
    
    # Résumé final
    print("\n" + "="*80)
    print("TRAITEMENT TERMINÉ!")
    print("="*80)
    print("\n Fichiers générés:")
    print("   • methode_coude.png")
    print("   • clusters_2d.png")
    print("   • clusters_3d.png")
    print("   • resultats_kmeans.csv")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()