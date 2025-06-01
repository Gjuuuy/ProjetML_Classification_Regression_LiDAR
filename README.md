# PROJET MACHINE LEARNING - MODÉLISATION SUR DONNÉES LIDAR

Ce projet de machine learning applique des techniques de **classification** et de **régression** à des données LiDAR pour l'analyse automatique d'environnements naturels, combinant l'identification de classes d'objets et la prédiction de hauteurs.

## Objectifs

Le projet est structuré en deux axes principaux utilisant des données LiDAR issues du réseau NEON (National Ecological Observatory Network) :

### 1. Classification des éléments du paysage
* **A. Chargement des Données**
* **B. Analyse du Dataset :** Identification de différentes classes :  
   - **Classe 1** : Sol nu  
   - **Classe 2** : Végétation basse  
   - **Classe 5** : Bâtiment  
   - **Classes 6-7** : Bruit (éliminées du dataset)  
* **C. Préparation du Dataset**
* **D. Classification :** Implémentation et évaluation de plusieurs modèles de classification supervisée :  
   - **Random Forest Classifier**  
   - **LightGBM Classifier** (GBM = Gradient Boosting Model)  
   - **MLP Classifier** (réseau de neurones multi-couches)  
   - **Voting Classifier** (Hard & Soft) : combinaison des modèles ci-dessus  
* **E. Bilan des Méthodes de Classification**


### 2. Régression pour la prédiction de hauteur
Prédiction de la variable continue **Z (hauteur)** à partir des autres caractéristiques LiDAR :
* Modélisation de la relation entre coordonnées, intensité, angles et hauteur
* Prédiction précise des altitudes pour la reconstruction 3D
* Évaluation de la qualité de prédiction avec métriques adaptées

## Structure du projet

```
├── notebooks/
│   ├── LIDAR_classification.ipynb    # Partie 1 : Classification des classes paysagères
│   ├── LIDAR_regression.ipynb        # Partie 2 : Régression pour prédiction de hauteur
│   └── environment.yml               # # Environnement Conda contenant toutes les dépendances nécessaires à l'exécution
└── README.md                         # Ce fichier
```

## Données

### Source des données
Les données proviennent du **NEON (National Ecological Observatory Network)** :
- **Dataset** : Discrete return LiDAR point cloud (DP1.30003.001)
- **Format** : Fichier .laz (LAS compressé)
- **Volume** : Plus de 16 millions de points LiDAR initialement
- **Échantillonnage** : 1% des données soit ~165 000 observations pour l'entraînement

### Variables utilisées

**Pour la classification :**
Après nettoyage et sélection des features :
- **X, Y** : Coordonnées géographiques
- **intensity** : Intensité du retour LiDAR
- **return_number** : Numéro du signal retour
- **num_returns** : Nombre total de signaux retours
- **scan_angle** : Angle de scan
- **gps_time** : Temps GPS
- **red, green, blue** : Valeurs RGB colorimétriques

**Pour la régression :**
Variables explicatives pour prédire la hauteur Z :
- **X, Y** : Coordonnées géographiques (corrélation avec relief)
- **intensity** : Intensité du retour (liée aux matériaux/surfaces)
- **return_number, num_returns** : Informations sur les retours multiples
- **scan_angle** : Angle de scan du capteur
- **gps_time** : Horodatage GPS
- **red, green, blue** : Informations colorimétriques

## Méthodologie Machine Learning

Le projet suit une approche rigoureuse basée sur les bonnes pratiques du machine learning supervisé.

### Étapes principales

1. **Préparation des données**
   - Échantillonnage représentatif (1% du dataset original)
   - Suppression des classes de bruit (6, 7)
   - Élimination des variables redondantes ou non informatives
   - Prévention du data leakage (suppression de `Z`, `user_data`)

2. **Preprocessing**
   - Normalisation MinMaxScaler dans l'intervalle [-1, 1]
   - Équilibrage des classes par sous-échantillonnage
   - Conservation de la polarité des données (coordonnées GPS)

3. **Modélisation**
   - Séparation train/test (90%/10%)
   - Validation croisée (5 folds pour classification, 10 folds pour régression)
   - Optimisation d'hyperparamètres avec `RandomizedSearchCV`
   - Détection et suppression des valeurs aberrantes (Z-score > 3 pour régression)
   - Évaluation avec métriques adaptées à chaque tâche
   - Feature engineering : génération de nouvelles variables avec la méthode Brouta (régression) et sélection des plus pertinentes via l’importance des caractéristiques d’un modèle Random Forest (régression + classification)

### Évaluation

**Métriques de Classification :**
- `accuracy` : Précision globale
- `f1-score` : Score F1 par classe
- `classification_report` : Rapport détaillé
- `confusion_matrix` : Matrice de confusion normalisée

**Métriques de Régression :**
- `RMSE` : Erreur quadratique moyenne
- `R²` : Coefficient de détermination
- `MAE` : Erreur absolue moyenne
- `MSE` : Erreur quadratique
- `Biais²` et `Variance` : Décomposition biais-variance

## Modèles testés

### Classification

### 1. Random Forest Classifier
- **Hyperparamètres optimaux** : n_estimators=200, min_samples_split=2, min_samples_leaf=2
- **Performance** : Excellent sur végétation haute, bon sur végétation basse, correct sur sol nu

### 2. LightGBM Classifier  
- **Hyperparamètres optimaux** : num_leaves=31, n_estimators=300, max_depth=10, learning_rate=0.05
- **Performance** : Comparable au Random Forest avec une vitesse d'entraînement supérieure

### 3. MLP Classifier (Réseau de neurones)
- **Architecture** : hidden_layer_sizes=(100, 50), activation='tanh', solver='adam'
- **Performance** : Résultats similaires aux méthodes ensemblistes

### 4. Voting Classifier Hard & Soft (Ensemble)
- **Composition** : Combinaison des 3 modèles précédents
- **Variantes** : Voting hard et soft testées
- **Performance** : Amélioration marginale par rapport aux modèles individuels

### Régression

### 1. Linear Regression
- **Méthode** : Régression linéaire classique comme baseline
- **Validation** : Cross-validation 10-folds avec `cross_val_predict`
- **Évaluation** : Métriques RMSE, R², biais², variance

### 2. Random Forest Regressor
- **Méthode** : Ensemble d'arbres de décision pour régression
- **Avantages** : Robuste aux outliers, gestion automatique des interactions

### 3. XGBoost Regressor
- **Méthode** : Gradient boosting optimisé
- **Hyperparamètres** : Optimisation avec `RandomizedSearchCV`
- **Performance** : Généralement supérieur pour prédiction de hauteurs

### 4. K-Nearest Neighbors Regressor
- **Méthode** : Régression basée sur les k plus proches voisins
- **Particularité** : Adapté aux données avec structures locales complexes

## Résultats

### Performances en Classification
- **Classe 5 (Batiment)** : ~85% de précision (meilleur résultat)
- **Classe 2 (végétation basse)** : ~83% de précision  
- **Classe 1 (sol nu)** : ~79% de précision (classe la plus difficile)

### Performances en Régression
- **Visualisation 3D** : Reconstruction fidèle du relief et des structures
- **Modèles évalués** : Comparaison RMSE, R², MAE entre Linear Regression, Random Forest, XGBoost, 


### Observations générales
- **Classification** : Les modèles montrent des performances satisfaisantes mais manquent de diversité, la classe "sol nu" reste difficile
- **Régression** : La prédiction de hauteur bénéficie des corrélations spatiales (X, Y) et de l'intensité
- **Data leakage évité** : Suppression des variables d'altitude directe en classification
- **Preprocessing crucial** : Normalisation et gestion des outliers améliorent significativement les résultats

## Installation et lancement

### Prérequis
```bash
conda env create -f environment.yml
conda activate lidar_analysis
```

## Améliorations possibles

### Classification
1. **Nouvelles features** : Métriques de texture, indices de voisinage, caractéristiques géométriques
2. **Augmentation des données** : Utilisation d'un échantillon plus large
3. **Features engineering** : Création de variables dérivées (pentes, densités locales)
4. **Modèles avancés** : Test d'architectures de deep learning spécialisées pour les nuages de points

### Régression  
1. **Modèles ensemble** : Combinaison des meilleurs régresseurs (Stacking, Blending)
2. **Features spatiales** : Intégration de métriques de voisinage et de texture locale
3. **Régularisation** : Test de Ridge, Lasso, Elastic Net pour contrôler le surapprentissage
4. **Validation temporelle** : Si données séquentielles, validation respectant l'ordre temporel


## Auteurs

**Binôme :**
- **Abdoulaye SAKO**
- **Mathias LE BAYON**

---

*Ce projet en deux volets démontre l'application de techniques de machine learning sur des données LiDAR, combinant classification pour l'identification d'objets paysagers et régression pour la prédiction précise de hauteurs, selon les bonnes pratiques du domaine.*
