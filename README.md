# 📡 Prédiction du Churn Client — Telco Customer Churn

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Terminé-success)
![Dataset](https://img.shields.io/badge/Dataset-7043%20lignes-informational)
![Modèle](https://img.shields.io/badge/Modèle-Gradient%20Boosting-teal)

> **Projet de Machine Learning End-to-End** réalisé dans le cadre du module **Machine Learning** encadré par M. Abdallah Khemais.
> **Auteure :** Yassmine Ben Aicha — École Polytechnique de Tunisie

---

## 📑 Sommaire

- [🎯 Contexte et problématique](#-contexte-et-problématique)
- [📊 Dataset](#-dataset)
- [📁 Structure du projet](#-structure-du-projet)
- [🔬 Méthodologie](#-méthodologie)
- [📈 Résultats](#-résultats)
- [⚙️ Installation et exécution](#️-installation-et-exécution)
- [✅ Conclusion](#-conclusion)

---

## 🎯 Contexte et problématique

Dans le secteur des télécommunications, **la rétention client est un enjeu stratégique majeur** : acquérir un nouveau client coûte 5 à 7 fois plus cher que d'en fidéliser un existant. La capacité à identifier à l'avance les clients susceptibles de résilier leur abonnement — le **churn** — permet d'agir en amont : offre de fidélisation ciblée, contact proactif, tarif personnalisé.

**Problématique :**

> À partir des caractéristiques d'un client (ancienneté, services souscrits, type de contrat, facturation...), prédire s'il va résilier son abonnement (`1`) ou rester (`0`).

C'est un problème de **classification binaire supervisée**.

---

## 📊 Dataset

| Caractéristique | Détail |
|---|---|
| **Source** | Publié initialement par IBM · Référencé sur Kaggle (*Telco Customer Churn*, par blastchar) |
| **Volume** | 7 043 clients · 21 variables (20 explicatives + 1 cible `Churn`) |
| **Variables numériques** | `tenure` (ancienneté), `MonthlyCharges`, `TotalCharges` |
| **Variables catégorielles** | Type de contrat, services internet, mode de paiement, options souscrites... |
| **Variable cible** | `Churn` : 0 = reste · 1 = résilie |
| **Déséquilibre des classes** | ~73.5% non-churn · ~26.5% churn |

**Justification du choix (source externe) :** dataset propre, taille largement suffisante (> 500 lignes), problématique métier claire et pertinente, absent des sources locales tunisiennes recommandées en priorité — justification conforme aux exigences du cahier des charges.

---

## 📁 Structure du projet

```
telco-customer-churn-ml/
│
├── 📓 notebook/
│   ├── 01_EDA_draft.ipynb              ← Analyse exploratoire des données
│   ├── 02_preprocessing_draft.ipynb    ← Nettoyage, encodage, split train/test
│   ├── 03_modeling_draft.ipynb         ← Entraînement et comparaison des 4 modèles
│   └── notebook_final.ipynb            ← Version propre · démarche complète A→Z
│
├── 📂 data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv    ← Données brutes originales
│   └── processed/
│       ├── X_train.csv                 ← Variables d'entraînement (transformées)
│       ├── X_test.csv                  ← Variables de test (transformées)
│       ├── y_train.csv                 ← Cible d'entraînement
│       └── y_test.csv                  ← Cible de test
│
├── 🤖 models/
│   ├── preprocessor.pkl                ← Pipeline de transformation sauvegardé
│   └── best_model.pkl                  ← Gradient Boosting optimisé (modèle final)
│
├── 📊 presentation/
│   └── presentation.pptx               ← Support de soutenance (8 slides)
│
└── 📄 README.md
```

---

## 🔬 Méthodologie

```
EDA → Prétraitement → Modélisation (4 modèles) → Optimisation (GridSearchCV) → Évaluation → Sauvegarde
```

### 1. Analyse exploratoire (EDA) — `01_EDA_draft.ipynb`

- Distribution des variables numériques (histogrammes, KDE, boxplots)
- Analyse de la variable cible : déséquilibre des classes (~26.5% churn)
- Corrélations avec le churn : `tenure` (négatif), `MonthlyCharges` (positif)
- Analyse catégorielle : les clients en contrat **Month-to-month** résilient beaucoup plus que ceux engagés sur 1 ou 2 ans
- Détection des valeurs manquantes : 11 sur `TotalCharges`

### 2. Prétraitement — `02_preprocessing_draft.ipynb`

| Étape | Décision | Justification |
|---|---|---|
| Suppression de `customerID` | Retiré | Identifiant sans valeur prédictive |
| `TotalCharges` (11 valeurs manquantes) | Remplacé par `0` | Ces clients ont `tenure = 0` (pas encore facturés) — la médiane aurait été arbitraire |
| Outliers (méthode IQR) | Aucun traitement | Aucune variable ne présente d'outliers significatifs |
| Variables numériques | `StandardScaler` | Nécessaire pour les modèles sensibles à l'échelle (LR, SVM) |
| Variables catégorielles | `OneHotEncoder (drop='first')` | Évite la colinéarité parfaite |
| Fit du pipeline | Sur le train uniquement | Évite le **data leakage** |
| Split | 80% train · 20% test (stratifié) | Préserve les proportions de churn dans les deux sous-ensembles |

### 3. Modélisation — `03_modeling_draft.ipynb`

Quatre algorithmes aux logiques différentes ont été comparés :

| Modèle | Type | Logique |
|---|---|---|
| Logistic Regression | Linéaire | Frontière linéaire, simple et interprétable |
| Random Forest | Ensembliste (Bagging) | Agrégation de N arbres de décision indépendants |
| **Gradient Boosting** | Ensembliste (Boosting) | Correction progressive des erreurs par des arbres séquentiels |
| SVM | Marge maximale | Cherche la frontière qui maximise la marge entre les classes |

**Métrique de comparaison : F1-score** — justifié par le déséquilibre des classes (~26.5% churn). L'Accuracy seule serait trompeuse : un modèle qui prédirait toujours "non-churn" obtiendrait déjà ~73% sans rien apprendre d'utile.

### 4. Optimisation — `notebook_final.ipynb`

Le meilleur modèle (Gradient Boosting) a été optimisé via **GridSearchCV** avec validation croisée **5-fold** :

```python
param_grid = {
    "n_estimators"  : [100, 200],
    "learning_rate" : [0.05, 0.1],
    "max_depth"     : [3, 5]
}
# Meilleurs paramètres retenus :
# learning_rate=0.05 · max_depth=3 · n_estimators=200
```

---

## 📈 Résultats

### Tableau comparatif des 4 modèles

| Modèle | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| 🥇 **Gradient Boosting** | **0.806** | **0.662** | 0.545 | **0.598** | **0.842** |
| Logistic Regression | 0.801 | 0.647 | **0.553** | 0.597 | 0.841 |
| SVM | 0.796 | 0.657 | 0.481 | 0.556 | 0.798 |
| Random Forest | 0.780 | 0.609 | 0.479 | 0.536 | 0.821 |

### Modèle final : Gradient Boosting optimisé

```
Accuracy  : 80.6%
Precision : 66.2%   → sur 100 alertes churn, 66 sont réelles
Recall    : 54.5%   → détecte 54 clients sur 100 qui vont vraiment partir
F1-score  : 0.598   ← métrique de sélection principale
ROC-AUC   : 0.842   ← excellente capacité de discrimination
```

### Matrice de confusion

|  | Prédit : Reste | Prédit : Churn |
|---|---|---|
| **Réel : Reste** | ✅ 935 (vrais négatifs) | ⚠️ 100 (faux positifs) |
| **Réel : Churn** | ❌ 179 (faux négatifs) | ✅ 195 (vrais positifs) |

**Lecture métier :** un **faux négatif** (client qui part sans être détecté) est le cas le plus coûteux pour l'entreprise — il n'est pas contacté et se perd définitivement. Un **faux positif** (alerte inutile) coûte moins cher : au pire, une offre de fidélisation envoyée inutilement.

### Variables les plus influentes (Gradient Boosting)

| Rang | Variable | Importance | Interprétation |
|---|---|---|---|
| 1 | `tenure` | 0.290 | Plus un client est ancien, moins il résilie |
| 2 | `InternetService_Fiber optic` | 0.190 | Les abonnés fibre résiliaient plus souvent |
| 3 | `PaymentMethod_Electronic check` | 0.128 | Mode de paiement lié à un profil plus volatile |
| 4 | `Contract_Two year` | 0.079 | L'engagement long protège contre le churn |
| 5 | `TotalCharges` | 0.070 | Lié à l'ancienneté et la facturation cumulée |

---

## ⚙️ Installation et exécution

```bash
# 1. Cloner le dépôt
git clone https://github.com/Yassmine-Aicha/telco-customer-churn-ml.git
cd telco-customer-churn-ml

# 2. Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install pandas numpy matplotlib seaborn scikit-learn jupyter joblib

# 4. Lancer le notebook final
jupyter notebook notebook/notebook_final.ipynb
```

> **Le notebook final** (`notebook_final.ipynb`) ne nécessite qu'un seul fichier : `data/raw/Telco-Customer-Churn.csv`. Il reproduit l'intégralité de la démarche en une seule exécution — EDA, prétraitement, modélisation, optimisation, évaluation, sauvegarde.

---

## ✅ Conclusion

Ce projet illustre une démarche complète de Machine Learning End-to-End :

- **EDA** : identification des variables clés et du déséquilibre des classes
- **Prétraitement** : pipeline reproductible, sans data leakage
- **Modélisation** : comparaison rigoureuse de 4 algorithmes aux logiques différentes
- **Optimisation** : GridSearchCV avec validation croisée 5-fold
- **Évaluation** : métriques adaptées au déséquilibre (F1, ROC-AUC), interprétation métier

**Limites et perspectives :**
- Le déséquilibre des classes pourrait être traité plus finement avec du rééquilibrage (SMOTE, `class_weight`)
- D'autres modèles (XGBoost, LightGBM) ou du feature engineering (ratios, interactions entre variables) pourraient améliorer le F1-score
- Perspective bonus : déploiement d'une application **Streamlit** utilisant `best_model.pkl` et `preprocessor.pkl` pour prédire le churn en temps réel

---

**Auteure :** Yassmine Ben Aicha
**Module :** Machine Learning — M. Abdallah Khemais
**Établissement :** École Polytechnique de Tunisie
