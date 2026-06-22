# 📡 Prédiction du Churn Client — Telco Customer Churn

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Terminé-success)

Projet de Machine Learning réalisé dans le cadre du module **Machine Learning** (M. Abdallah Khemais).
Objectif : prédire si un client d'une entreprise de télécommunications va **résilier son abonnement (churn)**, à partir de ses caractéristiques (ancienneté, services souscrits, type de contrat, facturation...).

---

## 📑 Sommaire

- [Contexte et problématique](#-contexte-et-problématique)
- [Dataset](#-dataset)
- [Structure du projet](#-structure-du-projet)
- [Méthodologie](#-méthodologie)
- [Résultats](#-résultats)
- [Installation et exécution](#-installation-et-exécution)
- [Conclusion](#-conclusion)

---

## 🎯 Contexte et problématique

Une entreprise de télécommunications souhaite identifier à l'avance les clients susceptibles de résilier leur abonnement, afin de pouvoir agir en amont (offre de fidélisation, contact proactif). C'est un problème de **classification binaire supervisée** :

> À partir des caractéristiques d'un client, prédire s'il va churner (`1`) ou rester (`0`).

## 📊 Dataset

| | |
|---|---|
| **Source** | Dataset publié initialement par IBM, largement utilisé en recherche et formation (référencé sur Kaggle : *Telco Customer Churn*, par blastchar) |
| **Volume** | 7 043 clients, 21 variables (20 explicatives + 1 cible) |
| **Type de variables** | Mix numérique (ancienneté, facturation) et catégoriel (services, contrat, paiement) |
| **Justification (source externe)** | Dataset propre, taille suffisante, problématique métier claire et pertinente (rétention client), absent des sources locales tunisiennes recommandées en priorité |

## 📁 Structure du projet

```
mon-projet/
├── notebook/
│   ├── 01_EDA_draft.ipynb              # Analyse exploratoire
│   ├── 02_preprocessing_draft.ipynb    # Nettoyage, encodage, split
│   ├── 03_modeling_draft.ipynb         # Comparaison et optimisation des modèles
│   └── notebook_final.ipynb            # Version propre, démarche complète de A à Z
├── data/
│   ├── raw/                            # Données brutes
│   └── processed/                      # Données prétraitées (train/test)
├── models/
│   ├── preprocessor.pkl                # Pipeline de transformation sauvegardé
│   └── best_model.pkl                  # Modèle final entraîné
├── presentation/
│   └── presentation.pptx               # Support de soutenance
└── README.md
```

## 🔬 Méthodologie

```
EDA  →  Prétraitement  →  Comparaison de 4 modèles  →  Optimisation (GridSearchCV)  →  Évaluation  →  Sauvegarde
```

**1. Analyse exploratoire (EDA)**
Étude de la distribution de la cible, des variables numériques et catégorielles, des corrélations avec le churn.

**2. Prétraitement**
- Suppression de `customerID` (sans valeur prédictive)
- Correction de `TotalCharges` (11 valeurs manquantes, remplacées par 0 — clients avec `tenure = 0`, pas encore facturés)
- Encodage de la cible en binaire
- Vérification des outliers (méthode IQR) — aucun traitement nécessaire
- `StandardScaler` (numérique) + `OneHotEncoder` (catégoriel), pipeline **fit uniquement sur le train** pour éviter le data leakage
- Split train/test stratifié (80/20)

**3. Modélisation**
Comparaison de 4 algorithmes aux logiques différentes :
- **Logistic Regression** — modèle linéaire, simple et interprétable
- **Random Forest** — ensembliste, agrégation d'arbres de décision
- **Gradient Boosting** — ensembliste, correction progressive des erreurs
- **SVM** — recherche de la marge de séparation maximale

**4. Optimisation**
Le modèle ayant obtenu le meilleur F1-score est optimisé via `GridSearchCV` (validation croisée 5-fold).

## 📈 Résultats

| Modèle | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| **Gradient Boosting** ⭐ | 0.806 | 0.662 | 0.545 | **0.598** | 0.842 |
| Logistic Regression | 0.801 | 0.647 | 0.553 | 0.597 | 0.841 |
| SVM | 0.796 | 0.657 | 0.481 | 0.556 | 0.798 |
| Random Forest | 0.780 | 0.609 | 0.479 | 0.536 | 0.821 |

**Pourquoi le F1-score comme critère de sélection ?** Les classes sont déséquilibrées (~26.5% churn / ~73.5% non-churn). L'Accuracy seule serait trompeuse : un modèle qui prédirait toujours "non-churn" aurait déjà ~73% d'Accuracy sans rien apprendre. Le F1-score (moyenne harmonique de Precision et Recall) est plus représentatif ici.

**Modèle final retenu : Gradient Boosting**, optimisé par GridSearchCV (`learning_rate=0.05, max_depth=3, n_estimators=200`).

**Variables les plus influentes :** ancienneté (`tenure`), type de contrat (`Contract`) et charges mensuelles (`MonthlyCharges`) — cohérent avec l'intuition métier : un client récent, en engagement court, avec une facture élevée, est plus à risque de churn.

## ⚙️ Installation et exécution

```bash
git clone https://github.com/TON-PSEUDO/NOM-DU-REPO.git
cd NOM-DU-REPO

python3 -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

pip install pandas numpy matplotlib seaborn scikit-learn jupyter joblib

jupyter notebook
# ouvrir notebook/notebook_final.ipynb et exécuter toutes les cellules
```

Le notebook final ne nécessite que le fichier `data/raw/Telco-Customer-Churn.csv` pour reproduire l'intégralité de la démarche (EDA → prétraitement → modélisation → sauvegarde) en une seule exécution.

## ✅ Conclusion

Ce projet illustre une démarche complète de Machine Learning de bout en bout : compréhension du problème métier, exploration et nettoyage des données, comparaison rigoureuse de plusieurs modèles, optimisation et évaluation critique des résultats.

**Limites et perspectives :**
- Le déséquilibre des classes pourrait être traité plus finement (SMOTE, `class_weight`)
- D'autres modèles (XGBoost, LightGBM) ou du feature engineering supplémentaire pourraient encore améliorer le F1-score
- Bonus envisagé : déploiement via une application Streamlit utilisant `best_model.pkl` et `preprocessor.pkl`

---

**Auteur(s) :** Yassmine Ben Aicha
**Module :** Machine Learning — M. Abdallah Khemais
