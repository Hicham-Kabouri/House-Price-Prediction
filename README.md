# Prédiction du prix des maisons (Ames, Iowa)

Projet de régression : prédire le prix de vente de maisons (`SalePrice`) à partir de leurs
caractéristiques (surface, qualité, quartier, année de construction, garage, ...).

Données : compétition Kaggle [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
Métrique de la compétition : RMSE entre le log du prix prédit et le log du prix réel (RMSLE).

## Résultats

Évaluation sur un jeu de test local (découpage classique train / test de `train.csv`) puis sur Kaggle.
Le modèle est entraîné sur le log du prix : le RMSE local est donc calculé sur le log, comme la
métrique de Kaggle.

| Modèle | RMSE (log) sur le jeu de test local | Score Kaggle |
| --- | --- | --- |
| Random Forest | 0.1562 | 0.15641 |
| XGBoost | 0.165 | 0.16281 |

Le score Kaggle est le seul vrai score sur des données jamais vues : `test.csv` ne contient pas les prix.
Sur ce projet, Random Forest obtient un meilleur score Kaggle qu'XGBoost.

## Structure du projet

```
.
├── data/
│   └── raw/                  # train.csv, test.csv, sample_submission.csv (non versionnés)
├── notebooks/                # exploration et essais
├── models/                   # modèle utilisé
├── src/                      # préparation des données
├── tests/                    # évaluation du modèle sur le jeu de test
├── requirements.txt          # bibliothèques et versions
├── .gitignore
└── README.md
```

## Installation (Windows, VS Code)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Puis télécharger `train.csv` et `test.csv` depuis la page Kaggle et les placer dans `data/raw/`.

## Utilisation

Exécuter les notebooks du dossier `notebooks/` dans l'ordre : `à compléter (noms et ordre)`.
Le fichier de soumission contient les colonnes `Id` et `SalePrice` (prix en dollars), prêt à être
déposé sur Kaggle.

## Méthode

**Lecture des données.** `NA` signifie souvent « l'équipement n'existe pas » (pas de garage, de
piscine, de sous-sol). Le fichier est lu avec `keep_default_na=False, na_values=["NA", ""]` et ces
absences sont remplacées par la catégorie `"None"`.

**Variables.**
- Les années deviennent des âges à la date de vente (`House_Age`, `Remod_Age`, `Garage_Age`) avec
  des indicateurs (`IsNewHouse`, `WasRemodeled`, `Has_Garage`).
- Le mois de vente est codé de façon cyclique (`MoSold_sin`, `MoSold_cos`).
- `MSSubClass` est une catégorie (One-Hot), pas un nombre.
- `log(1 + x)` sur les variables asymétriques (surfaces) et sur la cible.

**Valeurs extrêmes.**
- Retrait de `GrLivArea > 4000` dans le jeu d'entraînement seulement.
- Plafonnement IQR × 3 des variables numériques, avec des bornes apprises sur l'entraînement.
  Les colonnes dont l'IQR vaut 0 ne sont pas touchées.

**Préparation.** Médiane (numériques) ou mode (catégorielles) pour les vraies valeurs manquantes,
standardisation, One-Hot, puis sélection des variables corrélées à la cible (> 0,3) en retirant
les doublons (corrélation > 0,9 entre variables).

**Modèles.** Régression linéaire, SVR, Random Forest et XGBoost, entraînés sur le log du prix. Les
prédictions sont ramenées en dollars avec `expm1`.

**Évaluation.** Découpage classique train / test (`train_test_split`) : le jeu de test n'est pas
utilisé pour l'entraînement. Pas de validation croisée à ce stade.

**Pas de fuite de données.** Tout ce qui est appris (bornes, médianes, modes, moyennes, catégories,
sélection de variables) l'est sur le jeu d'entraînement uniquement, puis appliqué tel quel au jeu
de test et au fichier `test.csv` de Kaggle, sans rien recalculer.

## Limites et pistes d'amélioration

- L'évaluation repose sur un seul découpage train / test : le score peut varier de quelques
  centièmes selon le découpage. Une validation croisée à 5 plis donnerait une estimation plus stable.
- Le prétraitement est écrit sous forme de fonctions. Le regrouper dans un `Pipeline` scikit-learn
  (avec `TransformedTargetRegressor` pour le log de la cible) éviterait les oublis d'étape et
  rendrait la validation croisée sans fuite de données.
- La sélection par corrélation retire des variables utiles aux modèles d'arbres : comparer avec et
  sans cette étape.
- Réglage des hyperparamètres (`RandomizedSearchCV`), puis comparaison avec Ridge et Gradient Boosting.
- Analyse des erreurs : sur quelles maisons (quartier, tranche de prix) le modèle se trompe-t-il le plus ?
- Les données couvrent une seule ville et les années 2006 à 2010 : le modèle ne se généralise pas
  tel quel à un autre marché ou à une autre époque.

## Reproductibilité

Graine fixée à `42` (découpage et modèles). Versions des bibliothèques dans `requirements.txt`
(à générer avec `pip freeze > requirements.txt` après l'installation).

## Données et licence

Les données appartiennent à Kaggle et à leurs auteurs (Dean De Cock, jeu de données Ames Housing).
Elles ne sont pas incluses dans ce dépôt : voir les règles de la compétition.