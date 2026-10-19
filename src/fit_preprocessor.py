# fixer les parametre qu'on va utilisé 
MAX_GRLIVAREA = 4000
FACTEUR_IQR = 3   # 3 au lieu de 1,5 : sur des variables asymétriques, 1,5 en signale trop

# Colonnes numériques à ne jamais plafonner : identifiant, cible, codes de catégories
COLONNES_EXCLUES = ("Id", "SalePrice", "MSSubClass", "MoSold")


def retirer_grandes_maisons(train):
    """Retire les maisons avec GrLivArea > 4000 (à appliquer au TRAIN seulement).

    Si GrLivArea est déjà en log, le seuil est converti en log aussi.
    """
    if "GrLivArea" not in train.columns:
        return train

    en_log = train["GrLivArea"].max() < 100          # en vérifie si le variable est déja transformer en log1p ou non 
    seuil = np.log1p(MAX_GRLIVAREA) if en_log else MAX_GRLIVAREA # si en log1p en utilise seuil 100 si non en utilise 400

    train_propre = train[train["GrLivArea"] <= seuil]
    print(f"GrLivArea > {MAX_GRLIVAREA} : {len(train) - len(train_propre)} maison(s) retirée(s).")
    return train_propre


def apprendre_bornes_iqr(train, facteur=FACTEUR_IQR, colonnes_exclues=COLONNES_EXCLUES):
    """Calcule les bornes IQR de chaque variable numérique du TRAIN.

    Renvoie un dictionnaire {colonne: (borne_basse, borne_haute)}.
    Les colonnes dont l'IQR vaut 0 n'y figurent pas : on ne fait rien dessus.
    """
    bornes = {}
    ignorees = []

    for col in train.select_dtypes(include="number").columns:
        if col in colonnes_exclues:
            continue

        q1 = train[col].quantile(0.25)
        q3 = train[col].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            ignorees.append(col)          # bornes nulles : rien à faire
        else:
            bornes[col] = (q1 - facteur * iqr, q3 + facteur * iqr)

    print(f"IQR : {len(bornes)} variable(s) avec bornes ; {len(ignorees)} ignorée(s) (IQR = 0) : {ignorees}")
    return bornes


def appliquer_bornes_iqr(df, bornes):
    """Ramène à la borne les valeurs hors bornes. Les NaN restent des NaN."""
    df = df.copy()

    for col, (bas, haut) in bornes.items():
        if col not in df.columns:
            continue
        n_hors_bornes = int(((df[col] < bas) | (df[col] > haut)).sum())
        if n_hors_bornes > 0:
            print(f"  {col} : {n_hors_bornes} valeur(s) ramenée(s) dans [{bas:.2f} ; {haut:.2f}]")
        df[col] = df[col].clip(bas, haut) # La méthode clip() ramène les valeurs en dehors de l'intervalle vers la borne la plus proche.

    return df


# data preprocessing 


def fit_preprocessor(df, target_col='SalePrice'):

    df = df.copy()

    # ==========================================================
    # 0. AJOUT : retirer GrLivArea > 4000 (dans le TRAIN seulement)
    # ==========================================================

    df = retirer_grandes_maisons(df)

    # ==========================================================
    # 1. Séparer immédiatement X et y de TRAIN
    # ==========================================================

    y_train = df[target_col].copy()

    X_train = df.drop(
        columns=[target_col]
    ).copy()
    # ==========================================================
    # 2. Supprimer Id
    # ==========================================================

    if 'Id' in X_train.columns:
        X_train = X_train.drop(columns=['Id'])

    # ==========================================================
    # 3. Pourcentage de valeurs manquantes
    # ==========================================================

    nan_percentage = X_train.isna().mean() * 100

    print("Pourcentage de valeurs manquantes :")
    print(
        nan_percentage[nan_percentage > 0]
        .sort_values(ascending=False)
    )

    # ==========================================================
    # 4. Colonnes avec au moins 90 % de valeurs manquantes
    # ==========================================================

    colonnes_supprimees = nan_percentage[
        nan_percentage >= 90
    ].index.tolist()

    X_train = X_train.drop(
        columns=colonnes_supprimees
    )

    print(
        "Nombre de colonnes supprimées :",
        len(colonnes_supprimees)
    )

    # ==========================================================
    # 6. Identifier les variables numériques
    #    et catégorielles
    # ==========================================================

    variables_numeriques = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    # "toutes les colonnes non numériques" (avec pandas 3,
    # include=["object", "category"] ne détecte plus les colonnes de texte)
    variables_categorielles = [
        col for col in X_train.columns
        if col not in variables_numeriques
    ]

    # ==========================================================
    # 6b. AJOUT : IQR x 3 sur les variables numériques
    #     (avant l'imputation : les NaN sont ignorés dans le calcul des quartiles)
    #     Bornes apprises sur le TRAIN ; IQR = 0 -> on ne fait rien.
    # ==========================================================

    bornes_iqr = apprendre_bornes_iqr(X_train)

    colonnes_outliers = list(bornes_iqr.keys())

    X_train = appliquer_bornes_iqr(X_train, bornes_iqr)

    # ==========================================================
    # 7. Imputation initiale des variables numériques
    #    avec la médiane du TRAIN
    # ==========================================================

    medianes = {}

    for col in variables_numeriques:

        medianes[col] = X_train[col].median()

        X_train[col] = X_train[col].fillna(
            medianes[col]
        )

    # ==========================================================
    # 8. Imputation initiale des variables catégorielles
    #    avec le mode du TRAIN
    # ==========================================================

    modes = {}

    for col in variables_categorielles:

        mode = X_train[col].mode()

        if not mode.empty:

            modes[col] = mode.iloc[0]

            X_train[col] = X_train[col].fillna(
                modes[col]
            )

    # ==========================================================
    # 11. Standardisation
    # ==========================================================

    colonnes_standardiser = [
        col
        for col in variables_numeriques
    ]

    scaler = StandardScaler()

    X_train[colonnes_standardiser] = (
        scaler.fit_transform(
            X_train[colonnes_standardiser]
        )
    )



    # ==========================================================
    # 12. One-Hot Encoding
    # ==========================================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'onehot',
                OneHotEncoder(
                    sparse_output=False,
                    handle_unknown='ignore'
                ),
                variables_categorielles
            )
        ],
        remainder='passthrough'
    )

    # Apprentissage + transformation sur TRAIN
    X_array = preprocessor.fit_transform(X_train)

    # ==========================================================
    # 13. Récupérer les noms des colonnes finales
    # ==========================================================

    cat_encoded_names = (
        preprocessor
        .named_transformers_['onehot']
        .get_feature_names_out(
            variables_categorielles
        )
    )

    # Colonnes numériques/celles laissées telles quelles
    autres_colonnes = [
        col
        for col in X_train.columns
        if col not in variables_categorielles
    ]

    final_columns = (
        list(cat_encoded_names)
        +
        autres_colonnes
    )

    # ==========================================================
    # 14. Reconstruire le DataFrame final
    # ==========================================================

    X_processed = pd.DataFrame(
        X_array,
        columns=final_columns,
        index=X_train.index
    )

    # ==========================================================
    # 15. Stocker toutes les transformations apprises
    # ==========================================================

    transformations = {

        # Colonnes supprimées
        'colonnes_supprimees':
            colonnes_supprimees,

        # Types de variables
        'variables_numeriques':
            variables_numeriques,

        'variables_categorielles':
            variables_categorielles,

        # Imputation
        'medianes':
            medianes,

        'modes':
            modes,

        # Outliers
        'colonnes_outliers':
            colonnes_outliers,

        'bornes_iqr':
            bornes_iqr,


        # Standardisation
        'colonnes_standardiser':
            colonnes_standardiser,

        'scaler':
            scaler,

        # One-Hot Encoding
        'preprocessor':
            preprocessor,

        # Colonnes finales
        'final_columns':
            final_columns
    }

    return X_processed, y_train, transformations