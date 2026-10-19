
# ==========================================================
# AJOUT : appliquer sur le TEST ce qui a été appris sur le TRAIN
#         (mêmes étapes que fit_preprocessor, mais rien n'est recalculé)
# ==========================================================
 
def transform_data(df, transformations, target_col='SalePrice'):
 
    X = df.copy()
 
    # ==========================================================
    # 1. Retirer la cible si elle est présente, et Id
    #    (aucune ligne n'est retirée : GrLivArea > 4000 ne concerne que le train)
    # ==========================================================
 
    X = X.drop(columns=[target_col], errors='ignore')
    y = df[target_col] if target_col in df.columns else None
 
    if 'Id' in X.columns:
        X = X.drop(columns=['Id'])
 
    # ==========================================================
    # 2. Supprimer les mêmes colonnes que sur le train
    # ==========================================================
 
    X = X.drop(
        columns=transformations['colonnes_supprimees'],
        errors='ignore'
    )
 
    # ==========================================================
    # 3. Mêmes colonnes, dans le même ordre que sur le train
    #    (une colonne absente du test provoque une erreur claire)
    # ==========================================================
 
    colonnes_train = list(transformations['preprocessor'].feature_names_in_)
    X = X[colonnes_train]
 
    # ==========================================================
    # 6b. IQR : bornes du TRAIN (avant l'imputation, comme dans fit_preprocessor)
    # ==========================================================
 
    X = appliquer_bornes_iqr(X, transformations['bornes_iqr'])
 
    # ==========================================================
    # 7. Imputation des variables numériques : médianes du TRAIN
    # ==========================================================
 
    for col, mediane in transformations['medianes'].items():
        X[col] = X[col].fillna(mediane)
 
    # ==========================================================
    # 8. Imputation des variables catégorielles : modes du TRAIN
    # ==========================================================
 
    for col, mode in transformations['modes'].items():
        X[col] = X[col].fillna(mode)
 
    # ==========================================================
    # 11. Standardisation : moyenne et écart-type du TRAIN
    # ==========================================================
 
    colonnes_standardiser = transformations['colonnes_standardiser']
 
    X[colonnes_standardiser] = (
        transformations['scaler'].transform(
            X[colonnes_standardiser]
        )
    )
 
    # ==========================================================
    # 12. One-Hot Encoding : catégories du TRAIN
    #     (une catégorie inconnue donne des 0 partout, sans erreur)
    # ==========================================================
 
    X_array = transformations['preprocessor'].transform(X)
 
    # ==========================================================
    # 14. Reconstruire le DataFrame final
    # ==========================================================
 
    X_processed = pd.DataFrame(
        X_array,
        columns=transformations['final_columns'],
        index=X.index
    )
 
    return X_processed, y