def transform_feature_engineering(x, feature_info ):
    x = x.copy()

    manquantes = [
        c for c in feature_info['final_features']
        if c not in x.columns
    ]

    if manquantes:
        raise KeyError(f"Colonnes absentes du jeu à transformer : {manquantes}")

    return x[feature_info['final_features']].copy()