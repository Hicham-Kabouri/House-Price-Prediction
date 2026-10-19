def fit_feature_engineering(x, y):

    x = x.copy()

    # --------------------------------------------------
    # 1. Variables numériques
    # --------------------------------------------------

    num_data = x.select_dtypes(
        include=np.number
    )

    # --------------------------------------------------
    # 2. Corrélation avec la cible
    # --------------------------------------------------

    corr = num_data.corrwith(y)

    corr_abs = corr.abs().sort_values(
        ascending=False
    )

    print("Corrélation avec la variable cible :")
    print(corr_abs)

    # --------------------------------------------------
    # 3. Sélection selon la corrélation avec y
    # --------------------------------------------------

    threshold = 0.3

    selected_features = corr_abs[
        corr_abs > threshold
    ].index.tolist()

    selected_data = num_data[
        selected_features
    ]

    # --------------------------------------------------
    # 4. Corrélation entre les features sélectionnées
    # --------------------------------------------------

    corr_matrix = selected_data.corr().abs()

    upper = corr_matrix.where(
        np.triu(
            np.ones(corr_matrix.shape),
            k=1
        ).astype(bool)
    )

    # --------------------------------------------------
    # 5. Features à supprimer si corrélation > 0.9
    # --------------------------------------------------

    to_drop = [
        column
        for column in upper.columns
        if any(upper[column] > 0.9)
    ]

    # --------------------------------------------------
    # 6. Features finales
    # --------------------------------------------------

    final_features = [
        column
        for column in selected_features
        if column not in to_drop
    ]

    # --------------------------------------------------
    # 7. Dataset final TRAIN
    # --------------------------------------------------

    x_selected = x[
        final_features
    ].copy()

    # --------------------------------------------------
    # 8. Stocker les décisions prises sur TRAIN
    # --------------------------------------------------

    feature_info = {
        'threshold': threshold,
        'selected_features': selected_features,
        'to_drop': to_drop,
        'final_features': final_features
    }

    print("\nVariables supprimées :", to_drop)

    print(
        "Nombre de variables finales :",
        len(final_features)
    )

    print("\nVariables finales :")
    print(final_features)

    return x_selected, y ,feature_info