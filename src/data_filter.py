# faire une preprocessing prélimenaire 
def data_filter(data):
    data = data.copy()

    # 1. Traiter les absences réelles d'équipements
    colonnes_absence = [
        "Alley", "BsmtQual", "BsmtCond", "BsmtExposure",
        "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual",
        "GarageCond", "PoolQC", "Fence", "MiscFeature"
    ]

    cols_presentes = [
        c for c in colonnes_absence if c in data.columns
    ]

    data[cols_presentes] = (
        data[cols_presentes].fillna("None")
    )

    # 2. Créer les variables temporelles
    data = year_features(data)

    # 3. Transformer les variables asymétriques
    cols_log = [
        "MiscVal", "PoolArea", "LotArea", "3SsnPorch",
        "LowQualFinSF", "BsmtFinSF2", "ScreenPorch",
        "EnclosedPorch", "OpenPorchSF", "WoodDeckSF",
        "MasVnrArea", "LotFrontage", "BsmtFinSF1",
        "TotalBsmtSF", "1stFlrSF", "GrLivArea"
    ]

    cols_presentes = [
        c for c in cols_log if c in data.columns
    ]

    # Vérifier que les valeurs non manquantes sont >= 0
    for col in cols_presentes:
        valeurs = data[col].dropna()

        if (valeurs < 0).any():
            raise ValueError(
                f"Valeurs négatives détectées dans {col}"
            )

    data[cols_presentes] = np.log1p(
        data[cols_presentes]
    )

    # 4. MSSubClass est catégorielle
    if "MSSubClass" in data.columns:
        data["MSSubClass"] = (
            data["MSSubClass"].astype("string")
        )

    # 5. Cible en log, seulement si elle existe (absente du test Kaggle),
    #    puis placée à la fin
    if "SalePrice" in data.columns:
        if data["SalePrice"].max() > 100:             # garde-fou : pas de double log1p
            data["SalePrice"] = np.log1p(data["SalePrice"])
        data = data[
            [c for c in data.columns if c != "SalePrice"]
            + ["SalePrice"]
        ]

    return data