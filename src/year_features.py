
def year_features(df):
    df = df.copy()

    # 1. Age de la maison au moment de la vente
    df["House_Age"] = df["YrSold"] - df["YearBuilt"]

    # 2. Age depuis la dernière rénovation
    df["Remod_Age"] = df["YrSold"] - df["YearRemodAdd"]

    # 3. Indiquer si la maison est neuve
    df["IsNewHouse"] = (df["YrSold"] == df["YearBuilt"]).astype(int)

    # 4. Indiquer si la maison a été rénovée
    df["WasRemodeled"] = (
        df["YearRemodAdd"] != df["YearBuilt"]
    ).astype(int)

    # 5. Présence d'un garage
    df["Has_Garage"] = (
        df["GarageType"].notna()
        & (df["GarageType"] != "None")
    ).astype(int)

    # 6. Age du garage
    
    df["Garage_Age"] = df["YrSold"] - df["GarageYrBlt"]

    # Si la maison n'a pas de garage, on conserve NaN.
    df.loc[df["Has_Garage"] == 0, "Garage_Age"] = np.nan

    # 7. Encodage cyclique du mois de vente
    angle = 2 * np.pi * (df["MoSold"] - 1) / 12

    df["MoSold_sin"] = np.sin(angle)
    df["MoSold_cos"] = np.cos(angle)


    # 8. Supprimer les variables devenues inutiles
    df = df.drop(columns=[
        "YearBuilt",
        "YearRemodAdd",
        "GarageYrBlt",
        "MoSold"
    ])

    return df