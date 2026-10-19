# ==========================================================
# SOUMISSION KAGGLE
# ==========================================================

# 2. Sauvegarder les Id avant toute transformation
ids = test_raw["Id"].copy()

# 3. Préparation préliminaire : la même fonction que pour le train
#    (équipements absents -> "None", âges, log1p, MSSubClass en texte)
test_kaggle = data_filter(test_raw)

# 4. Ce qui a été appris sur le train (rien n'est recalculé)
X_kaggle , _   = transform_data(test_kaggle, transformations)
X_kaggle_sel = transform_feature_engineering(X_kaggle, feature_info)

# 5. Prédiction (en log), puis retour en dollars
pred_log     = model_rf.predict(X_kaggle_sel)
pred_dollars = np.expm1(pred_log)

# 6. Fichier de soumission
submission = pd.DataFrame({"Id": ids, "SalePrice": pred_dollars})
submission.to_csv("data/raw/submission_rf.csv", index=False)

# 7. Vérifications avant d'envoyer
print(submission.shape)                                   # 1459 lignes pour Kaggle
print(submission["SalePrice"].describe())                 # de l'ordre de 50 000 à 500 000
assert submission["SalePrice"].notna().all()
assert len(submission) == len(test_raw)
assert submission["Id"].is_unique