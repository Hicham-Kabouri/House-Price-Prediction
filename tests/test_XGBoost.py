#evaluation 

y_test_pred_xgb = model_xgb.predict(X_test_selected)

# evaluation de test
rmse_test_xgb = np.sqrt(mean_squared_error(y_test , y_test_pred_xgb))
r2_test_xgb = r2_score(y_test , y_test_pred_xgb)
print("===== TEST XGBOOST =====")
print("RMSE:", rmse_test_xgb)
print("R²  :", r2_test_xgb)