# test prediction avec random forest 
y_test_pred_rf = model_rf.predict(X_test_selected)

# evaluation de test
rmse_test_rf = np.sqrt(mean_squared_error(y_test , y_test_pred_rf))
r2_test_rf = r2_score(y_test , y_test_pred_rf)
print("===== TEST RANDOM FOREST =====")
print("RMSE:", rmse_test_rf)
print("R²  :", r2_test_rf)