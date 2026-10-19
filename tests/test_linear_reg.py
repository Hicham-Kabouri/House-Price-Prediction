# predict test par regression lineaire
y_test_pred = model.predict(X_test_selected)

# evaluation de test 
rmse_test = np.sqrt(mean_squared_error(y_test , y_test_pred))  
r2_test = r2_score(y_test , y_test_pred)
print("===== TEST =====")
print("RMSE:", rmse_test)
print("R²  :", r2_test)