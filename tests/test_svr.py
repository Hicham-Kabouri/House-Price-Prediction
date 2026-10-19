# evaluation de test 
y_test_pred_svr = model_svr.predict(X_test_selected)

mse_svr_test = mean_squared_error(y_test, y_test_pred_svr)
r2_svr_test = r2_score( y_test, y_test_pred_svr)
print('=======train=======')
print("RMSE", mse_svr_test)
print('R²', r2_svr_test)