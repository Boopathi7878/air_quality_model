import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load test data
test_data = pd.read_csv(r"D:\\Codes\\Projects\\ML\\air_quality_model\data\\raw\\processed.py")

# Prepare features and target for testing
X_test = test_data.drop(columns=["AQI"])
y_test = test_data["AQI"]

# Load trained models
rf_model = joblib.load(r"D:\\Codes\\Projects\\ML\\air_quality_model\\models\\random_forest_model.pkl")
nn_model = load_model(r"D:\\Codes\\Projects\\ML\\air_quality_model\\models\\neural_network_model.keras")

# evaluate Random Forest model
rf_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print(f"Random Forest - MSE: {rf_mse:.4f}, MAE: {rf_mae:.4f}, R²: {rf_r2:.4f}")

# evaluate Neural Network model
nn_pred = nn_model.predict(X_test)
nn_pred = nn_pred.ravel() if nn_pred.ndim > 1 else nn_pred.flatten()

nn_mse = mean_squared_error(y_test, nn_pred)
nn_mae = mean_absolute_error(y_test, nn_pred)
nn_r2 = r2_score(y_test, nn_pred)
print(f"Neural Network - MSE: {nn_mse:.4f}, MAE: {nn_mae:.4f}, R²: {nn_r2:.4f}")
