# Imports
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import joblib
import matplotlib.pyplot as plt

# -----------------------
# 1. Train models (simulate data with 6 features now)
# -----------------------
np.random.seed(42)

# Simulated features: temp, humidity, co2, no2, pm25, pm10
X = np.random.rand(1000, 6) * [50, 100, 10000, 1000, 500, 500]  
y = np.random.rand(1000) * 500  # Simulated AQI target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network model
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

nn_model.compile(optimizer='adam', loss='mse', metrics=['mse'])
nn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)

# Save NN model
nn_model.save('/content/nn_model.h5')

# Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save RF model
joblib.dump(rf_model, '/content/rf_model.pkl')

# -----------------------
# 2. Load models (with custom_objects fix for mse)
# -----------------------
nn_model_loaded = load_model('/content/nn_model.h5', custom_objects={'mse': 'mse'})
rf_model_loaded = joblib.load('/content/rf_model.pkl')

# -----------------------
# 3. User input for prediction
# -----------------------
print("Enter Environmental Data for AQI Prediction:")

temp = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
co2_level = float(input("CO2 Level (ppm): "))
no2_level = float(input("NO2 Level (ppm): "))
pm25_level = float(input("PM2.5 Level (µg/m³): "))
pm10_level = float(input("PM10 Level (µg/m³): "))

input_data = np.array([[temp, humidity, co2_level, no2_level, pm25_level, pm10_level]])

# Get predictions
nn_preds = nn_model_loaded.predict(input_data)
rf_preds = rf_model_loaded.predict(input_data)

# Display predictions
print(f"\nNeural Network Prediction: {nn_preds[0][0]:.2f} AQI")
print(f"Random Forest Prediction: {rf_preds[0]:.2f} AQI")

# -----------------------
# 4. Visualization
# -----------------------
labels = ['Neural Network', 'Random Forest']
values = [nn_preds[0][0], rf_preds[0]]

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.bar(labels, values, color=['skyblue', 'orange'])
plt.ylabel('Predicted AQI')
plt.title('AQI Prediction Comparison')
plt.show()
