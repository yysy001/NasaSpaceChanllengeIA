import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.model_selection import train_test_split

# Cargar el dataset
data = pd.read_csv('ndvi2.csv')

# Filtrar las columnas necesarias
filtered_columns = [
    'Category',
    'ID',
    'Latitude',
    'Longitude',
    'Date',
    'VNP13A3_001__1_km_monthly_NDVI',
    'VNP13A3_001__1_km_monthly_EVI'
]

filtered_df = data[filtered_columns]

print(filtered_df.shape)

filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

filtered_df = filtered_df[(filtered_df['VNP13A3_001__1_km_monthly_NDVI'] != 0) & 
                          (filtered_df['VNP13A3_001__1_km_monthly_EVI'] != 0)]

plt.figure(figsize=(12, 6))
plt.plot(filtered_df['Date'], filtered_df['VNP13A3_001__1_km_monthly_NDVI'], label='NDVI', color='green')
plt.plot(filtered_df['Date'], filtered_df['VNP13A3_001__1_km_monthly_EVI'], label='EVI', color='blue')

plt.title('Tendencia Temporal de NDVI y EVI')
plt.xlabel('Fecha')
plt.ylabel('Valores de NDVI y EVI')
plt.legend()
plt.xticks(rotation=45)
plt.grid()

plt.tight_layout()
plt.show()

data = filtered_df.rename(columns={'Date': 'ds', 'VNP13A3_001__1_km_monthly_NDVI': 'y'})
data['month'] = data['ds'].dt.month
data['day'] = data['ds'].dt.day
data['year'] = data['ds'].dt.year

X = data[['month', 'day', 'year']]
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X_train, y_train)

data['xgboost_prediction'] = xgb_model.predict(X)

prophet_model = Prophet()

prophet_model.add_regressor('xgboost_prediction')

prophet_model.fit(data)

future = prophet_model.make_future_dataframe(periods=90)

future['month'] = future['ds'].dt.month
future['day'] = future['ds'].dt.day
future['year'] = future['ds'].dt.year

future['xgboost_prediction'] = xgb_model.predict(future[['month', 'day', 'year']])

forecast = prophet_model.predict(future)

fig = prophet_model.plot(forecast)
plt.title('Predicción de NDVI con Regresor Externo XGBoost')
plt.xlabel('Fecha')
plt.ylabel('NDVI')
plt.show()
