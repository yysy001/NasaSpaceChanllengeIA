import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.model_selection import train_test_split

data = pd.read_csv('')

filtered_columns = [
    'Category',
    'ID',
    'Latitude',
    'Longitude',
    'Date',
    'VJ121A2_002_LST_Day_1KM',
    'VJ121A2_002_LST_Night_1KM'
]

filtered_df = data[filtered_columns]

print(filtered_df.shape)

filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

filtered_df = filtered_df[(filtered_df['VJ121A2_002_LST_Day_1KM'] != 0) & 
                          (filtered_df['VJ121A2_002_LST_Night_1KM'] != 0)]

plt.figure(figsize=(12, 6))

plt.plot(filtered_df['Date'], filtered_df['VJ121A2_002_LST_Day_1KM'], label='Temperatura Diurna (K)', color='orange')
plt.plot(filtered_df['Date'], filtered_df['VJ121A2_002_LST_Night_1KM'], label='Temperatura Nocturna (K)', color='blue')

plt.title('Tendencia Temporal de Temperaturas en Arequipa')
plt.xlabel('Fecha')
plt.ylabel('Temperatura (K)')
plt.legend()
plt.xticks(rotation=45)
plt.grid()

plt.tight_layout()
plt.show()


# Preparar los datos para XGBoost
data = filtered_df.rename(columns={'Date': 'ds', 'VJ121A2_002_LST_Day_1KM': 'y'})
data['month'] = data['ds'].dt.month
data['day'] = data['ds'].dt.day
data['year'] = data['ds'].dt.year

X = data[['month', 'day', 'year']]
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

data['xgboost_prediction'] = xgb_model.predict(X)

prophet_model = Prophet()

prophet_model.add_regressor('xgboost_prediction')

prophet_model.fit(data)

future = prophet_model.make_future_dataframe(periods=30)

future['month'] = future['ds'].dt.month
future['day'] = future['ds'].dt.day
future['year'] = future['ds'].dt.year

future['xgboost_prediction'] = xgb_model.predict(future[['month', 'day', 'year']])

forecast = prophet_model.predict(future)

fig = prophet_model.plot(forecast)
plt.title('Predicción de Temperatura Diurna con Regresor Externo XGBoost')
plt.xlabel('Fecha')
plt.ylabel('Temperatura (K)')
plt.show()