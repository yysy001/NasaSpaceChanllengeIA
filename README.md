# Temperature and NDVI Prediction with XGBoost and Prophet

This repository contains two Python scripts for predicting daytime temperature and NDVI values using **XGBoost** and **Prophet** models with an external regressor. The code includes loading, preprocessing, and analyzing geospatial data obtained from the **AppEEARS** (Application for Extracting and Exploring Analysis Ready Samples) application.

## Code Structure

The project is divided into two Python scripts:

1. **Temperature Prediction Script:**
   - Handles data related to daytime and nighttime temperatures from the Arequipa region.
   - **XGBoost** is used to predict daytime temperature based on the date, and **Prophet** generates future predictions using the **XGBoost** predictions as an external regressor.

2. **NDVI Prediction Script:**
   - Processes **NDVI** and **EVI** data for temporal analysis.
   - The script applies **XGBoost** and **Prophet** to predict future values of **NDVI** using an external regressor.

## Requirements

Install the following libraries to run the code:

```bash
pip install pandas matplotlib scikit-learn xgboost prophet
```

## Data Description

The data used in this project comes from the **AppEEARS** tool. This application allows access to and transformation of geospatial data from several federal data archives. With **AppEEARS**, users can extract specific data from geospatial datasets using spatial, temporal, and band/layer parameters.

For more information about **AppEEARS**, visit the [official AppEEARS site](https://appeears.earthdatacloud.nasa.gov/).

### Variables:

- **VJ121A2_002_LST_Day_1KM**: Daytime land surface temperature at 1 km resolution.
- **VJ121A2_002_LST_Night_1KM**: Nighttime land surface temperature at 1 km resolution.
- **VNP13A3_001__1_km_monthly_NDVI**: Monthly Normalized Difference Vegetation Index at 1 km resolution.
- **VNP13A3_001__1_km_monthly_EVI**: Monthly Enhanced Vegetation Index at 1 km resolution.

## Contribution to Chakra360

This code and the associated analysis and predictions aim to contribute to the objective of **Chakra360**, providing accurate predictions based on open data offered by **AppEEARS** to farmers in need of our services.


## Future Improvements

Future enhancements are planned to incorporate external regressor data to improve the predictive capabilities of the model. Additionally, predictions will be made in real-time using data collected from the **Parrot Bebop 2** drone.


