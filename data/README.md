# AgriPredict AI - Dataset Documentation

## Overview
This dataset contains empirical agricultural crop yield observations across key agro-climatic zones, compiled using historical data from the **Directorate of Economics and Statistics (DES)**, **Department of Agriculture & Farmers Welfare (DA&FW)**, Government of India, cross-referenced with **ICRISAT (International Crops Research Institute for the Semi-Arid Tropics)** and **Copernicus Sentinel-2 NDVI** spatial vegetation datasets.

## Dataset Metadata
- **File Name:** `crop_yield_dataset.csv`
- **Number of Records:** 2,500+ agro-climatic observation rows
- **Primary Crops Covered:**
  - `Rice` (Oryza sativa)
  - `Wheat` (Triticum aestivum)
  - `Maize` (Zea mays)
  - `Cotton` (Gossypium hirsutum)
  - `Sugarcane` (Saccharum officinarum)
- **Geographic Coverage:** Top agricultural states including Punjab, Uttar Pradesh, Maharashtra, Karnataka, Haryana, Madhya Pradesh, Gujarat, Andhra Pradesh, Tamil Nadu, and West Bengal.
- **License:** Open Government Data (OGD) / CC-BY 4.0 Creative Commons Attribution

## Schema & Feature Dictionary

| Feature Name | Data Type | Units | Description | Typical Agronomic Range |
|:---|:---|:---|:---|:---|
| `crop_type` | Categorical | String | Cultivated crop variety | Rice, Wheat, Maize, Cotton, Sugarcane |
| `state` | Categorical | String | Indian State / Agro-climatic zone | Punjab, Uttar Pradesh, etc. |
| `district` | Categorical | String | District administrative division | Ludhiana, Varanasi, etc. |
| `season` | Categorical | String | Cropping season | Kharif, Rabi, Summer, Whole Year |
| `soil_type` | Categorical | String | Dominant soil classification | Alluvial, Black, Clay, Loamy, Red, Sandy |
| `temperature` | Numerical | °C | Average temperature during vegetative/reproductive stage | 14.0°C - 38.0°C |
| `rainfall` | Numerical | mm | Cumulative seasonal precipitation | 250 mm - 2,200 mm |
| `humidity` | Numerical | % | Mean relative air humidity | 35% - 92% |
| `soil_moisture` | Numerical | % | Volumetric soil moisture content | 12% - 55% |
| `ndvi` | Numerical | Normalized [-1 to 1] | Sentinel-2 / MODIS Normalized Difference Vegetation Index | 0.20 - 0.88 |
| `yield` (Target) | Numerical | Tonnes / Hectare | Measured agricultural harvest yield | Rice (1.8-6.2), Wheat (2.2-5.8), Maize (2.0-7.5), Cotton (0.8-3.2), Sugarcane (45.0-105.0) |

## Data Preprocessing Pipeline
1. **Missing Value Imputation:** Handled via median imputation for numerical features and most frequent strategy for categorical features.
2. **Feature Encoding:** `OneHotEncoder(handle_unknown='ignore')` applied to categorical features (`crop_type`, `state`, `season`, `soil_type`).
3. **Feature Scaling:** `StandardScaler` applied to continuous numerical environmental variables (`temperature`, `rainfall`, `humidity`, `soil_moisture`, `ndvi`).
4. **Target Variable:** Continuous `yield` (t/ha) predicted using `RandomForestRegressor`.

## Limitations & Considerations
- Yield estimates represent farm-gate potential under standard agronomic practices.
- Catastrophic events such as severe hailstorms, pest infestations (e.g. desert locusts), or extreme flash floods may cause local anomalies beyond the trained weather bounds.
- Satellite NDVI values require cloud-masking during the peak monsoon season.
