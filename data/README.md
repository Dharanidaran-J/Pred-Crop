# AgriPredict AI - Comprehensive Agricultural Dataset Documentation

## Overview
This dataset contains **10,000 agro-climatic and soil chemistry observation records** spanning 6 major staple and commercial crops. It integrates ground macronutrients (N-P-K), soil physicochemical parameters (pH, organic carbon, moisture), atmospheric weather, topographic altitude, and farm management practices (irrigation, fertilizer, pesticide).

## Dataset Metadata
- **File Name:** `crop_yield_dataset.csv`
- **Total Records:** 10,000 complete observations (0 missing values)
- **Features:** 19 independent predictive variables + 1 target variable
- **Primary Crops Covered:**
  - `Rice` (Oryza sativa)
  - `Wheat` (Triticum aestivum)
  - `Maize` (Zea mays)
  - `Cotton` (Gossypium hirsutum)
  - `Sugarcane` (Saccharum officinarum)
  - `Potato` (Solanum tuberosum)
- **Geographic Regions:** North, South, East, West, Central

## Feature Dictionary & Agronomic Ranges

| Feature Name | Category | Data Type | Units / Values | Description |
|:---|:---|:---|:---|:---|
| `Crop_Type` | Categorical | String | Rice, Wheat, Potato, Cotton, Maize, Sugarcane | Target crop species |
| `Region` | Categorical | String | North, South, East, West, Central | Macro agro-ecological zone |
| `Season` | Categorical | String | Kharif, Rabi, Zaid | Cropping cultivation season |
| `Soil_Type` | Categorical | String | Loamy, Sandy, Silt, Clay | Dominant soil textural class |
| `Irrigation_Type` | Categorical | String | Drip, Canal, Rainfed, Sprinkler | Water delivery method |
| `N` | Soil Nutrient | Integer | kg/ha (20 - 180) | Available soil Nitrogen |
| `P` | Soil Nutrient | Integer | kg/ha (10 - 100) | Available soil Phosphorus |
| `K` | Soil Nutrient | Integer | kg/ha (15 - 150) | Available soil Potassium |
| `Soil_pH` | Soil Property | Float | pH scale (5.0 - 8.5) | Soil acidity / alkalinity |
| `Soil_Moisture` | Soil Property | Float | % (15% - 65%) | Volumetric water content |
| `Organic_Carbon` | Soil Property | Float | % (0.2% - 1.5%) | Soil organic matter fraction |
| `Temperature` | Weather | Float | °C (10.0°C - 40.0°C) | Mean ambient growth temperature |
| `Humidity` | Weather | Float | % (30% - 90%) | Relative atmospheric humidity |
| `Rainfall` | Weather | Float | mm (200 mm - 2500 mm) | Cumulative precipitation |
| `Sunlight_Hours` | Weather | Float | Hours/day (4 - 12 hrs) | Daily photoperiod duration |
| `Wind_Speed` | Weather | Float | km/h (2 - 25 km/h) | Average wind velocity |
| `Altitude` | Topography | Integer | Meters (50 - 1200 m) | Field elevation above sea level |
| `Fertilizer_Used` | Farm Practice | Float | kg/ha (50 - 350 kg/ha) | Total chemical/bio-fertilizer applied |
| `Pesticide_Used` | Farm Practice | Float | kg/ha (1 - 30 kg/ha) | Plant protection chemical load |
| `Crop_Yield_ton_per_hectare` | **Target** | Float | Tonnes / Hectare | Measured agricultural harvest yield |

## Crop Yield Distribution (Tonnes / Hectare)

| Crop Type | Sample Count | Mean Yield | Min Yield | Max Yield | Standard Deviation |
|:---|:---|:---|:---|:---|:---|
| **Cotton** | 1,666 | 7.43 t/ha | 1.76 | 12.86 | 1.73 |
| **Wheat** | 1,671 | 8.72 t/ha | 3.78 | 14.09 | 1.65 |
| **Rice** | 1,702 | 9.40 t/ha | 4.07 | 14.20 | 1.66 |
| **Maize** | 1,658 | 10.08 t/ha | 5.48 | 15.60 | 1.70 |
| **Potato** | 1,670 | 24.89 t/ha | 20.32 | 30.27 | 1.68 |
| **Sugarcane** | 1,633 | 74.83 t/ha | 69.69 | 80.99 | 1.64 |
