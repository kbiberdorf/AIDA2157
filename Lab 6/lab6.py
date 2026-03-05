import pandas as pd
import re

# Load raw data
df = pd.read_csv('C:\\Users\\kelse\\ml-project\\Lab 6\\canada_weather.csv')

def clean_temp(val):
    if pd.isna(val): return None
    # Replace unicode minus (−) with standard hyphen (-)
    val = str(val).replace('−', '-')
    # Extract the first float (Celsius)
    match = re.search(r"[-+]?\d*\.\d+|\d+", val)
    return float(match.group()) if match else None

def clean_elevation(val):
    if pd.isna(val): return 0.0
    # Strip 'm', commas, and take the part before the metric unit
    val = str(val).split('m')[0].replace(',', '')
    try: return float(val)
    except: return 0.0

def extract_lat(val):
    # Extract decimal latitude from the coordinate string
    match = re.search(r"([-+]?\d*\.\d+);", val)
    return float(match.group(1)) if match else None

# Apply cleaning
df['AnnualAvgLow_C'] = df['Annual(Avg. low °C (°F))'].apply(clean_temp)
df['Elevation_m'] = df['Elevation'].apply(clean_elevation)
df['Latitude'] = df['Location'].apply(extract_lat)

# Define target: Is_Cold_Zone (Annual Low < 0)
df['Is_Cold_Zone'] = (df['AnnualAvgLow_C'] < 0).astype(int)

# Export for SQL Import
df_cleaned = df[['Community', 'Elevation_m', 'Latitude', 'AnnualAvgLow_C', 'Is_Cold_Zone']]
df_cleaned.to_csv('cleaned_weather_data.csv', index=False)
print("Step 1 Complete: Cleaned CSV generated.")