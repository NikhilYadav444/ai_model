# backend/generate_dataset.py
import pandas as pd
import numpy as np
import random
import os

print("🌱 GENERATING SYNTHETIC SOIL-CROP DATASET...")

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Reproducibility
np.random.seed(42)
random.seed(42)

# Define possible crops and regions
crops = ['Wheat', 'Rice', 'Cotton', 'Bajra', 'Sugarcane']
regions = ['Punjab', 'Haryana']

# Ideal crop condition ranges (approximate, realistic for India)
crop_conditions = {
    'Wheat':     {'N': (60, 120), 'P': (30, 60), 'K': (70, 120), 'Temp': (15, 25), 'Humidity': (40, 70), 'Moisture': (20, 35)},
    'Rice':      {'N': (80, 150), 'P': (40, 80), 'K': (100, 180), 'Temp': (20, 35), 'Humidity': (60, 90), 'Moisture': (30, 50)},
    'Cotton':    {'N': (50, 100), 'P': (25, 55), 'K': (80, 140), 'Temp': (25, 40), 'Humidity': (30, 60), 'Moisture': (15, 30)},
    'Bajra':     {'N': (30, 80), 'P': (15, 40), 'K': (60, 110), 'Temp': (28, 45), 'Humidity': (20, 55), 'Moisture': (10, 25)},
    'Sugarcane': {'N': (100, 200), 'P': (50, 100), 'K': (150, 250), 'Temp': (20, 35), 'Humidity': (50, 85), 'Moisture': (25, 45)}
}

# Generate synthetic data
rows = []
for _ in range(10000):
    crop = random.choice(crops)
    region = random.choice(regions)
    cond = crop_conditions[crop]

    N = np.clip(np.random.normal(np.mean(cond['N']), (cond['N'][1]-cond['N'][0])/5), 10, 250)
    P = np.clip(np.random.normal(np.mean(cond['P']), (cond['P'][1]-cond['P'][0])/5), 5, 150)
    K = np.clip(np.random.normal(np.mean(cond['K']), (cond['K'][1]-cond['K'][0])/5), 30, 300)
    Temp = np.clip(np.random.normal(np.mean(cond['Temp']), (cond['Temp'][1]-cond['Temp'][0])/6), 10, 45)
    Humidity = np.clip(np.random.normal(np.mean(cond['Humidity']), (cond['Humidity'][1]-cond['Humidity'][0])/6), 20, 90)
    Moisture = np.clip(np.random.normal(np.mean(cond['Moisture']), (cond['Moisture'][1]-cond['Moisture'][0])/6), 5, 50)

    rows.append([region, N, P, K, Temp, Humidity, Moisture, crop])

# Create DataFrame
columns = ['Region', 'Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Soil_Moisture', 'Crop']
df = pd.DataFrame(rows, columns=columns)

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV
df.to_csv("data/soil_crop_dataset.csv", index=False)

print("✅ Dataset generated successfully!")
print(f"📊 Total Records: {len(df)}")
print(f"📁 Saved to: data/soil_crop_dataset.csv")
print("\n🌱 Crop Distribution:")
print(df['Crop'].value_counts())
print(f"\n📈 Dataset Preview:")
print(df.head())