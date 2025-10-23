# train_models.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("🚀 STARTING BULLETPROOF MODEL TRAINING")
print("=" * 60)

# Create models directory
os.makedirs('models', exist_ok=True)

# ==================== LOAD AND PREPARE DATA ====================
print("\n📊 LOADING DATASET...")

try:
    # Load your generated dataset
    df = pd.read_csv('data/soil_crop_dataset.csv')
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Display dataset info
    print(f"\n📈 DATASET OVERVIEW:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Features: {len(df.columns)}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check data types
    print(f"\n🔧 DATA TYPES:")
    print(df.dtypes)
    
    # Check crop distribution
    print(f"\n🌱 CROP DISTRIBUTION:")
    crop_counts = df['Crop'].value_counts()
    for crop, count in crop_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {crop}: {count} samples ({percentage:.1f}%)")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("Please make sure 'data/soil_crop_dataset.csv' exists")
    exit()

# ==================== DATA CLEANING & VALIDATION ====================
print("\n🧹 DATA CLEANING & VALIDATION...")

# Remove duplicates
initial_count = len(df)
df = df.drop_duplicates()
print(f"   Removed {initial_count - len(df)} duplicate rows")

# Check for missing values
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print(f"   Missing values found:")
    for col, missing in missing_values.items():
        if missing > 0:
            print(f"     {col}: {missing} missing values")
    df = df.dropna()
else:
    print("   ✅ No missing values found")

# Validate value ranges
print(f"\n🔍 VALUE RANGE VALIDATION:")
numeric_columns = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Soil_Moisture']
for col in numeric_columns:
    min_val = df[col].min()
    max_val = df[col].max()
    print(f"   {col}: {min_val:.1f} to {max_val:.1f}")

# Remove extreme outliers (beyond 3 standard deviations)
print(f"\n📊 REMOVING OUTLIERS...")
initial_shape = df.shape
for col in numeric_columns:
    mean = df[col].mean()
    std = df[col].std()
    df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]

print(f"   Removed {initial_shape[0] - df.shape[0]} outlier rows")
print(f"   Final dataset shape: {df.shape}")

# ==================== EXPLORATORY DATA ANALYSIS ====================
print("\n📈 EXPLORATORY DATA ANALYSIS...")

# Correlation matrix
print("   Calculating correlations...")
correlation_matrix = df[numeric_columns].corr()

# Summary statistics
print(f"\n📊 SUMMARY STATISTICS:")
print(df[numeric_columns].describe())

# ==================== FEATURE ENGINEERING ====================
print("\n🔧 FEATURE ENGINEERING...")

# Create additional features that might help models
df['NPK_Total'] = df['Nitrogen'] + df['Phosphorus'] + df['Potassium']
df['NP_Ratio'] = df['Nitrogen'] / (df['Phosphorus'] + 1e-5)  # Avoid division by zero
df['NK_Ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-5)

print(f"   Created new features: NPK_Total, NP_Ratio, NK_Ratio")
print(f"   New dataset shape: {df.shape}")

# ==================== PREPARE FEATURES FOR MODELS ====================
print("\n🎯 PREPARING FEATURES FOR TRAINING...")

# For Soil Moisture Prediction (Regression)
# Using basic features + engineered features
moisture_features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 
                    'NPK_Total', 'NP_Ratio', 'NK_Ratio']
X_moisture = df[moisture_features]
y_moisture = df['Soil_Moisture']

# For Crop Prediction (Classification)  
# Using all available features including soil moisture
crop_features = moisture_features + ['Soil_Moisture']
X_crop = df[crop_features]
y_crop = df['Crop']

print(f"   Moisture prediction - Features: {X_moisture.shape}")
print(f"   Crop prediction - Features: {X_crop.shape}")
print(f"   Moisture features: {moisture_features}")
print(f"   Crop features: {crop_features}")

# ==================== SPLIT DATA ====================
print("\n📊 SPLITTING DATA...")

# For moisture prediction
X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
    X_moisture, y_moisture, test_size=0.2, random_state=42, stratify=None
)

# For crop prediction
X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(
    X_crop, y_crop, test_size=0.2, random_state=42, stratify=y_crop
)

print(f"   Moisture - Train: {X_m_train.shape}, Test: {X_m_test.shape}")
print(f"   Crop - Train: {X_c_train.shape}, Test: {X_c_test.shape}")

# ==================== FEATURE SCALING ====================
print("\n⚖️ SCALING FEATURES...")

# Scale features for moisture prediction
scaler_reg = StandardScaler()
X_m_train_scaled = scaler_reg.fit_transform(X_m_train)
X_m_test_scaled = scaler_reg.transform(X_m_test)

# Scale features for crop prediction  
scaler_clf = StandardScaler()
X_c_train_scaled = scaler_clf.fit_transform(X_c_train)
X_c_test_scaled = scaler_clf.transform(X_c_test)

print("✅ Feature scaling completed")

# ==================== LABEL ENCODING ====================
print("\n🔤 ENCODING LABELS...")

le = LabelEncoder()
y_c_train_encoded = le.fit_transform(y_c_train)
y_c_test_encoded = le.transform(y_c_test)

print(f"   Crop classes: {list(le.classes_)}")
print(f"   Class mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ==================== MODEL TRAINING ====================
print("\n🤖 TRAINING MODELS...")

# Train Soil Moisture Model (Regression)
print("💧 TRAINING SOIL MOISTURE MODEL...")
moisture_model = RandomForestRegressor(
    n_estimators=200,        # More trees for better performance
    max_depth=15,            # Deeper trees for complex patterns
    min_samples_split=5,     # Prevent overfitting
    min_samples_leaf=2,      # Prevent overfitting
    max_features='sqrt',     # Better generalization
    random_state=42,
    n_jobs=-1,               # Use all CPU cores
    verbose=1
)
moisture_model.fit(X_m_train_scaled, y_m_train)
print("✅ Soil Moisture Model trained")

# Train Crop Classification Model
print("🌱 TRAINING CROP CLASSIFICATION MODEL...")
crop_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=1
)
crop_model.fit(X_c_train_scaled, y_c_train_encoded)
print("✅ Crop Classification Model trained")

# ==================== MODEL EVALUATION ====================
print("\n📈 EVALUATING MODEL PERFORMANCE...")

# Evaluate Moisture Model
y_m_pred = moisture_model.predict(X_m_test_scaled)
mse = mean_absolute_error(y_m_test, y_m_pred)
moisture_std = np.std(y_m_test)

print(f"💧 SOIL MOISTURE MODEL PERFORMANCE:")
print(f"   Mean Absolute Error: {mse:.2f}%")
print(f"   Test Data Range: {y_m_test.min():.1f}% to {y_m_test.max():.1f}%")
print(f"   Test Data Std: {moisture_std:.2f}%")
print(f"   Error/Std Ratio: {mse/moisture_std:.2f}")

# Show some actual vs predicted examples
print(f"\n   SAMPLE PREDICTIONS (Actual vs Predicted):")
for i in range(5):
    print(f"     {y_m_test.iloc[i]:.1f}% vs {y_m_pred[i]:.1f}% (Error: {abs(y_m_test.iloc[i] - y_m_pred[i]):.1f}%)")

# Evaluate Crop Model
y_c_pred = crop_model.predict(X_c_test_scaled)
accuracy = accuracy_score(y_c_test_encoded, y_c_pred)

print(f"\n🌱 CROP MODEL PERFORMANCE:")
print(f"   Overall Accuracy: {accuracy:.2%}")
print(f"   Correct Predictions: {np.sum(y_c_test_encoded == y_c_pred)}/{len(y_c_test)}")

# Detailed classification report
print(f"\n   DETAILED CLASSIFICATION REPORT:")
print(classification_report(y_c_test_encoded, y_c_pred, target_names=le.classes_))

# Per-class accuracy
print(f"   PER-CLASS ACCURACY:")
y_c_test_decoded = le.inverse_transform(y_c_test_encoded)
y_c_pred_decoded = le.inverse_transform(y_c_pred)

for crop in le.classes_:
    mask = y_c_test_decoded == crop
    if mask.sum() > 0:
        crop_accuracy = (y_c_pred_decoded[mask] == crop).mean()
        print(f"     {crop}: {crop_accuracy:.2%} ({mask.sum()} samples)")

# ==================== FEATURE IMPORTANCE ====================
print("\n🔍 FEATURE IMPORTANCE ANALYSIS...")

# Moisture model feature importance
print("💧 SOIL MOISTURE FEATURE IMPORTANCE:")
moisture_importance = pd.DataFrame({
    'feature': moisture_features,
    'importance': moisture_model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in moisture_importance.iterrows():
    print(f"   {row['feature']}: {row['importance']:.3f}")

# Crop model feature importance
print("\n🌱 CROP PREDICTION FEATURE IMPORTANCE:")
crop_importance = pd.DataFrame({
    'feature': crop_features,
    'importance': crop_model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in crop_importance.iterrows():
    print(f"   {row['feature']}: {row['importance']:.3f}")

# ==================== SAVE MODELS ====================
print("\n💾 SAVING MODELS AND ARTIFACTS...")

# Save models
joblib.dump(moisture_model, 'models/soil_moisture_model.pkl')
joblib.dump(crop_model, 'models/crop_classifier_model.pkl')
joblib.dump(scaler_reg, 'models/scaler_reg.pkl')
joblib.dump(scaler_clf, 'models/scaler_clf.pkl')
joblib.dump(le, 'models/label_encoder.pkl')

# Save feature lists for reference
joblib.dump(moisture_features, 'models/moisture_features.pkl')
joblib.dump(crop_features, 'models/crop_features.pkl')

print("✅ All models and artifacts saved to 'models/' directory")

# ==================== COMPREHENSIVE TESTING ====================
print("\n🧪 COMPREHENSIVE MODEL TESTING...")

# Test with realistic scenarios
test_scenarios = [
    # (N, P, K, Temp, Humidity, Expected_Crop)
    (80, 45, 120, 22, 65, 'Wheat'),      # Cool weather, moderate nutrients
    (120, 60, 160, 28, 75, 'Rice'),      # High nutrients, warm & humid
    (70, 35, 100, 32, 45, 'Cotton'),     # Warm, dry, moderate nutrients
    (50, 25, 80, 35, 40, 'Bajra'),       # Hot, dry, low nutrients
    (150, 80, 200, 28, 70, 'Sugarcane'), # Very high nutrients
]

print("📝 REAL-WORLD TEST SCENARIOS:")
print("   N    P    K   Temp Humid | Pred_Moisture | Pred_Crop | Expected")
print("   " + "-" * 55)

for scenario in test_scenarios:
    N, P, K, Temp, Humidity, expected_crop = scenario
    
    # Create feature array with engineered features
    NPK_Total = N + P + K
    NP_Ratio = N / (P + 1e-5)
    NK_Ratio = N / (K + 1e-5)
    
    # Prepare moisture prediction input
    moisture_input = np.array([[N, P, K, Temp, Humidity, NPK_Total, NP_Ratio, NK_Ratio]])
    moisture_input_scaled = scaler_reg.transform(moisture_input)
    moisture_pred = moisture_model.predict(moisture_input_scaled)[0]
    
    # Prepare crop prediction input (includes moisture)
    crop_input = np.array([[N, P, K, Temp, Humidity, NPK_Total, NP_Ratio, NK_Ratio, moisture_pred]])
    crop_input_scaled = scaler_clf.transform(crop_input)
    crop_encoded = crop_model.predict(crop_input_scaled)[0]
    crop_pred = le.inverse_transform([crop_encoded])[0]
    
    status = "✅" if crop_pred == expected_crop else "❌"
    print(f"   {N:3.0f} {P:3.0f} {K:3.0f} {Temp:4.0f} {Humidity:5.0f} | "
          f"{moisture_pred:6.1f}%     | {crop_pred:8} | {expected_crop:8} {status}")

# ==================== MODEL METADATA ====================
print("\n📋 MODEL METADATA SUMMARY:")

model_info = {
    'dataset_size': len(df),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'moisture_model_mae': float(mse),
    'crop_model_accuracy': float(accuracy),
    'feature_count_moisture': len(moisture_features),
    'feature_count_crop': len(crop_features),
    'crop_classes': list(le.classes_),
    'test_set_size': len(X_m_test)
}

joblib.dump(model_info, 'models/model_metadata.pkl')

print(f"   Dataset Size: {model_info['dataset_size']:,}")
print(f"   Training Date: {model_info['training_date']}")
print(f"   Moisture Model MAE: {model_info['moisture_model_mae']:.2f}%")
print(f"   Crop Model Accuracy: {model_info['crop_model_accuracy']:.2%}")
print(f"   Crop Classes: {model_info['crop_classes']}")

print("\n🎉 BULLETPROOF MODEL TRAINING COMPLETED!")
print("📁 All models saved in 'models/' directory")
print("🚀 Your Flask app is now ready to use these high-quality models!")
print("💡 Models include feature engineering and comprehensive validation")