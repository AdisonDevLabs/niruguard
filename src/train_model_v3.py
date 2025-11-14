import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- V3 'GRAND CHAMPION' CONFIG ---
DATA_PATH = os.path.join('data', 'processed', 'training_data_v3.csv')
MODEL_PATH = os.path.join('models', 'corruption_risk_model_v3.pkl') # New V3 model

def train_model_v3():
    # 1. Load the processed V3 data
    print("V3 🏆: Loading processed V3 data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {DATA_PATH}")
        print("Please make sure you have run 'src/features/build_features_v3.py' successfully.")
        return
    
    print("✅ V3 Data loaded.")
    
    # 2. Define V3 Features (X) and Target (y)
    # X = The 'Grand Champion' Red Flags
    # y = The 'Risk Label'
    
    # --- V3 FEATURES ---
    features = [
        'amount', 
        'is_direct_procurement', 
        'is_round_amount', 
        'suspicious_timing', 
        'supplier_award_count',
        'new_supplier_direct_deal'
    ]
    target = 'risk_label'
    
    X = df[features]
    y = df[target]
    
    # 3. Split Data into Training and Testing sets
    # 'stratify=y' is critical: It ensures both sets get a 7.1% slice of the '1's
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Split data: {len(X_train)} training samples, {len(X_test)} testing samples.")
    
    # 4. Initialize and Train the V3 AI Model
    print("V3 🏆: Training the 'GRAND CHAMPION' RandomForestClassifier...")
    
    # --- THE CRITICAL UPGRADE ---
    # class_weight='balanced' tells the model to hunt for the rare 7.1%
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced' 
    )
    
    model.fit(X_train, y_train)
    print("✅ V3 Model trained.")
    
    # 5. Evaluate the V3 Model
    print("🔬 Evaluating V3 model performance...")
    y_pred = model.predict(X_test)
    
    print("\n--- V3 Detailed Report (The *Real* Score) ---")
    print(classification_report(y_test, y_pred))
    
    # 6. Save the trained V3 model
    print(f"💾 Saving trained V3 model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("✅ 'Grand Champion' Model saved. 'Day 5' (part 2) is complete!")

if __name__ == "__main__":
    train_model_v3()