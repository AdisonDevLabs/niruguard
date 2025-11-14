import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- V2 CONFIG ---
DATA_PATH = os.path.join('data', 'processed', 'training_data_v2.csv')
MODEL_PATH = os.path.join('models', 'corruption_risk_model_v2.pkl') # New V2 model

def train_model_v2():
    # 1. Load the processed V2 data
    print("V2 🤖: Loading processed V2 data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {DATA_PATH}")
        print("Please make sure you have run 'src/features/build_features_v2.py' successfully.")
        return
    
    print("✅ V2 Data loaded.")
    
    # 2. Define V2 Features (X) and Target (y)
    # X = The 6 'Genius' Red Flags
    # y = The 'Risk Label'
    
    # --- V2 FEATURES ---
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # 'stratify=y' is critical: It ensures both train and test sets get a 6% slice of the '1's
    
    print(f"Split data: {len(X_train)} training samples, {len(X_test)} testing samples.")
    
    # 4. Initialize and Train the V2 AI Model
    print("V2 🤖: Training the ADVANCED RandomForestClassifier...")
    
    # --- THE CRITICAL UPGRADE ---
    # class_weight='balanced' tells the model to "pay extra attention" to the rare 6%
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced' # This is the magic
    )
    
    model.fit(X_train, y_train)
    print("✅ V2 Model trained.")
    
    # 5. Evaluate the V2 Model
    print("🔬 Evaluating V2 model performance...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- V2 Model Performance ---")
    print(f"🎯 Overall Accuracy: {accuracy * 100:.2f}%")
    print("   (NOTE: Accuracy is misleading here. The Detailed Report is what matters.)")
    
    # --- THE IMPORTANT PART ---
    # We must look at the 'recall' for label 1.
    print("\n--- V2 Detailed Report (The *Real* Score) ---")
    print(classification_report(y_test, y_pred))
    
    # 6. Save the trained V2 model
    print(f"💾 Saving trained V2 model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("✅ Genius Model saved. 'Day 4' (part 2) is complete!")

if __name__ == "__main__":
    train_model_v2()