import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIG ---
DATA_PATH = os.path.join('data', 'processed', 'training_data.csv')
MODEL_PATH = os.path.join('models', 'corruption_risk_model.pkl')

def train_model():
    # 1. Load the processed data
    print("⏳ Loading processed data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {DATA_PATH}")
        print("Please make sure you have run 'src/features/build_features.py' successfully.")
        return
    
    print("✅ Data loaded.")
    
    # 2. Define Features (X) and Target (y)
    # X = The 'Red Flags' (our inputs)
    # y = The 'Risk Label' (our output)
    features = ['amount', 'is_direct_procurement', 'is_round_amount', 'missing_data_count']
    target = 'risk_label'
    
    X = df[features]
    y = df[target]
    
    # 3. Split Data into Training and Testing sets
    # 80% for training, 20% for testing. 'random_state=42' ensures we get the same split every time.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Split data: {len(X_train)} training samples, {len(X_test)} testing samples.")
    
    # 4. Initialize and Train the AI Model
    print("🤖 Training the RandomForestClassifier...")
    # A RandomForest is great for this. It's like a group of "experts" (trees) voting.
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all your CPU cores
    model.fit(X_train, y_train)
    print("✅ Model trained.")
    
    # 5. Evaluate the Model
    print("🔬 Evaluating model performance...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- Model Performance ---")
    print(f"🎯 Accuracy on Test Data: {accuracy * 100:.2f}%")
    
    # Show a detailed report (Precision, Recall)
    # Precision (0): How good at finding NON-risk?
    # Precision (1): How good at finding HIGH-risk?
    print("\n--- Detailed Report ---")
    print(classification_report(y_test, y_pred))
    
    # 6. Save the trained model
    print(f"💾 Saving trained model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("✅ Model saved. 'Day 1' is complete!")

if __name__ == "__main__":
    train_model()