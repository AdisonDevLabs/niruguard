import pandas as pd
import numpy as np
import os

# --- CONFIG: Define file paths ---
DATA_DIR = os.path.join('data', 'raw')
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'training_data.csv')

# Define the filenames we need
MAIN_FILE = os.path.join(DATA_DIR, 'main.csv')
AWARDS_FILE = os.path.join(DATA_DIR, 'awards.csv')

def clean_currency(x):
    """Helper to turn 'KES 100,000' or 100000.0 into a number."""
    if pd.isna(x): return 0
    if isinstance(x, (int, float)): return x
    # Remove 'KES', commas, and spaces
    clean_str = str(x).replace('KES', '').replace(',', '').replace(' ', '')
    try:
        return float(clean_str)
    except:
        return 0

def load_and_merge():
    """
    Loads and intelligently merges the required data from the 5 CSVs.
    We will focus on main.csv and awards.csv for our core features.
    """
    try:
        print("⏳ Loading main.csv...")
        # --- FIX V2 ---
        # Using the *actual* column names from your file: '_link' and 'tender_procurementMethod'
        main_df = pd.read_csv(MAIN_FILE, 
                            usecols=['_link', 'tender_procurementMethod'],
                            low_memory=False)
        
        print("⏳ Loading awards.csv...")
        # --- FIX V3 ---
        # Using the *actual* column names. REMOVED 'status' column which caused the crash.
        awards_df = pd.read_csv(AWARDS_FILE, 
                              usecols=['_link_main', 'value_amount'],
                              low_memory=False)

        print("✅ Data loaded. Starting merge...")
        
        # --- Data Cleaning & Merging ---
        
        # --- FIX V3 ---
        # REMOVED the filter for 'status == active'
        
        # A contract can have multiple awards. For this MVP, we simplify:
        # We will take the *first* active award for each contract.
        awards_df = awards_df.drop_duplicates(subset=['_link_main'], keep='first')
        
        # --- FIX V2 ---
        # Merge main data (procurement method) with awards data (price)
        # This is the core "database JOIN"
        # We link main_df's '_link' column to awards_df's '_link_main' column
        merged_df = pd.merge(main_df, awards_df, 
                             left_on='_link', right_on='_link_main', 
                             how='inner')
        
        print(f"✅ Merge complete. Total {len(merged_df)} awarded contracts found.")
        return merged_df

    except FileNotFoundError as e:
        print(f"❌ Error: File not found. Make sure {e.filename} is in data/raw/")
        return None
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return None

def engineer_features(df):
    """Calculates the 'Red Flags' from the merged data."""
    print("⚙️ Engineering features (Calculating Red Flags)...")
    
    # Create a clean dataframe for the model
    model_df = pd.DataFrame()

    # --- FEATURE 1: TENDER VALUE ---
    # --- FIX V2 --- Using 'value_amount'
    model_df['amount'] = df['value_amount'].apply(clean_currency)

    # --- FEATURE 2: SOLE SOURCE (Direct Procurement) ---
    # --- FIX V2 --- Using 'tender_procurementMethod'
    model_df['is_direct_procurement'] = df['tender_procurementMethod'].apply(
        lambda x: 1 if str(x).lower() == 'direct' else 0
    )

    # --- FEATURE 3: ROUND NUMBER PRICING (Fraud indicator) ---
    # 1 if amount is > 1000 and perfectly divisible by 1000, else 0
    model_df['is_round_amount'] = model_df['amount'].apply(
        lambda x: 1 if x > 1000 and x % 1000 == 0 else 0
    )

    # --- FEATURE 4: DATA COMPLETENESS (Transparency Score) ---
    # Count how many empty spots are in the *original* merged row
    model_df['missing_data_count'] = df.isnull().sum(axis=1)

    # --- (TARGET) OUR SYNTHETIC RISK LABEL ---
    # This is what the AI will learn to predict.
    # We create a "Risk Score" based on our red flags.
    
    risk_score = (model_df['is_direct_procurement'] * 2) + \
                 (model_df['is_round_amount'] * 1.5) + \
                 (model_df['missing_data_count'] >= 1).astype(int) # Penalize any missing data
                 
    # We define "High Risk" as any contract with a score >= 1.5
    model_df['risk_label'] = (risk_score >= 1.5).astype(int)

    print("✅ Features engineered.")
    
    # We only keep the columns the AI needs to learn
    final_features = ['amount', 'is_direct_procurement', 'is_round_amount', 'missing_data_count', 'risk_label']
    model_df = model_df[final_features]
    
    return model_df

def main():
    merged_df = load_and_merge()
    
    if merged_df is not None:
        model_df = engineer_features(merged_df)
        
        # Save
        model_df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"💾 Processed data saved to: {PROCESSED_DATA_PATH}")
        print("\n--- Preview of data for the AI ---")
        print(model_df.head())
        print(f"\nDistribution of Risk Labels:\n{model_df['risk_label'].value_counts(normalize=True)}")

if __name__ == "__main__":
    main()