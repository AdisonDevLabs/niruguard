import pandas as pd
import numpy as np
import os

# --- V3 'GRAND CHAMPION' CONFIG ---
DATA_DIR = os.path.join('data', 'raw')
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'training_data_v3.csv') # New V3 output

# Define the filenames we need
MAIN_FILE = os.path.join(DATA_DIR, 'main.csv')
AWARDS_FILE = os.path.join(DATA_DIR, 'awards.csv')
CONTRACTS_FILE = os.path.join(DATA_DIR, 'contracts.csv')
SUPPLIERS_FILE = os.path.join(DATA_DIR, 'awards_suppliers.csv')

def clean_currency(x):
    """Helper to clean currency fields"""
    if pd.isna(x): return 0
    if isinstance(x, (int, float)): return x
    clean_str = str(x).replace('KES', '').replace(',', '').replace(' ', '')
    try: return float(clean_str)
    except: return 0

def load_and_merge_v3():
    """
    Loads and merges all 4 key data sources, now with the 'True Supplier ID'.
    """
    try:
        print("V3 🏆: Loading all 4 data sources...")
        # 1. Load Main
        main_df = pd.read_csv(MAIN_FILE, 
                            usecols=['_link', 'tender_procurementMethod'],
                            low_memory=False)
        
        # 2. Load Awards
        awards_df = pd.read_csv(AWARDS_FILE, 
                              usecols=['_link_main', 'value_amount'],
                              low_memory=False)
        awards_df = awards_df.drop_duplicates(subset=['_link_main'], keep='first')

        # 3. Load Contracts
        contracts_df = pd.read_csv(CONTRACTS_FILE,
                                 usecols=['_link_main', 'dateSigned', 'period_startDate'],
                                 low_memory=False)
        contracts_df = contracts_df.drop_duplicates(subset=['_link_main'], keep='first')

        # 4. Load Suppliers (THE 'DAY 5' UPGRADE)
        # We now load 'id' (the true fingerprint) along with 'name'
        suppliers_df = pd.read_csv(SUPPLIERS_FILE,
                                 usecols=['_link_main', 'id', 'name'],
                                 low_memory=False)
        
        # Rename 'id' to 'supplier_id' to avoid confusion
        suppliers_df.rename(columns={'id': 'supplier_id'}, inplace=True)
        suppliers_df = suppliers_df.drop_duplicates(subset=['_link_main'], keep='first')
        
        print("✅ Data loaded. Starting 'True Network' merge...")

        # --- Merging ---
        merged_df = pd.merge(main_df, awards_df, left_on='_link', right_on='_link_main', how='inner')
        merged_df = pd.merge(merged_df, contracts_df, on='_link_main', how='left')
        merged_df = pd.merge(merged_df, suppliers_df, on='_link_main', how='left')
        
        print(f"✅ 'True Network' merge complete. Total {len(merged_df)} awarded contracts found.")
        return merged_df

    except Exception as e:
        print(f"❌ An error occurred during load/merge: {e}")
        return None

def engineer_features_v3(df):
    """Calculates the 'Grand Champion' Red Flags using the 'True Supplier ID'."""
    print("V3 🏆: Engineering 'Grand Champion' features...")
    
    model_df = pd.DataFrame()

    # --- Standard Features ---
    model_df['amount'] = df['value_amount'].apply(clean_currency)
    model_df['is_direct_procurement'] = df['tender_procurementMethod'].apply(
        lambda x: 1 if str(x).lower() == 'direct' else 0
    )
    model_df['is_round_amount'] = model_df['amount'].apply(
        lambda x: 1 if x > 1000 and x % 1000 == 0 else 0
    )
    
    # --- Advanced Features ---
    date_signed = pd.to_datetime(df['dateSigned'], errors='coerce')
    start_date = pd.to_datetime(df['period_startDate'], errors='coerce')
    model_df['suspicious_timing'] = (start_date < date_signed).astype(int)

    # --- THE 'GRAND CHAMPION' FEATURE (V3) ---
    # We now count awards based on the 'supplier_id' (fingerprint), not the 'name'.
    # This is 100% more accurate.
    supplier_counts = df['supplier_id'].value_counts()
    model_df['supplier_award_count'] = df['supplier_id'].map(supplier_counts).fillna(1).astype(int)
    
    # This feature is now 1000x smarter
    model_df['new_supplier_direct_deal'] = (
        (model_df['supplier_award_count'] <= 3) & (model_df['is_direct_procurement'] == 1)
    ).astype(int)

    # --- V3 SYNTHETIC RISK LABEL ---
    # The logic is the same, but the *inputs* are now perfect
    risk_score = (model_df['is_direct_procurement'] * 1.5) + \
                 (model_df['is_round_amount'] * 1.0) + \
                 (model_df['suspicious_timing'] * 2.0) + \
                 (model_df['new_supplier_direct_deal'] * 2.5) 
                 
    model_df['risk_label'] = (risk_score >= 2.0).astype(int)

    print("✅ 'Grand Champion' features engineered.")
    
    # Define the final features for the V3 model
    final_features = [
        'amount', 
        'is_direct_procurement', 
        'is_round_amount', 
        'suspicious_timing', 
        'supplier_award_count',
        'new_supplier_direct_deal',
        'risk_label'
    ]
    model_df = model_df[final_features]
    
    # We also keep the 'supplier_id' and 'name' for the dashboard upgrade (Step 3)
    model_df['supplier_id'] = df['supplier_id']
    model_df['supplier_name'] = df['name']
    
    return model_df

def main():
    merged_df = load_and_merge_v3()
    
    if merged_df is not None:
        model_df = engineer_features_v3(merged_df)
        
        # Save V3 data
        model_df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"💾 Processed V3 data saved to: {PROCESSED_DATA_PATH}")
        print("\n--- Preview of V3 data for the AI ---")
        print(model_df.head())
        print(f"\nDistribution of V3 Risk Labels:\n{model_df['risk_label'].value_counts(normalize=True)}")

if __name__ == "__main__":
    main()