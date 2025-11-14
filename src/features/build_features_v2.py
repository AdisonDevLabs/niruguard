import pandas as pd
import numpy as np
import os

# --- V2 CONFIG ---
DATA_DIR = os.path.join('data', 'raw')
# We are creating a NEW, more powerful data file
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'training_data_v2.csv') 

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

# --- FIX V5 ---
# The old 'convert_to_datetime' helper function is removed, as it caused the error.
# We will use pd.to_datetime directly inside engineer_features_v2.

def load_and_merge_v2():
    """
    Loads and intelligently merges all 4 key data sources.
    """
    try:
        print("V2 ⚙️: Loading all 4 data sources...")
        # 1. Load Main (Procurement Method)
        main_df = pd.read_csv(MAIN_FILE, 
                            usecols=['_link', 'tender_procurementMethod'],
                            low_memory=False)
        
        # 2. Load Awards (Price)
        awards_df = pd.read_csv(AWARDS_FILE, 
                              usecols=['_link_main', 'value_amount'],
                              low_memory=False)
        awards_df = awards_df.drop_duplicates(subset=['_link_main'], keep='first')

        # 3. Load Contracts (Timing)
        # We need the full columns, not just a subset, as we merge on _link_main
        contracts_df = pd.read_csv(CONTRACTS_FILE,
                                 usecols=['_link_main', 'dateSigned', 'period_startDate'],
                                 low_memory=False)
        contracts_df = contracts_df.drop_duplicates(subset=['_link_main'], keep='first')

        # 4. Load Suppliers (The "Who")
        suppliers_df = pd.read_csv(SUPPLIERS_FILE,
                                 usecols=['_link_main', 'name'],
                                 low_memory=False)
        suppliers_df['name'] = suppliers_df['name'].str.strip().str.upper()
        suppliers_df = suppliers_df.drop_duplicates(subset=['_link_main'], keep='first')
        
        print("✅ Data loaded. Starting advanced merge...")

        # --- Merging ---
        # Base: main + awards
        merged_df = pd.merge(main_df, awards_df, left_on='_link', right_on='_link_main', how='inner')
        
        # Add contract timing data
        merged_df = pd.merge(merged_df, contracts_df, on='_link_main', how='left')
        
        # Add supplier data
        merged_df = pd.merge(merged_df, suppliers_df, on='_link_main', how='left')
        
        print(f"✅ Merge complete. Total {len(merged_df)} awarded contracts found.")
        return merged_df

    except Exception as e:
        print(f"❌ An error occurred during load/merge: {e}")
        return None

def engineer_features_v2(df):
    """Calculates the advanced 'Red Flags' from the merged data."""
    print("V2 ⚙️: Engineering advanced features...")
    
    model_df = pd.DataFrame()

    # --- OLD FEATURES ---
    model_df['amount'] = df['value_amount'].apply(clean_currency)
    model_df['is_direct_procurement'] = df['tender_procurementMethod'].apply(
        lambda x: 1 if str(x).lower() == 'direct' else 0
    )
    model_df['is_round_amount'] = model_df['amount'].apply(
        lambda x: 1 if x > 1000 and x % 1000 == 0 else 0
    )
    
    # --- NEW FEATURE 5: SUSPICIOUS TIMING (FIX V5) ---
    # Use pd.to_datetime directly on the series.
    # errors='coerce' will turn any bad dates into 'NaT' (Not a Time), which is safe.
    date_signed = pd.to_datetime(df['dateSigned'], errors='coerce')
    start_date = pd.to_datetime(df['period_startDate'], errors='coerce')
    # This comparison will be False for any NaT/NaN rows, which is what we want.
    model_df['suspicious_timing'] = (start_date < date_signed).astype(int)

    # --- NEW FEATURE 6: SUPPLIER HISTORY ---
    # Count all awards for each supplier
    supplier_counts = df['name'].value_counts()
    # Map this count back to each contract. 
    # .fillna(1) assumes a supplier with no name (NaN) is a "new" one.
    model_df['supplier_award_count'] = df['name'].map(supplier_counts).fillna(1).astype(int)
    
    # Flag 1 if this is a "new" supplier (<= 3 awards) getting a direct deal
    model_df['new_supplier_direct_deal'] = (
        (model_df['supplier_award_count'] <= 3) & (model_df['is_direct_procurement'] == 1)
    ).astype(int)

    # --- V2 SYNTHETIC RISK LABEL ---
    # Our risk score is now much smarter
    risk_score = (model_df['is_direct_procurement'] * 1.5) + \
                 (model_df['is_round_amount'] * 1.0) + \
                 (model_df['suspicious_timing'] * 2.0) + \
                 (model_df['new_supplier_direct_deal'] * 2.5) # This is a major red flag
                 
    # We define "High Risk" as any score >= 2.0
    model_df['risk_label'] = (risk_score >= 2.0).astype(int)

    print("✅ Advanced features engineered.")
    
    # Define the final features for the V2 model
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
    
    return model_df

def main():
    merged_df = load_and_merge_v2()
    
    if merged_df is not None:
        model_df = engineer_features_v2(merged_df)
        
        # Save V2 data
        model_df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"💾 Processed V2 data saved to: {PROCESSED_DATA_PATH}")
        print("\n--- Preview of V2 data for the AI ---")
        print(model_df.head())
        print(f"\nDistribution of V2 Risk Labels:\n{model_df['risk_label'].value_counts(normalize=True)}")

if __name__ == "__main__":
    main()