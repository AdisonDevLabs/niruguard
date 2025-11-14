import streamlit as st
import pandas as pd
import os

# --- V3 'GRAND CHAMPION' CONFIG ---
DATA_DIR = os.path.join('data', 'raw')
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'training_data_v3.csv')
PARTIES_FILE = os.path.join(DATA_DIR, 'parties.csv')

# --- LOAD ALL DATA (CACHE IT) ---
@st.cache_data()
def load_all_data():
    """
    Loads both the processed V3 data and the parties 'address book'.
    This is the 'database' for our intelligence platform.
    """
    try:
        print("Supplier_360 LOAD: Loading V3 training data...")
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        st.error(f"FATAL: V3 Data file not found at {PROCESSED_DATA_PATH}.")
        return None, None

    try:
        print("Supplier_360 LOAD: Loading parties.csv...")
        # We use the 'id' (fingerprint) and 'name'
        parties_df = pd.read_csv(PARTIES_FILE, 
                                 usecols=['id', 'name'], 
                                 low_memory=False)
        parties_df.rename(columns={'id': 'supplier_id', 'name': 'official_name'}, inplace=True)
        # Drop duplicates, keep the first official name for each ID
        parties_df = parties_df.drop_duplicates(subset=['supplier_id'], keep='first')
        
    except FileNotFoundError:
        st.error(f"FATAL: parties.csv file not found at {PARTIES_FILE}.")
        return None, None
    except Exception as e:
        # Handle case where 'id' or 'name' column isn't found
        st.error(f"Error loading parties.csv: {e}")
        return None, None
    
    # We merge our V3 data with the official party names
    # This ensures we have the *most official* name from the 'address book'
    # We use the 'supplier_id' (fingerprint) as the key
    
    # We must ensure supplier_id is the same type.
    # We know parties_df['supplier_id'] is a string (e.g., '83297').
    # Let's ensure our main df's supplier_id is also a string.
    df['supplier_id'] = df['supplier_id'].astype(str)
    parties_df['supplier_id'] = parties_df['supplier_id'].astype(str)

    # Left join to keep all contracts, even if they're missing a party file entry
    full_db = pd.merge(df, parties_df, on='supplier_id', how='left')
    
    # Fill in any missing 'official_name' with the name from the awards file
    full_db['official_name'] = full_db['official_name'].fillna(full_db['supplier_name'])
    
    print("Supplier_360 LOAD: All data loaded and merged.")
    return full_db

# --- PAGE SETUP ---
st.set_page_config(
    page_title="NiruGuard V3 | Supplier 360",
    page_icon="🎯",
    layout="wide" # Use wide layout for a dashboard
)

st.title("🎯 Supplier 360° Intelligence Platform")
st.markdown("Investigate any supplier to see their full risk profile and contract history.")

# Load the data
df = load_all_data()

if df is not None:
    # --- SUPPLIER SELECTION ---
    st.header("Select a Supplier to Investigate")
    
    # Get a unique list of supplier names and IDs
    # We drop suppliers with no name
    supplier_list = df.dropna(subset=['official_name'])
    # Create a display list: "Name (ID: 12345)"
    supplier_list['display_name'] = supplier_list['official_name'] + " (ID: " + supplier_list['supplier_id'] + ")"
    
    # Get a unique list of display names
    unique_suppliers = supplier_list['display_name'].unique()
    
    # --- Create the search box ---
    selected_supplier_display = st.selectbox(
        "Search for a supplier by name or ID:",
        options=unique_suppliers,
        index=None, # Default to empty
        placeholder="Type name or ID to search..."
    )

    # --- SUPPLIER DOSSIER ---
    if selected_supplier_display:
        # Extract the ID from the display name
        selected_id = selected_supplier_display.split(' (ID: ')[-1].replace(')', '')
        
        # Filter our main database for this one supplier
        supplier_df = df[df['supplier_id'] == selected_id].copy()
        
        supplier_name = supplier_df['official_name'].iloc[0]
        st.header(f"Dossier: {supplier_name}", divider="rainbow")

        # --- Top-Level Metrics ---
        st.subheader("Key Performance Indicators (KPIs)")
        
        total_contracts = len(supplier_df)
        total_value = supplier_df['amount'].sum()
        high_risk_count = supplier_df['risk_label'].sum()
        high_risk_value = supplier_df[supplier_df['risk_label'] == 1]['amount'].sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Contracts Won", f"{total_contracts}")
        col2.metric("Total Contract Value", f"KES {total_value:,.0f}")
        col3.metric("High Risk Contracts", f"{high_risk_count}", 
                     delta=f"{high_risk_value:,.0f} KES at risk", 
                     delta_color="inverse")
        
        # --- Risk Breakdown ---
        st.subheader("Risk Profile")
        
        col4, col5 = st.columns(2)
        with col4:
            st.markdown("##### Contracts by Risk")
            # Create a simple DataFrame for the pie chart
            risk_counts = supplier_df['risk_label'].value_counts().reset_index()
            risk_counts['risk_label'] = risk_counts['risk_label'].map({0: 'Low Risk', 1: 'High Risk'})
            st.bar_chart(risk_counts, x='risk_label', y='count')

        with col5:
            st.markdown("##### Contracts by Type")
            # Map 0/1 to 'Open'/'Direct'
            supplier_df['method'] = supplier_df['is_direct_procurement'].map({0: 'Open', 1: 'Direct'})
            method_counts = supplier_df['method'].value_counts()
            st.bar_chart(method_counts)
        
        # --- High Risk Contracts Table ---
        st.subheader("High Risk Contract Table")
        high_risk_df = supplier_df[supplier_df['risk_label'] == 1]
        
        if len(high_risk_df) > 0:
            st.warning(f"Found {len(high_risk_df)} high-risk contracts.")
            # Show the key details
            st.dataframe(high_risk_df[[
                'amount',
                'is_direct_procurement',
                'is_round_amount',
                'suspicious_timing',
                'new_supplier_direct_deal'
            ]])
        else:
            st.success("This supplier has no high-risk contracts in our database.")
            
else:
    st.error("The main database could not be loaded. Please check file paths and run build scripts.")