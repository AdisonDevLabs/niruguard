import streamlit as st
import pandas as pd
import joblib
import os

# --- V2 CONFIG ---
MODEL_PATH = os.path.join('models', 'corruption_risk_model_v2.pkl')

# --- LOAD V2 MODEL ---
@st.cache_data() # Cache the model so it only loads once
def load_model():
    """Loads the saved V2 AI model from the file."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"FATAL: V2 Model file not found at {MODEL_PATH}. Did you run 'src/train_model_v2.py'?")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the V2 model: {e}")
        return None

model = load_model()

# --- PAGE SETUP ---
st.set_page_config(
    page_title="NiruGuard V2 (Advanced)",
    page_icon="🏆",
    layout="centered"
)

# --- APP HEADER ---
st.title("🏆 NiruGuard V2: Advanced Risk Detector")
st.markdown("**(Best Overall Edition)**: This AI uses advanced behavioral and timing analysis to find fraudulent contracts.")

# --- INPUT FORM ---
st.header("Analyze a New Contract (Advanced)")

# Create a form for user input
with st.form("contract_form_v2"):
    
    st.info("This advanced model uses behavioral 'Red Flags' to find risk.")
    
    # We will use two columns for a cleaner layout
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Contract Amount
        amount = st.number_input(
            "Contract Amount (in KES)",
            min_value=0.0,
            step=1000.0,
            value=5000000.0, # Example: 5M
            help="Enter the total value of the contract."
        )
    
    with col2:
        # 2. Procurement Method
        procurement_method = st.selectbox(
            "Procurement Method",
            ("Direct", "Open"), # 'Direct' is the high-risk one
            help="Was this an open tender or a direct (sole-source) award?"
        )
    
    st.subheader("Advanced Red Flags (V2)")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # 3. Supplier History
        supplier_award_count = st.number_input(
            "Supplier Total Awards",
            min_value=1,
            step=1,
            value=1, # Example: A brand new supplier
            help="How many contracts has this supplier *ever* won? (1 = new)"
        )
        
    with col4:
        # 4. Suspicious Timing
        suspicious_timing = st.checkbox(
            "Suspicious Timing?",
            value=True, # Example
            help="Was the work scheduled to start *before* the contract was signed?"
        )
    
    # Submit Button
    submitted = st.form_submit_button("Analyze Risk (V2 Engine)")

# --- V2 PREDICTION LOGIC ---
if model is not None and submitted:
    # 1. Engineer the V2 features from the user's input
    
    # --- Base Features ---
    is_direct_procurement = 1 if procurement_method == "Direct" else 0
    is_round_amount = 1 if amount > 1000 and amount % 1000 == 0 else 0
    
    # --- Advanced V2 Features ---
    # 'suspicious_timing' (1 if checked, 0 if not)
    is_suspicious_timing = 1 if suspicious_timing else 0
    
    # 'new_supplier_direct_deal' (1 if supplier awards <= 3 AND it's a direct deal)
    is_new_supplier_direct_deal = 1 if (supplier_award_count <= 3 and is_direct_procurement == 1) else 0

    # 2. Create a DataFrame for the V2 model
    # The order MUST match the training script
    input_data = pd.DataFrame(
        {
            'amount': [amount],
            'is_direct_procurement': [is_direct_procurement],
            'is_round_amount': [is_round_amount],
            'suspicious_timing': [is_suspicious_timing],
            'supplier_award_count': [supplier_award_count],
            'new_supplier_direct_deal': [is_new_supplier_direct_deal]
        }
    )
    
    # 3. Make the prediction
    prediction_proba = model.predict_proba(input_data)[0]
    prediction = prediction_proba.argmax()
    confidence = prediction_proba.max()

    # 4. Display the result
    st.header("V2 Risk Analysis Result")
    
    if prediction == 1: # High Risk
        st.error(f"""
            ### 🚨 ALERT: HIGH RISK DETECTED
            **Confidence:** {confidence * 100:.0f}%
            
            This contract shows multiple, severe behavioral red flags. **Strongly recommend immediate audit.**
        """)
    else: # Low Risk
        st.success(f"""
            ### ✅ RESULT: LOW RISK
            **Confidence:** {confidence * 100:.0f}%
            
            This contract does not trigger advanced behavioral red flags.
        """)

    # --- Show "Why?" (Explainability) ---
    st.subheader("V2 Analysis Breakdown (The 'Red Flags'):")
    st.markdown(f"- **Sole-Source (Direct):** `{'Yes' if is_direct_procurement == 1 else 'No'}`")
    st.markdown(f"- **Suspicious Round Amount:** `{'Yes' if is_round_amount == 1 else 'No'}`")
    st.markdown(f"- **Suspicious Timing (Work before contract):** `{'Yes' if is_suspicious_timing == 1 else 'No'}`")
    st.markdown(f"- **New Supplier on Direct Deal:** `{'Yes' if is_new_supplier_direct_deal == 1 else 'No'}`")

else:
    st.info("Fill out the V2 form above to analyze a contract's risk level.")