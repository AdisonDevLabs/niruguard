import streamlit as st
import pandas as pd
import joblib
import os

# --- CONFIG ---
MODEL_PATH = os.path.join('models', 'corruption_risk_model.pkl')

# --- LOAD MODEL ---
@st.cache_data() # Cache the model so it only loads once
def load_model():
    """Loads the saved AI model from the file."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"FATAL: Model file not found at {MODEL_PATH}. Did you run 'src/train_model.py'?")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model()

# --- PAGE SETUP ---
# Set the title and icon for the browser tab
st.set_page_config(
    page_title="NiruGuard Risk Detector",
    page_icon="🛡️",
    layout="centered"
)

# --- APP HEADER ---
st.title("🛡️ NiruGuard: Corruption Risk Detector")
st.markdown("An AI tool to detect high-risk procurement contracts, built for the AI Hackathon 2025.")

# --- INPUT FORM ---
st.header("Analyze a New Contract")

# Create a form for user input
with st.form("contract_form"):
    
    # 1. Contract Amount
    amount = st.number_input(
        "Contract Amount (in KES)",
        min_value=0.0,
        step=1000.0,
        value=500000.0, # Default example
        help="Enter the total value of the contract."
    )
    
    # 2. Procurement Method
    procurement_method = st.selectbox(
        "Procurement Method",
        ("Open", "Direct"), # We simplify to the two most important
        help="Was this an open tender or a direct (sole-source) award?"
    )
    
    # 3. Missing Data (as a simple proxy)
    missing_data = st.checkbox(
        "Is contract information incomplete?",
        help="Check this box if key details (like dates or supplier info) are missing."
    )
    
    # Submit Button
    submitted = st.form_submit_button("Analyze Risk")

# --- PREDICTION LOGIC ---
if model is not None and submitted:
    # 1. Engineer the features from the user's input
    # This MUST match the features we trained on
    
    # is_direct_procurement (1 if 'Direct', 0 if 'Open')
    is_direct_procurement = 1 if procurement_method == "Direct" else 0
    
    # is_round_amount (1 if > 1000 and divisible by 1000)
    is_round_amount = 1 if amount > 1000 and amount % 1000 == 0 else 0
    
    # missing_data_count (1 if checked, 0 if not)
    missing_data_count = 1 if missing_data else 0

    # 2. Create a DataFrame for the model
    # The model expects a pandas DataFrame in a specific order
    input_data = pd.DataFrame(
        {
            'amount': [amount],
            'is_direct_procurement': [is_direct_procurement],
            'is_round_amount': [is_round_amount],
            'missing_data_count': [missing_data_count]
        }
    )
    
    # 3. Make the prediction
    prediction_proba = model.predict_proba(input_data)[0]
    prediction = prediction_proba.argmax()
    confidence = prediction_proba.max()

    # 4. Display the result
    st.header("Risk Analysis Result")
    
    if prediction == 1: # High Risk
        st.error(f"""
            ### 🚨 ALERT: HIGH RISK DETECTED
            **Confidence:** {confidence * 100:.0f}%
            
            This contract flags multiple "Red Flags" associated with corruption or fraud.
            Recommend for immediate audit.
        """)
    else: # Low Risk
        st.success(f"""
            ### ✅ RESULT: LOW RISK
            **Confidence:** {confidence * 100:.0f}%
            
            This contract appears to be standard and does not trigger automated red flags.
        """)

    # --- Show "Why?" (Explainability) ---
    st.subheader("Analysis Breakdown (The 'Red Flags'):")
    st.markdown(f"- **Sole-Source (Direct):** `{'Yes' if is_direct_procurement == 1 else 'No'}`")
    st.markdown(f"- **Suspicious Round Amount:** `{'Yes' if is_round_amount == 1 else 'No'}`")
    st.markdown(f"- **Incomplete Data:** `{'Yes' if missing_data_count == 1 else 'No'}`")

else:
    st.info("Fill out the form above to analyze a contract's risk level.")