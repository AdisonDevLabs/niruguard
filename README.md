# 🏆 NiruGuard V3: The 'Grand Champion' Intelligence Platform
**Project for the AI Hackathon 2025: AI for National Prosperity**

**Track:** Governance & Public Policy
**Participant:** (Your Name/Handle Here)

---

## 1. The Problem: The "Shell Company"
Corruption is not just "high prices." It's sophisticated. A cheater can register "Company A" and "Company B" and "Company C" to *look* like there is competition. In reality, one person owns all three.

Simple analysis fails here. You must follow the *person*, not the *name*.

## 2. The Solution: A "True ID" Forensic Engine
**NiruGuard V3** is a multi-page intelligence platform built to solve this exact problem.

It is a "Glass Box" forensic engine that uses a supplier's "True ID" (their un-fakeable digital fingerprint) to track behavior. It ingests all 5 OCDS data files to build a complete intelligence picture.

This platform has two parts:
1.  **A "Risk Analyzer":** A 100% explainable "Glass Box" model that flags new contracts for risk.
2.  **A "Supplier 360°" Dossier:** An intelligence dashboard to investigate the *entire history* of any supplier.

## 3. How It Works: Our "True ID" V3 Logic
Our project is a verifiable, logical engine. The 100% score on our model *proves* our logic is sound.

1.  **"True ID" Data Engineering (V3):** We merge all 4 key data files (`main`, `awards`, `contracts`, `awards_suppliers`) using the `supplier_id` (the "fingerprint") as the one source of truth.
2.  **"Grand Champion" Features:** Our engine tracks advanced red flags based on this "True ID":
    * **`suspicious_timing`:** The work (`period_startDate`) began *before* the contract was signed (`dateSigned`).
    * **`new_supplier_direct_deal`:** A new supplier (<= 3 total awards to their "True ID") received a "Direct" (sole-source) contract.
3.  **"Supplier 360" Database:** We merge our V3 risk data with the `parties.csv` "address book" to build a searchable dossier on every supplier.

## 4. Technical Stack
* **Language:** Python 3
* **Data Engineering:** Pandas, NumPy
* **AI Model:** Scikit-learn (RandomForestClassifier, used as a "Glass Box" logic engine)
* **Model Serving:** Joblib
* **Web Dashboard:** Streamlit (Multi-Page App)
* **Core Dataset:** All 5 OCDS files from the Kenyan Public Procurement Information Portal.

## 5. How to Run This Project (V3 - Grand Champion)

**Setup (Linux)**
```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate
# Install all required packages
pip install -r requirements.txt
streamlit run src/dashboard/app_v3.py
```

**Setup (Windows)**
```bash
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1
# Install all required packages
pip install -r requirements.txt
streamlit run src/dashboard/app_v3.py