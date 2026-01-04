import sys
import os
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from guardrails.decision_safety import decision_safety_engine
from guardrails.risk_scoring import risk_aware_decision_engine
from guardrails.safety_rules import apply_guardrails
from temporal.trend_analysis import analyze_trend



from cv.thermal_preprocessing import preprocess_thermal_image
from cv.anomaly_detection import detect_thermal_anomaly
from cv.fault_interpretation import interpret_motor_fault
from rag.vector_store import MaintenanceVectorStore
from llm.prompt_templates import build_prompt
from llm.llama_inference import MaintenanceLLM

st.set_page_config(page_title="Thermal Maintenance Copilot", layout="centered")

st.title("🔥 Explainable Thermal Predictive Maintenance LLM")
st.caption("Decision-support tool for industrial motor maintenance")

MODEL_PATH = "C:/llama-b7613-bin-win-cuda-12.4-x64/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# -------------------------
# Session State
# -------------------------
if "results" not in st.session_state:
    st.session_state.results = None

if "feature_history" not in st.session_state:
    st.session_state.feature_history = []  # for STEP-3

uploaded_file = st.file_uploader(
    "Upload a motor thermal image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Thermal Image")

    if st.button("Analyze Thermal Image"):
        with st.spinner("Running full analysis..."):

            image_path = "temp_uploaded_image.png"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # ---- CV ----
            img = preprocess_thermal_image(image_path)
            features, _ = detect_thermal_anomaly(img)

            # ---- STEP-3: store temporal history ----
            st.session_state.feature_history.append({
                "severity_score": features["severity_score"],
                "temperature_delta": features["temperature_delta"]
            })

            fault_info = interpret_motor_fault(features)

            # ---- RAG ----
            store = MaintenanceVectorStore()
            retrieved_logs = store.retrieve(
                "Motor bearing overheating with localized hotspot"
            )

            # ---- LLM ----
            prompt = build_prompt(features, fault_info, retrieved_logs)
            llm = MaintenanceLLM(
                model_path=MODEL_PATH,
                n_threads=8,
                n_gpu_layers=20
            )

            llm_response = llm.generate(prompt)
            final_decision = apply_guardrails(llm_response)

            # ---- Guardrails ----
            safety_report = decision_safety_engine(
                features, fault_info, retrieved_logs,
                final_decision.model_dump()
            )

            risk_report = risk_aware_decision_engine(
                fault_info, retrieved_logs,
                final_decision.model_dump()
            )

            # ---- STEP-3: Temporal Trend Analysis ----
            trend_report = analyze_trend(st.session_state.feature_history)

            # ---- Save all results ----
            st.session_state.results = {
                "features": features,
                "fault_info": fault_info,
                "retrieved_logs": retrieved_logs,
                "llm_response": llm_response,
                "final_decision": final_decision,
                "safety_report": safety_report,
                "risk_report": risk_report,
                "trend_report": trend_report
            }

# -------------------------
# Render UI
# -------------------------
if st.session_state.results:
    r = st.session_state.results

    st.subheader("🧪 Thermal Anomaly Features")
    st.json(r["features"])

    st.subheader("🛠 Engineering Interpretation")
    st.json(r["fault_info"])

    st.subheader("📚 Retrieved Historical Incidents")
    for log in r["retrieved_logs"]:
        st.json(log)

    st.subheader("🤖 LLM Raw Reasoning")
    st.text(r["llm_response"])

    st.subheader("✅ Final Guarded Decision")
    st.json(r["final_decision"].model_dump())

    st.subheader("🛡 Decision Safety Report")
    st.json(r["safety_report"])

    st.subheader("🚨 Risk-Aware Decision Assessment")
    st.json(r["risk_report"])

    if r["risk_report"]["risk_level"] == "CRITICAL":
        st.error("🚨 CRITICAL — Immediate human intervention required")
    elif r["risk_report"]["risk_level"] == "CAUTION":
        st.warning("⚠️ CAUTION — Proceed with supervision")
    else:
        st.success("🟢 SAFE — Low operational risk")

    # -------------------------
    # STEP-3 UI 
    # -------------------------
    st.subheader("📈 Temporal Trend Analysis")
    st.json(r["trend_report"])

    if r["trend_report"]["urgency"] == "high":
        st.error("🔥 TREND: Worsening rapidly — schedule maintenance immediately")
    elif r["trend_report"]["urgency"] == "medium":
        st.warning("⚠️ TREND: Slowly worsening — plan maintenance")
    else:
        st.success("🟢 TREND: Stable — continue monitoring")
