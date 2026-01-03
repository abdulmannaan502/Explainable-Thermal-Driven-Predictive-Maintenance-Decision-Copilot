# app/ui.py
import sys
import os
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from cv.thermal_preprocessing import preprocess_thermal_image
from cv.anomaly_detection import detect_thermal_anomaly
from cv.fault_interpretation import interpret_motor_fault
from rag.vector_store import MaintenanceVectorStore
from llm.prompt_templates import build_prompt
from llm.llama_inference import MaintenanceLLM
from guardrails.safety_rules import apply_guardrails

st.set_page_config(page_title="Thermal Maintenance Copilot", layout="centered")

st.title("🔥 Explainable Thermal Predictive Maintenance LLM")
st.caption("Decision-support tool for industrial motor maintenance")

# ---- Model Path ----
MODEL_PATH = "C:/llama-b7613-bin-win-cuda-12.4-x64/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# ---- Upload Image ----
uploaded_file = st.file_uploader(
    "Upload a motor thermal image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Thermal Image", use_column_width=True)

    if st.button("Analyze Thermal Image"):
        with st.spinner("Analyzing thermal anomaly..."):

            # Save uploaded image temporarily
            image_path = "temp_uploaded_image.png"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # ---- CV PIPELINE ----
            img = preprocess_thermal_image(image_path)
            features, _ = detect_thermal_anomaly(img)

        st.subheader("🧪 Thermal Anomaly Features")
        st.json(features)

        # ---- FAULT INTERPRETATION ----
        fault_info = interpret_motor_fault(features)
        st.subheader("🛠 Engineering Interpretation")
        st.json(fault_info)

        # ---- RAG RETRIEVAL ----
        with st.spinner("Retrieving similar historical incidents..."):
            store = MaintenanceVectorStore()
            query = "Motor bearing overheating with localized hotspot and high temperature"
            retrieved_logs = store.retrieve(query)

        st.subheader("📚 Retrieved Historical Maintenance Incidents")
        for log in retrieved_logs:
            st.json(log)

        # ---- LLM REASONING ----
        with st.spinner("Generating decision support (local LLaMA)..."):
            prompt = build_prompt(features, fault_info, retrieved_logs)

            llm = MaintenanceLLM(
                model_path=MODEL_PATH,
                n_threads=8,
                n_gpu_layers=20
            )

            llm_response = llm.generate(prompt)

        st.subheader("🤖 LLM Raw Reasoning")
        st.text(llm_response)

        # ---- GUARDRAILS ----
        final_decision = apply_guardrails(llm_response)

        st.subheader("✅ Final Guarded Decision")
        st.json(final_decision.model_dump())
