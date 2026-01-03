# demo.py
from urllib import response
from guardrails.safety_rules import apply_guardrails
from llm.llama_inference import MaintenanceLLM

# STEP 1: CV — Thermal preprocessing & anomaly detection
from cv.thermal_preprocessing import preprocess_thermal_image
from cv.anomaly_detection import detect_thermal_anomaly

# STEP 2: Engineering fault interpretation
from cv.fault_interpretation import interpret_motor_fault

# STEP 3: RAG retrieval
from rag.vector_store import MaintenanceVectorStore

# STEP 4: Prompt builder
from llm.prompt_templates import build_prompt


def main():
    # ---- INPUT IMAGE ----
    image_path = "data/thermal_images/motor_bearing_overheat/bearing_0.png"

    # ---- CV PIPELINE ----
    img = preprocess_thermal_image(image_path)
    features, _ = detect_thermal_anomaly(img)

    print("\n=== Thermal Anomaly Features ===")
    print(features)

    # ---- ENGINEERING INTERPRETATION ----
    fault_info = interpret_motor_fault(features)

    print("\n=== Engineering Interpretation ===")
    for k, v in fault_info.items():
        print(f"{k}: {v}")

    # ---- RAG RETRIEVAL ----
    store = MaintenanceVectorStore()
    query = "Motor bearing overheating with localized hotspot and high temperature"
    retrieved_logs = store.retrieve(query)

    print("\n=== Retrieved Maintenance Incidents ===")
    for r in retrieved_logs:
        print(r)

    # ---- BUILD PROMPT ----
    llm_prompt = build_prompt(features, fault_info, retrieved_logs)

    print("\n=== LLM PROMPT (Preview) ===")
    print(llm_prompt)

    # ---- LOCAL LLaMA INFERENCE ----
    MODEL_PATH = (
        "C:/llama-b7613-bin-win-cuda-12.4-x64/models/"
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    )

    llm = MaintenanceLLM(
        model_path=MODEL_PATH,
        n_threads=8,
        n_gpu_layers=20
    )

    print("\n=== RAW LLM OUTPUT ===")
    response = llm.generate(llm_prompt)
    final_decision = apply_guardrails(response)
    
    print("\n=== FINAL GUARDED DECISION (COPILOT OUTPUT) ===")
    print(final_decision.model_dump_json(indent=2))



if __name__ == "__main__":
    main()
