# 🔥 Explainable Thermal-Driven Predictive Maintenance Decision Copilot

## 📌 Overview
This project implements an **Explainable, Evidence-Grounded Predictive Maintenance Decision Copilot** for industrial equipment (focused on electric motors).  
Instead of using black-box image classifiers, the system extracts **interpretable thermal features**, reasons over **historical maintenance incidents using RAG**, and uses a **local LLaMA LLM** to generate **safe, auditable maintenance recommendations**.

The system is designed as a **decision-support copilot**, not an autonomous controller.

---

## 🎯 Problem Statement
In industrial plants, thermal cameras are widely used to monitor motors, pumps, and bearings. However:

- Engineers manually inspect thermal images
- Correlating anomalies with historical failures is time-consuming
- Most AI solutions are black-box classifiers
- LLM-based tools risk hallucinated or unsafe recommendations

This leads to **reactive maintenance**, unplanned downtime, and higher costs.

---

## 💡 Solution Summary
This project builds an **end-to-end, explainable AI pipeline** that:

1. Detects thermal anomalies from motor thermal images  
2. Converts image data into **engineering-meaningful features**  
3. Retrieves similar **historical maintenance incidents (RAG)**  
4. Uses a **local LLaMA model** for reasoning (offline, secure)  
5. Applies **guardrails and confidence gating** before output  

The final output is a **structured, safe maintenance decision**, not raw AI text.

---

## 🧠 System Architecture

Thermal Image  
→ Explainable CV (Hotspot, ΔT, Severity)  
→ Engineering Fault Interpretation  
→ RAG (Historical Maintenance Logs)  
→ Local LLaMA (Reasoning Only)  
→ Guardrails & Safety Checks  
→ Final Decision Copilot Output (JSON)

---

## 🔍 Key Design Principles
- Explainability first (no black-box CNNs)
- No image training required
- Evidence-grounded reasoning via RAG
- Local LLM inference
- Human-in-the-loop safety

---

## 🧪 Data Strategy

### Thermal Images
- Synthetic thermal images simulating realistic motor surface temperature distributions
- Physics-consistent patterns:
  - Smooth gradients → normal operation
  - Localized hotspots → bearing overheating

### Historical Data
- Synthetic but realistic maintenance logs
- Used only at inference time for reasoning

---

## ⚙️ Pipeline Breakdown

### Explainable Thermal Feature Extraction
- Mean temperature
- Maximum temperature
- Temperature delta (ΔT)
- Hotspot count
- Severity score

### Engineering Fault Interpretation
Rule-based logic converts features into:
- Suspected fault
- Risk level
- Engineering reasoning
- Recommended inspection steps

### Retrieval-Augmented Generation (RAG)
- Historical incidents embedded into FAISS
- Similar cases retrieved at inference time
- Injected into LLM prompt to ground reasoning

### Local LLaMA Reasoning
- Uses llama.cpp (GGUF)
- Performs reasoning only
- Produces structured maintenance decisions

### Guardrails & Safety
- Schema validation
- Confidence gating
- No autonomous actions

---

## ✅ Example Output

```json
{
  "failure_mode": "bearing_wear",
  "reasoning": "Localized high-temperature anomaly and historical maintenance records indicate bearing-related degradation.",
  "recommended_action": "Inspect bearing condition and lubrication. Prepare for bearing replacement if abnormal wear is confirmed.",
  "downtime_hours_min": 2,
  "downtime_hours_max": 6,
  "repair_cost_usd_min": 300,
  "repair_cost_usd_max": 1200,
  "confidence": 0.85
}
```

---

## 🖥 User Interface
A lightweight Streamlit UI is provided for demo purposes:
- Upload thermal image
- View extracted features
- View historical incidents
- View LLM reasoning
- View final guarded decision

---

## 🏭 Real-World Workflow Fit
The system fits into factory workflows by enabling early detection of degradation and evidence-based maintenance planning before catastrophic failure.

---

## 🚫 Limitations
- Not an autonomous control system
- Uses synthetic data for prototyping
- Requires human validation before action

---

## 📈 Expandability
- Extendable to pumps, gearboxes, compressors
- Can integrate vibration, acoustic, and IoT sensors
- Can connect to CMMS / ERP systems

---

## 🧑‍💻 Tech Stack
- Python
- OpenCV / NumPy
- FAISS
- Sentence Transformers
- llama.cpp (GGUF)
- Pydantic
- Streamlit

---

## 🏁 Final Note
This project demonstrates how explainable AI, historical knowledge, and local LLMs can be combined to create trustworthy predictive maintenance systems suitable for real industrial deployment.

**This is a decision model — not an autonomous controller.**
