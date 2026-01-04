# 🔥 Explainable Thermal-Driven Predictive Maintenance Decision Copilot

An end-to-end **AI decision-support system** for industrial predictive maintenance that combines **thermal image analysis**, **retrieval-augmented reasoning (RAG)**, **local LLM inference**, and **explicit safety & risk guardrails**.

This project is designed for **real factories**, focusing on:
- Explainability
- Safety
- Human-in-the-loop decision making

---

## 📌 One-Line Summary
Instead of blindly predicting failures, this system **detects thermal anomalies, reasons using historical evidence, applies safety guardrails, and decides whether to recommend or escalate actions to humans**.

---

## 🧱 System Architecture

Thermal Image
↓
Thermal CV Anomaly Detection
↓
Structured Anomaly Features
↓
Engineering Fault Interpretation
↓
RAG (Historical Maintenance Logs)
↓
Local LLM Reasoning (Mistral-7B)
↓
PHASE 2: Safety Guardrails & Risk Scoring
↓
PHASE 3: Temporal Trend Analysis
↓
Human-in-the-Loop Decision Support


---

## ⚙️ Tech Stack

- Computer Vision: OpenCV, NumPy
- LLM: Mistral-7B (GGUF) via llama.cpp (local, offline)
- RAG: In-memory vector retrieval of maintenance logs
- UI: Streamlit
- Guardrails: Custom safety + confidence validation
- Trend Analysis: Lightweight temporal reasoning
- Runtime: Fully local (no cloud, no APIs)

---

# 🟢 PHASE 1 — Explainable Fault Understanding

### Goal
Answer:
> **“What is happening to the machine right now?”**

### What Happens

#### 1. Thermal Image Processing
- Thermal image converted to grayscale heatmap
- Hotspots detected statistically (not ML black box)

#### 2. Anomaly Feature Extraction
Extracted features:
- Mean temperature
- Maximum temperature
- Temperature delta
- Hotspot count
- Severity score

These features are **explicit, inspectable, and explainable**.

#### 3. Engineering Interpretation
Rule-based logic maps features → fault hypotheses:
- Bearing overheating
- Bearing wear
- Lubrication degradation

This mimics how a **maintenance engineer reasons**, not a neural black box.

#### 4. RAG – Historical Maintenance Retrieval
- Retrieves similar past incidents
- Provides:
  - Failure modes
  - Actions taken
  - Downtime
  - Repair costs
- Grounds the LLM in **real evidence**

#### 5. Local LLM Reasoning
Using Mistral-7B (offline):
- Explains failure mode
- Recommends safe next steps
- Estimates downtime & cost
- Outputs confidence score

---

### Phase 1 Example Output
Failure Mode: Bearing wear
Recommended Action: Inspect bearing and prepare replacement
Downtime: 2–6 hours
Cost: $300–$1200
Confidence: 0.85

---

# 🔵 PHASE 2 — Safety Guardrails & Risk Governance

Phase 2 answers:
> **“Is it safe to act on this recommendation?”**

This is the **core differentiator** of the project.

---

## 1️⃣ Guardrails Layer

### Why Guardrails?
LLMs can:
- Hallucinate
- Output invalid numbers
- Sound confident when uncertain

### What Guardrails Do
- Robustly parse LLM output
- Clamp confidence to [0–1]
- Block unsafe recommendations
- Force human review when needed

If confidence is low:
> The system **refuses automation** and escalates to humans.

---

## 2️⃣ Decision Safety Engine

Evaluates:
- Evidence strength (from RAG)
- Signal consistency
- Action safety

Example output:
action_safe: true
evidence_strength: strong
escalate_to_human: false

---

## 3️⃣ Risk-Aware Decision Scoring

Explicitly computes:
- Severity score
- Impact score
- Uncertainty
- Risk score
- Risk level: SAFE / CAUTION / CRITICAL

Example:
risk_score: 6.9
risk_level: CAUTION

Risk is **visible, auditable, and explainable**.

---

# 🟣 PHASE 3 — Temporal Trend Analysis

Answers:
> **“Is this problem getting worse over time?”**

### What It Adds
- Tracks anomaly features over time
- Detects worsening / stable / improving trends
- Refuses to hallucinate trends if data is insufficient

Example:
trend: insufficient_data
urgency: low

---

## 🧠 Why This Project Stands Out

| Typical Project | This Copilot |
|----------------|-------------|
| Predicts failure | Governs decisions |
| Black-box ML | Explainable reasoning |
| One-shot output | Temporal awareness |
| No safety layer | Explicit guardrails |
| Blind automation | Human-in-the-loop |

---

## 🏭 Real Factory Workflow Mapping

1. Operator captures thermal image
2. Copilot detects anomaly
3. Retrieves similar past incidents
4. Explains fault reasoning
5. Applies safety & risk checks
6. Analyzes trend
7. Recommends or escalates action

---

## ⚠️ Disclaimer

This system is:
- ❌ NOT an autonomous controller
- ✅ A decision-support copilot
- Designed to assist human engineers

---
