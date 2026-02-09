# LLM-Assisted PRA COREP Reporting Assistant (Prototype)

## Overview

UK banks subject to the PRA Rulebook must submit COREP regulatory returns that accurately reflect capital, risk exposures, and prudential metrics using complex EBA/PRA reporting templates. Preparing these returns is labour-intensive and error-prone because analysts must interpret dense regulatory text and translate it into precise COREP data fields.

This project implements a **prototype LLM-assisted regulatory reporting assistant** focused on a **well-scoped subset of COREP** (Own Funds – CA1).
The system demonstrates **end-to-end behaviour**:

> **Natural-language question → regulatory text retrieval → structured LLM output → populated COREP template extract → validation + audit trail**

The design prioritises **traceability, reproducibility, and guardrails**, rather than free-form LLM output.

---

## Scope (Deliberately Constrained)

This is a **prototype**, not a full COREP engine.

Implemented scope:

* COREP **Own Funds (CA1)** template only
* Limited set of CA1 fields (CET1, deductions, Tier 1, Tier 2, Total Own Funds)
* Offline ingestion of PRA Rulebook / COREP instruction text
* Retrieval-augmented generation with strict schema enforcement

Out of scope (by design):

* Full COREP taxonomy coverage
* XBRL generation
* Direct regulatory submission
* Automated reconciliation with bank source systems

---

## Key Features

### 1. Retrieval-Augmented Regulatory Reasoning

* Regulatory text is ingested, chunked, embedded, and indexed using FAISS
* Queries retrieve the **most relevant rulebook / instruction passages**
* LLM output is grounded strictly in retrieved evidence

### 2. Structured LLM Output (Schema-Locked)

* LLM outputs **JSON only**, validated against a Pydantic schema
* Computed values (e.g. CET1 after deductions) must satisfy consistency rules
* Invalid JSON or schema violations trigger automatic retries

### 3. COREP Template Mapping

* Structured output is mapped into a **human-readable COREP CA1 extract**
* Output mirrors real COREP row/column identifiers (e.g. `CA1.r040.c010`)

### 4. Validation Guardrails

* Numerical consistency checks (e.g. Tier 1 = CET1 + AT1)
* Sign and deduction sanity checks
* Validation issues are explicitly reported (ERROR / WARN)

### 5. Audit Trail & Citations

* **Every populated COREP field includes citations**
* Citations must reference retrieved regulatory chunks only
* Citation coverage is measured and enforced
* Audit log clearly shows *which rule paragraphs justify each field*

### 6. Reproducible Runs

* Each execution can be saved as a **run bundle**:

  * Structured JSON
  * Template extract (CSV)
  * Validation report
  * Audit log
  * Retrieval evidence
  * Index manifest snapshot

This enables review, replay, and inspection without rerunning the system.

---

## High-Level Architecture

```
User Question + Scenario
        |
        v
CLI (corep ask / corep run)
        |
        v
Retriever (FAISS + embeddings)
        |
        v
LLM (MOCK or HF Router)
  - schema enforcement
  - citation repair loop
        |
        v
Validation Engine
        |
        v
COREP Template Renderer
        |
        v
Audit Log + Run Artifacts
```

---

## Technology Stack

* **Python 3.10+**
* **Typer** (CLI)
* **Pydantic** (schema enforcement)
* **Sentence Transformers** (embeddings)
* **FAISS** (vector search)
* **Hugging Face Router** (real LLM option)
* **Rich** (CLI output)

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e .
```

---

## Configuration

Create `.env`:

```env
LLM_MODE=MOCK   # or HF_ROUTER

# Retrieval
INDEX_DIR=./data/index
TOP_K=5

# LLM (only required for HF_ROUTER)
HF_API_KEY=your_key_here
HF_MODEL=HuggingFaceH4/zephyr-7b-beta:featherless-ai
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=900
LLM_RETRIES=2

# Audit guardrails
REQUIRE_PER_FIELD_CITATIONS=true
MIN_CITATION_COVERAGE=1.0
```

---

## Typical Workflow

### 1. Ingest Regulatory Text

```bash
corep ingest -i ./data/raw -o ./data/processed/chunks.jsonl
```

### 2. Build Retrieval Index

```bash
corep index -c ./data/processed/chunks.jsonl --index-dir ./data/index
```

### 3. Ask a Question (Interactive)

```bash
corep ask \
  -q "How should we report CET1 deductions for software intangibles?" \
  -s '{"reporting_date":"2026-01-31","currency":"GBP","entity":{"legal_entity_id":"BANK1","consolidation_level":"SOLO"},"inputs":{"cet1_before_deductions":120,"intangibles_deduction":8,"dta_future_profit_deduction":5,"at1":10,"t2":15}}'
```

### 4. Run and Save Audit-Ready Artifacts

```bash
corep run \
  -q "How should we report CET1 deductions for software intangibles?" \
  -s '{"reporting_date":"2026-01-31","currency":"GBP","entity":{"legal_entity_id":"BANK1","consolidation_level":"SOLO"},"inputs":{"cet1_before_deductions":120,"intangibles_deduction":8,"dta_future_profit_deduction":5,"at1":10,"t2":15}}' \
  --out ./runs
```

---

## Output Artifacts (Example)

```
runs/
  CA1/
    20260209T143355Z/
      structured.json          # schema-validated LLM output
      template_extract.csv     # human-readable COREP extract
      audit_log.json           # field-level citations
      retrieved.json           # retrieved regulatory text
      validation.json          # guardrail results
      manifest_snapshot.json   # index metadata used
      summary.json             # run overview
```

---

## Design Principles

* **Determinism over creativity**
  LLMs are constrained by schema, citations, and validations.

* **Traceability first**
  Every number is explainable via retrieved regulatory text.

* **Defensive architecture**
  Validation and citation checks exist even when the LLM behaves correctly.

* **Prototype honesty**
  Scope is intentionally narrow to demonstrate correctness, not completeness.

---

## Intended Use

This project is intended as:

* A **technical prototype** for regulatory reporting assistance
* A **screening submission** demonstrating system design, not regulatory advice
* A foundation that could be extended to additional COREP templates

