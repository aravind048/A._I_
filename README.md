# ACMF Streamlit Demo — Part 5

This version adds the first complete learner interaction on top of Part 4.

Run from the ACMF project root:

```powershell
streamlit run .\acmf_demo\app.py
```

## Part 5 flow

```text
Run Cycle 1
    ↓
Understand learner state
    ↓
Decision engine
    ↓
RAG retrieval
    ↓
LLM intervention
    ↓
Learner enters answer
    ↓
Evaluate answer
    ↓
Learner model update
```

The UI is only an orchestration/presentation layer. Existing ACMF modules remain responsible for learner modeling, decision making, retrieval, intervention generation, answer checking, effectiveness evaluation, and mastery update.
