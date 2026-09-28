import streamlit as st

from services.acmf_engine import ACMFDemoEngine
from ui.styles import inject_styles

st.set_page_config(
    page_title="ACMF | Adaptive Cognitive Mentorship",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

# -----------------------------
# Session initialization
# -----------------------------
if "engine" not in st.session_state:
    st.session_state.engine = ACMFDemoEngine()

if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = st.session_state.engine.CONCEPT

if "last_completed_action" not in st.session_state:
    st.session_state.last_completed_action = None

if "last_completed_intervention" not in st.session_state:
    st.session_state.last_completed_intervention = None

if "last_completed_cycle" not in st.session_state:
    st.session_state.last_completed_cycle = None

engine = st.session_state.engine

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">M.TECH CASE STUDY • PROOF-OF-CONCEPT</div>
        <div class="hero-title">Adaptive Cognitive Mentorship Framework</div>
        <div class="hero-subtitle">
            A visual demonstration of learner-state tracking, adaptive pedagogical
            decisions, grounded intervention generation, and feedback-driven adaptation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("### 🧭 Demo Controls")

    # Once a learner starts Cycle 1, changing the topic would invalidate the
    # current learner state and intervention. Keep the topic fixed for the run.
    topic_locked = bool(engine.history or engine.current_action)

    selected_topic = st.selectbox(
        "Choose a learning topic",
        options=engine.available_topics,
        index=engine.available_topics.index(st.session_state.selected_topic),
        help="Choose a topic before starting the learner session.",
        disabled=topic_locked,
    )

    if selected_topic != st.session_state.selected_topic and not topic_locked:
        st.session_state.selected_topic = selected_topic
        st.session_state.engine = ACMFDemoEngine(selected_topic)
        st.session_state.last_completed_action = None
        st.session_state.last_completed_intervention = None
        st.session_state.last_completed_cycle = None
        st.rerun()

    if st.button("↻ Reset Current Topic", use_container_width=True):
        st.session_state.engine = ACMFDemoEngine(selected_topic)
        st.session_state.selected_topic = selected_topic
        st.session_state.last_completed_action = None
        st.session_state.last_completed_intervention = None
        st.session_state.last_completed_cycle = None
        st.rerun()

    st.markdown("---")
    st.markdown("**Current topic**")
    st.caption(selected_topic.title())

    st.markdown("**Cycles completed**")
    st.caption(f"{len(engine.history)} / 3")

    st.markdown("**Error evidence**")
    error_count = engine.get_display_state()["error_count"]
    st.caption(str(error_count))

    if engine.history:
        latest = engine.history[-1]
        st.markdown("**Latest response**")
        st.caption("Correct" if latest["correct"] else "Incorrect")

    if topic_locked:
        st.info("Topic is locked for the current learner session.")

    st.markdown("---")
    st.caption("The UI is a demonstration layer over the ACMF pipeline.")

# -----------------------------
# Progress strip
# -----------------------------
total_cycles = 3
completed_cycles = len(engine.history)
cycle_in_progress = engine.current_cycle if engine.current_action else None

progress = min(completed_cycles / total_cycles, 1.0)

if cycle_in_progress:
    progress_text = f"Cycle {cycle_in_progress} in progress • Awaiting learner response"
elif completed_cycles >= total_cycles:
    progress_text = f"Demonstration complete • {completed_cycles} of {total_cycles} cycles"
else:
    progress_text = f"{completed_cycles} of {total_cycles} cycles completed • Ready for Cycle {completed_cycles + 1}"

st.progress(progress, text=progress_text)

# -----------------------------
# Current learner state
# -----------------------------
state = engine.get_display_state()

st.markdown("### 👤 Learner State")
st.caption(
    "Current learner state before the next interaction. "
    "This state is maintained by the ACMF learner model."
)

# Topic is intentionally omitted here because it is already shown in the sidebar.
c1, c2, c3, c4 = st.columns([1, 1, 1, 1.15])

with c1:
    st.markdown(
        f"""
        <div class="state-card">
            <div class="state-label">Mastery</div>
            <div class="state-value">{state['mastery']:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="state-card">
            <div class="state-label">Error Count</div>
            <div class="state-value">{state['error_count']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    repeated_error = "YES" if state["repeated_error"] else "NO"

    st.markdown(
        f"""
        <div class="state-card">
            <div class="state-label">Repeated Error</div>
            <div class="state-value">{repeated_error}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    effectiveness = state["previous_effectiveness"] or "—"

    st.markdown(
        f"""
        <div class="state-card">
            <div class="state-label">Previous Effectiveness</div>
            <div class="state-value state-value-small">
                {effectiveness}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Pipeline state
# -----------------------------
st.markdown("### ⚙️ ACMF Pipeline")
st.caption("Each stage is executed by the underlying ACMF pipeline.")

pipeline = [
    ("UNDERSTAND", "Learner state"),
    ("DECIDE", "Pedagogical action"),
    ("RETRIEVE", "Relevant knowledge"),
    ("GENERATE", "LLM intervention"),
    ("EVALUATE", "Learner response"),
    ("UPDATE", "Learner model"),
]

cols = st.columns(len(pipeline))

for col, (step, description) in zip(cols, pipeline):
    with col:
        status = engine.pipeline_status.get(step, "pending")
        icon = "✓" if status == "done" else "○"

        st.markdown(
            f"""
            <div class="pipeline-card {'pipeline-done' if status == 'done' else ''}">
                <div class="pipeline-icon">{icon}</div>
                <div class="pipeline-step">{step}</div>
                <div class="pipeline-desc">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------
# Decision / intervention area
# -----------------------------
if engine.current_action:
    decision_title = "### 🎯 Current Pedagogical Decision"
    decision_action = engine.current_action
    decision_intervention = engine.current_intervention
    decision_note = (
        f"Selected for the learner state at the start of Cycle {engine.current_cycle}. "
        "The learner response is still pending."
    )
    decision_status = "current"

elif st.session_state.last_completed_action:
    decision_title = "### 🎯 Latest Pedagogical Decision"
    decision_action = st.session_state.last_completed_action
    decision_intervention = st.session_state.last_completed_intervention

    latest = engine.history[-1] if engine.history else None
    if latest:
        result_text = "Correct" if latest["correct"] else "Incorrect"
        decision_note = (
            f"Used during Cycle {latest['cycle']}. "
            f"Learner response: {result_text}. "
            f"Effectiveness: {latest['effectiveness']}."
        )
    else:
        decision_note = "Used during the most recently completed cycle."

    decision_status = "completed"

else:
    decision_title = "### 🎯 Pedagogical Decision"
    decision_action = None
    decision_intervention = None
    decision_note = None
    decision_status = "empty"

st.markdown(decision_title)

left, right = st.columns([0.95, 1.05], gap="large")

with left:
    with st.container(border=True):
        if decision_action:
            st.markdown(
                f'<div class="decision-badge">{decision_action}</div>',
                unsafe_allow_html=True,
            )
            st.write(decision_note)

            if decision_status == "completed" and engine.history:
                latest = engine.history[-1]
                result_text = "Correct" if latest["correct"] else "Incorrect"

                st.markdown(
                    f"""
                    <div class="history-row">
                        <span>Cycle outcome</span>
                        <strong>{result_text}</strong>
                    </div>
                    <div class="history-row">
                        <span>Effectiveness</span>
                        <strong>{latest["effectiveness"]}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info(
                "Run the next cycle to let the decision engine select an action."
            )

with right:
    with st.container(border=True):
        st.markdown("**Intervention**")

        if decision_intervention:
            st.markdown(decision_intervention)
        else:
            st.caption("The generated intervention will appear here.")

# -----------------------------
# Learner interaction
# -----------------------------
st.markdown("### ✍️ Learner Interaction")

if engine.current_action and engine.current_question:
    st.caption(
        "Answer the assessment question below. The intervention above is generated for this same question."
    )
    st.code(
        engine.QUESTION,
        language="python",
    )
else:
    st.caption("Run the next ACMF cycle to receive the next assessment question.")

answer_enabled = bool(engine.current_action)

answer = st.text_input(
    "Learner answer",
    value="",
    placeholder="Enter your answer...",
    disabled=not answer_enabled,
    key=f"learner_answer_cycle_{engine.current_cycle}",
)

if not answer_enabled:
    st.caption(
        "Run the next ACMF cycle first. The intervention will then be generated for the learner."
    )
else:
    st.caption(
        "Enter the learner's response, then evaluate it to complete this cycle."
    )

b1, b2 = st.columns([1, 1])

with b1:
    can_run_next = (
        not engine.current_action
        and len(engine.history) < total_cycles
    )

    run_label = (
        f"▶ Run Cycle {completed_cycles + 1}"
        if completed_cycles < total_cycles
        else "✓ Demonstration Complete"
    )

    if st.button(
        run_label,
        type="primary",
        use_container_width=True,
        disabled=not can_run_next,
    ):
        try:
            engine.prepare_next_cycle()
            st.rerun()
        except Exception as exc:
            st.error(f"Cycle execution failed: {exc}")

with b2:
    if st.button(
        "✓ Evaluate Learner Answer",
        use_container_width=True,
        disabled=not answer_enabled,
    ):
        try:
            st.session_state.last_completed_action = engine.current_action
            st.session_state.last_completed_intervention = engine.current_intervention
            st.session_state.last_completed_cycle = engine.current_cycle

            engine.evaluate_answer(answer)
            st.rerun()
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

# -----------------------------
# Latest cycle result
# -----------------------------
if engine.history:
    latest = engine.history[-1]
    result_text = "Correct" if latest["correct"] else "Incorrect"
    result_class = "history-correct" if latest["correct"] else "history-wrong"

    st.markdown(
        f"""
        <div class="history-card {result_class}">
            <div>
                <strong>Latest completed cycle • Cycle {latest['cycle']}</strong>
            </div>
            <div class="history-row">
                <span>Question</span>
                <strong>{latest['question_id']}</strong>
            </div>
            <div class="history-row">
                <span>Learner response</span>
                <strong>{result_text}</strong>
            </div>
            <div class="history-row">
                <span>Pedagogical action</span>
                <strong>{latest['action']}</strong>
            </div>
            <div class="history-row">
                <span>Mastery</span>
                <strong>{latest['mastery_before']:.4f} → {latest['mastery_after']:.4f}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Cycle history
# -----------------------------
st.markdown("### 📈 Interaction History")

if engine.history:
    for item in reversed(engine.history):
        result_class = "history-correct" if item["correct"] else "history-wrong"

        st.markdown(
            f"""
            <div class="history-card {result_class}">
                <div>
                    <strong>Cycle {item['cycle']}</strong>
                    <span class="history-action">{item['action']}</span>
                </div>
                <div class="history-row">
                    <span>Question</span>
                    <strong>{item['question_id']}</strong>
                </div>
                <div class="history-row">
                    <span>Mastery</span>
                    <strong>{item['mastery_before']:.4f} → {item['mastery_after']:.4f}</strong>
                </div>
                <div class="history-row">
                    <span>Response</span>
                    <strong>{'Correct' if item['correct'] else 'Incorrect'}</strong>
                </div>
                <div class="history-row">
                    <span>Effectiveness</span>
                    <strong>{item['effectiveness']}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.caption("No cycles completed yet.")

# -----------------------------
# Final takeaway
# -----------------------------
st.markdown(
    """
    <div class="takeaway">
        <strong>ACMF in one line:</strong>
        Observe → Understand → Decide → Intervene → Evaluate → Update → Adapt
    </div>
    """,
    unsafe_allow_html=True,
)
