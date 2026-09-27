from interaction.question_handler import check_answer
from evaluation.effectiveness import evaluate_intervention
from llm.ollama_client import generate_intervention
from llm.prompt_builder import build_intervention_prompt
from rag.retrieval import retrieve
from rag.query_builder import build_retrieval_query
from decision.decision_engine import choose_action
from learner.learner_model import LearnerModel
from pathlib import Path
import sys

# ------------------------------------------------------------
# Make the ACMF project root importable.
# Expected structure:
#
# ACMF/
# ├── learner/
# ├── decision/
# ├── rag/
# ├── llm/
# ├── evaluation/
# ├── interaction/
# └── acmf_demo/
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ACMFDemoEngine:
    """
    UI-facing orchestration layer.

    Important:
    This class does NOT replace ACMF components.
    It only coordinates the existing components for the Streamlit demo.
    """

    CONCEPT = "references"
    ERROR_PATTERN = "reference_aliasing_misunderstanding"

    QUESTION = """a = [10, 20]
b = a
b.append(30)

print(a)"""

    EXPECTED_ANSWER = "[10,20,30]"

    def __init__(self):
        self.learner = LearnerModel("L001")
        self.learner.initialize_concept(
            self.CONCEPT,
            mastery=0.35,
        )

        self.current_cycle = 0
        self.current_action = None
        self.current_intervention = None

        # Preserve the most recently completed decision/intervention for UI display.
        self.last_action = None
        self.last_intervention = None
        self.current_query = None
        self.current_chunks = []
        self.current_prompt = None

        self.pending_learner_state = None
        self.pending_previous_effectiveness = None

        self.history = []

        self.pipeline_status = {
            "UNDERSTAND": "pending",
            "DECIDE": "pending",
            "RETRIEVE": "pending",
            "GENERATE": "pending",
            "EVALUATE": "pending",
            "UPDATE": "pending",
        }

    def get_display_state(self):
        learner_state = self.learner.get_state(
            self.CONCEPT,
            self.ERROR_PATTERN,
        )

        return {
            "mastery": learner_state["mastery"],
            "error_count": learner_state["error_count"],
            "repeated_error": learner_state["repeated_error"],
            "previous_effectiveness": (
                self.learner.get_last_intervention_effectiveness(
                    self.CONCEPT
                )
            ),
        }

    def prepare_next_cycle(self):
        """
        Runs the ACMF stages up to intervention generation.

        Flow:
            learner state
                ↓
            decision engine
                ↓
            retrieval query
                ↓
            RAG retrieval
                ↓
            prompt
                ↓
            LLM intervention
        """

        if self.current_cycle >= 3:
            raise RuntimeError(
                "The controlled three-cycle demonstration is complete. "
                "Reset the experiment to run it again."
            )

        self.current_cycle += 1

        # Reset pipeline status for the new cycle.
        self.pipeline_status = {
            "UNDERSTAND": "pending",
            "DECIDE": "pending",
            "RETRIEVE": "pending",
            "GENERATE": "pending",
            "EVALUATE": "pending",
            "UPDATE": "pending",
        }

        # 1. UNDERSTAND
        learner_state = self.learner.get_state(
            self.CONCEPT,
            self.ERROR_PATTERN,
        )

        previous_effectiveness = (
            self.learner.get_last_intervention_effectiveness(
                self.CONCEPT
            )
        )

        self.pending_learner_state = learner_state
        self.pending_previous_effectiveness = previous_effectiveness

        self.pipeline_status["UNDERSTAND"] = "done"

        # 2. DECIDE
        conceptual_error = True

        self.current_action = choose_action(
            mastery=learner_state["mastery"],
            conceptual_error=conceptual_error,
            repeated_error=learner_state["repeated_error"],
            previous_effectiveness=previous_effectiveness,
        )

        self.pipeline_status["DECIDE"] = "done"

        # 3. BUILD RETRIEVAL QUERY
        self.current_query = build_retrieval_query(
            concept=self.CONCEPT,
            error_pattern=self.ERROR_PATTERN,
            question=self.QUESTION,
            pedagogical_action=self.current_action,
        )

        # 4. RETRIEVE
        self.current_chunks = retrieve(
            self.current_query,
            top_k=3,
        )

        self.pipeline_status["RETRIEVE"] = "done"

        # 5. BUILD PROMPT
        self.current_prompt = build_intervention_prompt(
            learner_state=learner_state,
            pedagogical_action=self.current_action,
            retrieved_chunks=self.current_chunks,
        )

        # 6. GENERATE
        self.current_intervention = generate_intervention(
            self.current_prompt
        )

        self.pipeline_status["GENERATE"] = "done"

    def evaluate_answer(self, user_answer):
        """
        Completes the cycle after the learner submits an answer.

        Flow:
            learner answer
                ↓
            correctness
                ↓
            effectiveness
                ↓
            learner evidence
                ↓
            mastery update
                ↓
            intervention history
        """

        if not self.current_action:
            raise RuntimeError(
                "Run the next ACMF cycle before evaluating an answer."
            )

        if not user_answer.strip():
            raise ValueError("Please enter a learner answer.")

        correct = check_answer(
            user_answer,
            self.EXPECTED_ANSWER,
        )

        # 7. EVALUATE INTERVENTION
        effectiveness = evaluate_intervention(
            follow_up_correct=correct,
            explanation_provided=False,
            explanation_correct=False,
        )

        self.pipeline_status["EVALUATE"] = "done"

        # 8. RECORD EVIDENCE
        if correct:
            self.learner.record_evidence(
                concept=self.CONCEPT,
                correct=True,
            )
        else:
            self.learner.record_evidence(
                concept=self.CONCEPT,
                correct=False,
                error_pattern=self.ERROR_PATTERN,
            )

        # 9. UPDATE LEARNER MODEL
        mastery_before = self.pending_learner_state["mastery"]

        mastery_after = self.learner.update_knowledge(
            self.CONCEPT,
            correct,
        )

        self.learner.record_intervention(
            concept=self.CONCEPT,
            action=self.current_action,
            effectiveness=effectiveness,
        )

        self.pipeline_status["UPDATE"] = "done"

        # Store a UI-friendly cycle record.
        self.history.append(
            {
                "cycle": self.current_cycle,
                "mastery_before": mastery_before,
                "mastery_after": mastery_after,
                "error_count": self.pending_learner_state["error_count"],
                "repeated_error": self.pending_learner_state[
                    "repeated_error"
                ],
                "previous_effectiveness": (
                    self.pending_previous_effectiveness
                ),
                "action": self.current_action,
                "correct": correct,
                "effectiveness": effectiveness,
                "retrieval_query": self.current_query,
                "retrieved_chunks": self.current_chunks,
                "intervention": self.current_intervention,
            }
        )

        # Preserve the completed cycle output before clearing the pending cycle.
        self.last_action = self.current_action
        self.last_intervention = self.current_intervention

        # Clear the current intervention so the next cycle starts cleanly.
        self.current_action = None
        self.current_intervention = None

        # Preserve the most recently completed decision/intervention for UI display.
        self.last_action = None
        self.last_intervention = None
        self.current_query = None
        self.current_chunks = []
        self.current_prompt = None
        self.pending_learner_state = None
        self.pending_previous_effectiveness = None
