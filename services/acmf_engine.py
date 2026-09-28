import json
from pathlib import Path
import sys

from interaction.question_handler import check_answer
from evaluation.effectiveness import evaluate_intervention
from llm.ollama_client import generate_intervention
from llm.prompt_builder import build_intervention_prompt
from rag.retrieval import retrieve
from rag.query_builder import build_retrieval_query
from decision.decision_engine import choose_action
from learner.learner_model import LearnerModel

# ------------------------------------------------------------
# Make the ACMF project root importable.
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ACMFDemoEngine:
    """
    UI-facing orchestration layer.

    Important:
    This class does NOT replace ACMF components.
    It only coordinates the existing components for the Streamlit demo.

    The UI supports controlled topic selection using the existing question
    bank. This is a demonstration feature, not a research experiment.
    """

    QUESTIONS_PATH = PROJECT_ROOT / "experiments" / "questions.json"
    TOPIC_ERROR_PATTERNS = {
        "variables": "variable_assignment_misunderstanding",
        "conditions": "conditional_logic_misunderstanding",
        "loops": "loop_iteration_misunderstanding",
        "references": "reference_aliasing_misunderstanding",
    }

    def __init__(self, concept="references"):
        self.questions = self._load_questions()
        self.available_topics = self._build_topic_list()
        self.set_topic(concept)
        self._initialize_runtime()

    def _load_questions(self):
        with open(self.QUESTIONS_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)

        questions = payload.get("questions", [])
        if not questions:
            raise RuntimeError("No questions found in the ACMF question bank.")

        return questions

    def _build_topic_list(self):
        return sorted({item["concept"] for item in self.questions})

    def _questions_for_topic(self):
        topic_questions = [
            item for item in self.questions
            if item["concept"] == self.CONCEPT
        ]

        if not topic_questions:
            raise ValueError(
                f"No questions are configured for topic '{self.CONCEPT}'."
            )

        return topic_questions

    def _initialize_runtime(self):
        self.topic_questions = self._questions_for_topic()
        self.ERROR_PATTERN = self.TOPIC_ERROR_PATTERNS[self.CONCEPT]

        self.learner = LearnerModel("L001")
        self.learner.initialize_concept(
            self.CONCEPT,
            mastery=0.35,
        )

        self.current_cycle = 0
        self.current_action = None
        self.current_intervention = None
        self.current_query = None
        self.current_chunks = []
        self.current_prompt = None
        self.current_question = None
        self.current_expected_answer = None

        self.last_action = None
        self.last_intervention = None
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

    def set_topic(self, concept):
        """Select one of the topics supported by the existing question bank."""
        if not self.available_topics:
            raise RuntimeError("No ACMF topics are available.")

        if concept not in self.available_topics:
            raise ValueError(
                f"Unsupported topic '{concept}'. "
                f"Choose one of: {', '.join(self.available_topics)}."
            )

        self.CONCEPT = concept

    def reset(self, concept=None):
        """Reset the demo session, optionally switching to another topic."""
        if concept is not None:
            self.set_topic(concept)

        self._initialize_runtime()

    @property
    def QUESTION(self):
        """Return the question for the current cycle/topic."""
        if self.current_question:
            return self.current_question["question"]
        return self.topic_questions[0]["question"]

    @property
    def EXPECTED_ANSWER(self):
        """Return the expected answer for the current cycle/topic."""
        if self.current_question:
            return self.current_expected_answer
        return self.topic_questions[0]["expected_answer"]

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

        # Use the existing question bank. Cycle 1/2/3 maps to the first,
        # second, and third question for the selected topic.
        question_index = min(self.current_cycle - 1, len(self.topic_questions) - 1)
        self.current_question = self.topic_questions[question_index]
        self.current_expected_answer = self.current_question["expected_answer"]

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
        # The current prototype treats the selected concept as the learning
        # context. A future version can replace this with automated error
        # classification from the learner response.
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
            self.current_expected_answer,
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

        self.history.append(
            {
                "cycle": self.current_cycle,
                "question_id": self.current_question["id"],
                "mastery_before": mastery_before,
                "mastery_after": mastery_after,
                "error_count": self.pending_learner_state["error_count"],
                "repeated_error": self.pending_learner_state["repeated_error"],
                "previous_effectiveness": self.pending_previous_effectiveness,
                "action": self.current_action,
                "correct": correct,
                "effectiveness": effectiveness,
                "retrieval_query": self.current_query,
                "retrieved_chunks": self.current_chunks,
                "intervention": self.current_intervention,
            }
        )

        self.last_action = self.current_action
        self.last_intervention = self.current_intervention

        # Clear pending cycle state. The completed decision/intervention is
        # preserved in history and by the Streamlit session state.
        self.current_action = None
        self.current_intervention = None
        self.current_query = None
        self.current_chunks = []
        self.current_prompt = None
        self.pending_learner_state = None
        self.pending_previous_effectiveness = None
