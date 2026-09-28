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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ACMFDemoEngine:
    """UI-facing orchestration layer for the ACMF demonstration."""

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
        self.learner.initialize_concept(self.CONCEPT, mastery=0.35)

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
        if not self.available_topics:
            raise RuntimeError("No ACMF topics are available.")

        if concept not in self.available_topics:
            raise ValueError(
                f"Unsupported topic '{concept}'. "
                f"Choose one of: {', '.join(self.available_topics)}."
            )

        self.CONCEPT = concept

    def reset(self, concept=None):
        if concept is not None:
            self.set_topic(concept)
        self._initialize_runtime()

    @property
    def QUESTION(self):
        if self.current_question:
            return self.current_question["question"]
        return self.topic_questions[0]["question"]

    @property
    def EXPECTED_ANSWER(self):
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
                self.learner.get_last_intervention_effectiveness(self.CONCEPT)
            ),
        }

    def prepare_next_cycle(self):
        if self.current_cycle >= 3:
            raise RuntimeError(
                "The controlled three-cycle demonstration is complete. "
                "Reset the experiment to run it again."
            )

        self.current_cycle += 1

        # Cycle N uses the Nth question for the selected topic. The same
        # question is also supplied to the intervention prompt.
        question_index = min(
            self.current_cycle - 1,
            len(self.topic_questions) - 1,
        )
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

        learner_state = self.learner.get_state(
            self.CONCEPT,
            self.ERROR_PATTERN,
        )
        previous_effectiveness = (
            self.learner.get_last_intervention_effectiveness(self.CONCEPT)
        )

        self.pending_learner_state = learner_state
        self.pending_previous_effectiveness = previous_effectiveness
        self.pipeline_status["UNDERSTAND"] = "done"

        # Controlled prototype behaviour: the selected topic supplies the
        # learning context; error classification is not inferred from UI text.
        conceptual_error = True

        self.current_action = choose_action(
            mastery=learner_state["mastery"],
            conceptual_error=conceptual_error,
            repeated_error=learner_state["repeated_error"],
            previous_effectiveness=previous_effectiveness,
        )
        self.pipeline_status["DECIDE"] = "done"

        self.current_query = build_retrieval_query(
            concept=self.CONCEPT,
            error_pattern=self.ERROR_PATTERN,
            question=self.QUESTION,
            pedagogical_action=self.current_action,
        )

        self.current_chunks = retrieve(self.current_query, top_k=3)
        self.pipeline_status["RETRIEVE"] = "done"

        # Critical consistency fix: the LLM receives the exact assessment
        # question that is displayed in Learner Interaction.
        self.current_prompt = build_intervention_prompt(
            learner_state=learner_state,
            pedagogical_action=self.current_action,
            retrieved_chunks=self.current_chunks,
            question=self.QUESTION,
        )

        self.current_intervention = generate_intervention(self.current_prompt)
        self.pipeline_status["GENERATE"] = "done"

    def evaluate_answer(self, user_answer):
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

        effectiveness = evaluate_intervention(
            follow_up_correct=correct,
            explanation_provided=False,
            explanation_correct=False,
        )
        self.pipeline_status["EVALUATE"] = "done"

        self.learner.record_evidence(
            concept=self.CONCEPT,
            correct=correct,
            error_pattern=None if correct else self.ERROR_PATTERN,
        )

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

        self.current_action = None
        self.current_intervention = None
        self.current_query = None
        self.current_chunks = []
        self.current_prompt = None
        self.pending_learner_state = None
        self.pending_previous_effectiveness = None
