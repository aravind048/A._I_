from model_utils import get_skill_embeddings
from similarity import calculate_similarity
from text_utils import clean_text


def compare_skills(employee_skills: str, job_skills: str) -> float:
    """
    Compares employee skills and job skills using BERT embeddings
    and calculates similarity.

    Args:
        employee_skills (str): Employee's skills text.
        job_skills (str): Job's required skills text.

    Returns:
        float: Similarity score (0-1)
    """
    # 1️⃣ Clean the text
    cleaned_employee_skills = clean_text(employee_skills)
    cleaned_job_skills = clean_text(job_skills)

    # 2️⃣ Get embeddings (using global model + tokenizer)
    employee_embeddings = get_skill_embeddings(cleaned_employee_skills)
    job_embeddings = get_skill_embeddings(cleaned_job_skills)

    # 3️⃣ Compute similarity
    similarity_score = calculate_similarity(employee_embeddings, job_embeddings)
    return similarity_score
