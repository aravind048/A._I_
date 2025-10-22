from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

def calculate_similarity(employee_embeddings, job_embeddings):
    """
    Calculates the cosine similarity between employee and job skill embeddings.

    Args:
        employee_embeddings: Torch tensor or numpy array representing employee embedding.
        job_embeddings: Torch tensor or numpy array representing job embedding.

    Returns:
        float: Cosine similarity score (0–1)
    """
    # Convert torch tensors to numpy if needed
    if isinstance(employee_embeddings, torch.Tensor):
        employee_embeddings = employee_embeddings.detach().cpu().numpy()
    if isinstance(job_embeddings, torch.Tensor):
        job_embeddings = job_embeddings.detach().cpu().numpy()

    # Ensure embeddings are 2D
    if employee_embeddings.ndim == 1:
        employee_embeddings = employee_embeddings.reshape(1, -1)
    if job_embeddings.ndim == 1:
        job_embeddings = job_embeddings.reshape(1, -1)

    # Compute similarity
    similarity_score = cosine_similarity(employee_embeddings, job_embeddings)[0][0]
    return round(float(similarity_score), 6)
