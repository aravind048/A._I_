import torch
from similarity import calculate_similarity
from text_utils import clean_text
from model_loader import model, tokenizer  # reuse the same model/tokenizer

def load_state_dict_custom(state_dict):
    """
    Loads filtered state_dict into global model.
    """
    

    global model
    model.load_state_dict(state_dict, strict=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"⚠️ Missing keys: {missing}, Unexpected keys: {unexpected}")
    model.eval()
    model.eval()

def get_skill_embeddings(text: str, device='cpu') -> torch.Tensor:
    """
    Generates BERT embeddings for given text using the global model.

    Args:
        text (str): Cleaned, preprocessed string.
        device (str): 'cpu' or 'cuda'

    Returns:
        torch.Tensor: Sentence embedding vector (1, hidden_dim)
    """
    encoded_input = tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=128
    )

    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    model.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        # Normalize
        cls_embedding = cls_embedding / cls_embedding.norm(dim=1, keepdim=True)

    return cls_embedding

def compare_skills(employee_skills, job_skills, device='cpu'):
    cleaned_employee_skills = clean_text(employee_skills)
    cleaned_job_skills = clean_text(job_skills)

    emp_emb = get_skill_embeddings(cleaned_employee_skills, device=device)
    job_emb = get_skill_embeddings(cleaned_job_skills, device=device)

    similarity_score = calculate_similarity(emp_emb, job_emb)
    return float(similarity_score)


