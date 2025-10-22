import torch
from transformers import BertModel, BertTokenizer

def load_bert_model(model_path='bert_model.pt', device='cpu'):
    """
    Load the pre-trained BERT model's state dictionary for inference.
    Returns both the model and tokenizer.
    """
    try:
        # Load base BERT architecture
        model = BertModel.from_pretrained('bert-base-uncased')

        # Load trained weights
        state_dict = torch.load(model_path, map_location=device)

        # Optional filtering (only if keys are prefixed with 'bert.')
        if any(k.startswith('bert.') for k in state_dict.keys()):
            state_dict = {k.replace('bert.', ''): v for k, v in state_dict.items() if k.startswith('bert.')}

        # Load the state dict into model
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        # Load tokenizer once
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        print("✅ BERT model and tokenizer loaded successfully on", device)
        return model, tokenizer

    except Exception as e:
        print("❌ Error loading BERT model:", str(e))
        return None, None

# For Flask integration, load globally
model, tokenizer = load_bert_model()
