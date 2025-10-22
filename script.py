from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
sentence = "GenAI is revolutionizing how we build intelligent systems."
embedding = model.encode(sentence)

# Check type and shape
import numpy as np
embedding_np = np.array(embedding)
print("Shape:", embedding_np.shape)
