import streamlit as st
from model_loader import load_bert_model
from model_utils import compare_skills
import torch

# Load BERT model once
state_dict = torch.load('bert_model.pt', map_location='cpu')
model = load_bert_model('bert_model.pt', device='cpu')

# Streamlit UI
st.title("Resume vs Job Skills Similarity Checker")

employee_skills = st.text_area("Enter Candidate/Resume Skills")
job_skills = st.text_area("Enter Job Description/Required Skills")

if st.button("Compare"):
    if employee_skills.strip() == "" or job_skills.strip() == "":
        st.warning("Please enter both candidate and job skills!")
    else:
        score = compare_skills(employee_skills, job_skills, device='cpu')
        st.success(f"Similarity Score: {score:.4f}")
