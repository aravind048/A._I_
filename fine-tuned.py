import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import time

# --------------------------
# 🔧 Load fine-tuned model
# --------------------------
@st.cache_resource
def load_model():
    model_path = "fine-tuned-model - mnr"
    return SentenceTransformer(model_path)

model = load_model()

# --------------------------
# 🎨 Page Configuration
# --------------------------
st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --------------------------
# 🧭 App Header
# --------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        color: #555;
        font-size: 18px;
        margin-bottom: 40px;
    }
    </style>
    <div class="main-title">🤖 Smart Resume-Job Matcher</div>
    <div class="sub-title">AI-powered similarity analysis using fine-tuned Sentence Transformer</div>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# 📝 Input Section
# --------------------------
st.markdown("### 🧾 Enter Resume and Job Description")

col1, col2 = st.columns(2)

with col1:
    resume_text = st.text_area("📄 Resume Text", height=200, placeholder="Paste resume content here...")

with col2:
    job_text = st.text_area("💼 Job Description", height=200, placeholder="Paste job description here...")

# --------------------------
# 🚀 Inference Button
# --------------------------
if st.button("🚀 Analyze Match"):
    if resume_text.strip() and job_text.strip():
        with st.spinner("Analyzing similarity... 🧠"):
            time.sleep(0.5)
            resume_emb = model.encode(resume_text, convert_to_tensor=True)
            job_emb = model.encode(job_text, convert_to_tensor=True)
            similarity = util.cos_sim(resume_emb, job_emb).item()

        # --------------------------
        # 🎯 Threshold Logic
        # --------------------------
        threshold = 0.75
        label = "Select" if similarity >= threshold else "Reject"

        # --------------------------
        # 🌈 Color Coding + Emoji
        # --------------------------
        if similarity >= 0.80:
            color = "🟢"
            bar_color = "green"
            comment = "Excellent Match! 🔥"
        elif similarity >= 0.60:
            color = "🟡"
            bar_color = "gold"
            comment = "Moderate Match! ⚙️"
        else:
            color = "🔴"
            bar_color = "red"
            comment = "Low Match. Consider revising resume. 🧩"

        # --------------------------
        # 📊 Display Results
        # --------------------------
        st.markdown("### 📊 Similarity Score")

        # Animate progress bar
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)
        for i in range(int(similarity * 100)):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        progress_bar.progress(int(similarity * 100))

        st.markdown(
            f"<h3 style='text-align:center; color:{bar_color};'>Similarity: {similarity:.4f} {color}</h3>",
            unsafe_allow_html=True,
        )

        # Final verdict
        st.markdown(
            f"""
            <div style='text-align:center; font-size:22px; margin-top:10px;'>
                <b>Verdict:</b> {label} <br>
                <span style='color:#888;'>{comment}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # # Show thresholds visually
        # st.markdown("---")
        # st.markdown("#### 🎯 Match Categories")
        # st.markdown(
        #     """
        #     - 🟢 **High Match (≥ 0.80):** Strong alignment between resume and job description  
        #     - 🟡 **Medium Match (0.60–0.79):** Partial alignment, some improvement possible  
        #     - 🔴 **Low Match (< 0.60):** Weak alignment — consider adding relevant keywords
        #     """
        # )

    else:
        st.warning("⚠️ Please enter both resume and job description before analysis.")

# --------------------------
# 👣 Footer
# --------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#777;'>Built with ❤️ using Streamlit & Sentence Transformers</div>",
    unsafe_allow_html=True,
)
