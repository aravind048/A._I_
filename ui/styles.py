import streamlit as st


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --acmf-navy: #17233c;
            --acmf-blue: #315c91;
            --acmf-blue-soft: #edf4fb;
            --acmf-green: #16805b;
            --acmf-green-soft: #eaf8f1;
            --acmf-red: #b54747;
            --acmf-red-soft: #fff0f0;
            --acmf-text: #1e293b;
            --acmf-muted: #64748b;
            --acmf-border: #d9e1ea;
            --acmf-bg: #f6f8fb;
        }

        .stApp {
            background: var(--acmf-bg);
            color: var(--acmf-text);
        }

        /* ---------------------------------------------------------
           Theme-safe typography
           ---------------------------------------------------------
           Do NOT apply one text colour to the entire .stApp.
           Streamlit renders the sidebar and main page with different
           backgrounds. A global colour rule makes sidebar text disappear
           when the sidebar uses a dark theme.
        */

        /* Main content */
        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] h4,
        [data-testid="stMain"] p,
        [data-testid="stMain"] label,
        [data-testid="stMain"] li,
        [data-testid="stMain"] div[data-testid="stCaptionContainer"] {
            color: var(--acmf-text);
        }

        [data-testid="stMain"] [data-testid="stMetricLabel"] {
            color: var(--acmf-muted) !important;
        }

        [data-testid="stMain"] [data-testid="stMetricValue"] {
            color: var(--acmf-text) !important;
        }

        [data-testid="stMain"] [data-testid="stMetricDelta"] {
            color: var(--acmf-muted) !important;
        }

        [data-testid="stMain"] [data-testid="stProgress"] p {
            color: var(--acmf-muted) !important;
        }

        /* Sidebar: explicitly use light text because the current
           Streamlit sidebar is dark. */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] div[data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stCaption {
            color: #f8fafc !important;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(248, 250, 252, 0.18) !important;
        }

        [data-testid="stSidebar"] [data-testid="stButton"] > button {
            color: #f8fafc !important;
            background: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid rgba(248, 250, 252, 0.22) !important;
        }

        [data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
            background: rgba(255, 255, 255, 0.10) !important;
            border-color: rgba(248, 250, 252, 0.38) !important;
        }

        .block-container {
            max-width: 1450px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            background:
                linear-gradient(135deg, #17233c 0%, #284b76 100%);
            border-radius: 22px;
            padding: 2.2rem 2.4rem;
            margin-bottom: 1.5rem;
            color: white;
            box-shadow: 0 12px 30px rgba(23, 35, 60, 0.18);
        }

        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.13em;
            font-size: 0.72rem;
            font-weight: 700;
            opacity: 0.72;
            margin-bottom: 0.55rem;
        }

        .hero-title {
            font-size: 2.05rem;
            font-weight: 800;
            line-height: 1.15;
        }

        .hero-subtitle {
            max-width: 900px;
            margin-top: 0.65rem;
            font-size: 0.98rem;
            line-height: 1.6;
            opacity: 0.86;
        }

        .pipeline-card {
            min-height: 112px;
            border: 1px solid var(--acmf-border);
            background: white;
            border-radius: 15px;
            padding: 0.85rem;
            transition: 0.2s ease;
        }

        .pipeline-done {
            border-color: #9ed8c1;
            background: var(--acmf-green-soft);
        }

        .pipeline-icon {
            font-size: 1.25rem;
            font-weight: 800;
            color: var(--acmf-green);
        }

        .pipeline-step {
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            color: var(--acmf-text);
            margin-top: 0.35rem;
        }

        .pipeline-desc {
            font-size: 0.72rem;
            color: var(--acmf-muted);
            margin-top: 0.3rem;
            line-height: 1.35;
        }

        .state-card {
            background: #ffffff;
            border: 1px solid #d9e1ea;
            border-radius: 16px;
            padding: 18px 20px;
            min-height: 92px;
            box-sizing: border-box;
        }

        .state-label {
            font-size: 0.82rem;
            line-height: 1.2;
            margin-bottom: 12px;
        }

        .state-value {
            font-size: 1.55rem;
            line-height: 1.15;
            font-weight: 500;
            white-space: nowrap;
            overflow: visible;
        }

        .state-value-small {
            font-size: 1.15rem;
        }

        .panel {
            background: white;
            border: 1px solid var(--acmf-border);
            border-radius: 18px;
            padding: 1.2rem 1.35rem;
            min-height: 150px;
            box-shadow: 0 5px 16px rgba(15, 23, 42, 0.04);
        }

        .decision-badge {
            display: inline-block;
            padding: 0.55rem 0.85rem;
            border-radius: 999px;
            background: var(--acmf-blue-soft);
            color: var(--acmf-blue);
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            border: 1px solid #c8d9eb;
        }

        .history-card {
            background: white;
            border: 1px solid var(--acmf-border);
            border-left: 5px solid #94a3b8;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.75rem;
        }

        .history-correct {
            border-left-color: var(--acmf-green);
        }

        .history-wrong {
            border-left-color: var(--acmf-red);
        }

        .history-action {
            margin-left: 0.7rem;
            font-size: 0.72rem;
            font-weight: 800;
            color: var(--acmf-blue);
            letter-spacing: 0.04em;
        }

        .history-row {
            display: flex;
            justify-content: space-between;
            border-top: 1px solid #eef2f6;
            margin-top: 0.55rem;
            padding-top: 0.45rem;
            font-size: 0.78rem;
            color: var(--acmf-muted);
        }

        .takeaway {
            margin-top: 1.5rem;
            padding: 1rem 1.2rem;
            border-radius: 15px;
            background: var(--acmf-navy);
            color: white;
            text-align: center;
            font-size: 0.9rem;
        }

        [data-testid="stMetric"] {
            background: white;
            border: 1px solid var(--acmf-border);
            border-radius: 15px;
            padding: 0.75rem 1rem;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.035);
        }

        div[data-testid="stButton"] > button {
            border-radius: 10px;
            min-height: 2.65rem;
            font-weight: 700;
        }

        textarea, input {
            border-radius: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
