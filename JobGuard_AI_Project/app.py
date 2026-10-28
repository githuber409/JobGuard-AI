import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier

st.set_page_config(page_title="JobGuard AI", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.main {background: #f7f9fc;}
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(120deg,#10233f,#1f5684); color: white; margin-bottom: 1.2rem;}
.hero h1 {color:white; margin-bottom:0.35rem;}
.hero p {color:#e1edf8; font-size:1.05rem;}
.small-note {font-size:0.88rem;color:#5c6675;}
</style>
<div class="hero">
<h1>🛡️ JobGuard AI</h1>
<p>Fake Job Posting Detector · NLP + Machine Learning</p>
<span>Screen job descriptions for patterns associated with fraudulent listings.</span>
</div>
""", unsafe_allow_html=True)

st.warning("Educational prototype only. A prediction is not proof that a job is a scam. Verify employers independently and never pay to obtain a job.")

DEMO_ROWS = [
("Junior Python Developer","We are looking for a junior Python developer to join our engineering team. Candidates should know Python, Git and basic SQL. Apply through the careers page. Interviews include a technical discussion.","Full-time role with mentorship, annual leave and standard benefits. Applicants should have a degree or equivalent project experience.",0),
("Work from home data entry","Earn $5000 weekly from home with no interview and no experience. Send a registration fee today to reserve your guaranteed position. Limited vacancies, act immediately.","Pay a refundable processing fee and buy our starter kit before your appointment.",1),
("Cybersecurity Intern","Our security team is seeking an intern to assist with log analysis, vulnerability management and documentation. Basic networking and Linux knowledge preferred. Submit a CV through the company careers portal.","Paid internship, supervisor mentoring, structured learning plan and defined project goals.",0),
("Secret shopper payment agent","URGENT! Get rich quickly. We will send a cheque; deposit it and transfer part of the money back. No skills needed. Contact a personal email address now.","Immediate selection. Upfront onboarding fee required. Guaranteed high income.",1),
("Customer Support Associate","The support team helps customers resolve account and product questions. Applicants should have clear written communication and be comfortable with shift work. Apply on our official website.","Training provided, health coverage where eligible and paid time off.",0),
("International funds transfer assistant","Work only 1 hour per day and earn huge income. No background check. Use your personal bank account to receive and forward payments. Pay a small verification charge.","Large guaranteed income with no contract or formal interview.",1),
("Graduate Software Trainee","We are hiring graduate trainees for software testing and application support. Familiarity with programming fundamentals is helpful. Shortlisted candidates will complete two interview rounds.","Training, feedback sessions and compensation described in the formal offer.",0),
("Online recruitment coordinator","Congratulations, you are selected without applying. Share identity documents and pay for a mandatory laptop delivery before receiving your contract.","Immediate joining; payment required before employment documents are issued.",1),
("IT Help Desk Intern","Assist the IT team with ticket triage, device setup and basic troubleshooting. Applicants should understand operating systems and networking fundamentals.","Part-time internship with a named supervisor and a documented selection process.",0),
("Crypto task optimizer","Earn guaranteed daily profits by completing simple tasks. Deposit money to unlock higher commissions. No interview, no qualifications and no contract needed.","Pay a deposit to activate your account and unlock your earnings.",1),
("Marketing Analyst","Analyse campaign metrics, prepare weekly reports and collaborate with the marketing team. Experience with spreadsheets is useful. Apply using the official company portal.","Role details, reporting line and salary band will be shared during the recruitment process.",0),
("Remote assistant immediate hire","We hire everyone instantly. Send your bank details to receive your joining bonus. Pay a small refundable fee to verify your application.","No interview required. Guaranteed high salary regardless of experience.",1),
]

def normalize_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()

def combine_fields(df):
    fields = ["title", "company_profile", "description", "requirements", "benefits",
              "employment_type", "required_experience", "required_education", "industry", "function", "location"]
    present = [c for c in fields if c in df.columns]
    if not present:
        return pd.Series([""] * len(df), index=df.index)
    return df[present].fillna("").astype(str).agg(" ".join, axis=1).map(normalize_text)

@st.cache_resource(show_spinner=False)
def fit_model(texts_tuple, labels_tuple):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1,2),
                                  max_features=50000, sublinear_tf=True, min_df=1)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])
    model.fit(list(texts_tuple), list(labels_tuple))
    return model

@st.cache_data(show_spinner=False)
def get_demo_data():
    return pd.DataFrame(DEMO_ROWS, columns=["title","description","requirements","fraudulent"]).assign(
        company_profile="", benefits=lambda d: d["requirements"], employment_type="", required_experience="",
        required_education="", industry="", function="", location=""
    )

def warning_signals(text):
    checks = {
        "Upfront fee or payment requested": r"\b(fee|pay|payment|deposit|processing charge|starter kit|verification charge)\b",
        "Unrealistic or guaranteed earnings": r"\b(guaranteed|earn \$?\d+|huge income|get rich|daily profits|high salary)\b",
        "Urgency or pressure language": r"\b(urgent|act immediately|limited vacancies|immediate hire|today only)\b",
        "No interview or qualifications": r"\b(no interview|without interview|no experience|no qualifications|hire everyone)\b",
        "Requests to move money or use bank account": r"\b(transfer money|forward payments|personal bank account|deposit it|bank details)\b",
        "Unclear or informal application route": r"\b(personal email|contact .* now|send .* whatsapp)\b",
    }
    return [label for label, pattern in checks.items() if re.search(pattern, text, re.I)]

with st.sidebar:
    st.header("Project controls")
    st.write("Choose how the model gets its training data.")
    source = st.radio(
    "Training source",
    ["Full dataset on server", "Built-in demo examples", "Upload a CSV dataset"],
)
    uploaded = None

if source == "Full dataset on server":
    import os

    from pathlib import Path
    local_csv = Path(__file__).resolve().parent / "fake_job_postings.csv"

    if not os.path.exists(local_csv):
        st.error("Dataset file not found on the server.")
        st.stop()

    try:
        df = pd.read_csv(local_csv)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    if "fraudulent" not in df.columns:
        st.error("The CSV needs a `fraudulent` column.")
        st.stop()

    df["fraudulent"] = pd.to_numeric(
        df["fraudulent"], errors="coerce"
    )
    df = df[df["fraudulent"].isin([0, 1])].copy()

    if len(df) < 10 or df["fraudulent"].nunique() < 2:
        st.error("The dataset must contain at least 10 labelled rows with both classes.")
        st.stop()

    data_label = "full JobGuard AI dataset"

elif source == "Built-in demo examples":
    df = get_demo_data()
    data_label = "small illustrative demo set"

else:
    if uploaded is None:
        st.info("Upload a labelled CSV to train the model on your dataset.")
        st.stop()

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    if "fraudulent" not in df.columns:
        st.error("The CSV needs a `fraudulent` column: 0 for legitimate and 1 for fake.")
        st.stop()

    df["fraudulent"] = pd.to_numeric(
        df["fraudulent"], errors="coerce"
    )
    df = df[df["fraudulent"].isin([0, 1])].copy()

    if len(df) < 10 or df["fraudulent"].nunique() < 2:
        st.error("Please upload at least 10 labelled rows containing both classes (0 and 1).")
        st.stop()

    data_label = "uploaded dataset"


texts = combine_fields(df).map(normalize_text)
labels = df["fraudulent"].astype(int)
usable = texts.str.len() > 0
texts, labels = texts[usable], labels[usable]
if len(texts) < 4 or labels.nunique() < 2:
    st.error("Not enough usable labelled examples to train. Check the dataset.")
    st.stop()

# Split only when enough examples exist in each class; tiny demo set is for interface demonstration.
counts = labels.value_counts()
can_split = len(labels) >= 10 and counts.min() >= 2
if can_split:
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    model = fit_model(tuple(X_train), tuple(y_train))
    test_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, test_pred),
        "Precision (fake class)": precision_score(y_test, test_pred, zero_division=0),
        "Recall (fake class)": recall_score(y_test, test_pred, zero_division=0),
        "F1 (fake class)": f1_score(y_test, test_pred, zero_division=0),
    }
    model = fit_model(tuple(texts), tuple(labels))
else:
    model = fit_model(tuple(texts), tuple(labels))
    metrics = None

tab_check, tab_data, tab_about = st.tabs(["🔎 Check a job", "📊 Dataset & evaluation", "ℹ️ How it works"])

with tab_check:
    st.subheader("Analyse a job posting")
    st.write(
        "Choose a dataset example to load automatically, "
        "or enter your own job posting below."
    )

    # Keep a manageable selection containing both classes.
    sample_df = df.loc[texts.index].groupby(
        "fraudulent", group_keys=False
    ).head(50).sort_index()

    def load_selected_example():
        row = sample_df.loc[st.session_state["sample_row_index"]]

        st.session_state["job_title"] = normalize_text(
            row.get("title", "")
        )
        st.session_state["job_company"] = normalize_text(
            row.get("company_profile", "")
        )
        st.session_state["job_description"] = normalize_text(
            row.get("description", "")
        )

        parts = [
            normalize_text(row.get("requirements", "")),
            normalize_text(row.get("benefits", "")),
        ]
        st.session_state["job_requirements"] = " ".join(
            part for part in parts if part
        )

    st.markdown("**Try an example from your dataset**")

    selected_idx = st.selectbox(
        "Choose a job posting",
        options=sample_df.index.tolist(),
        format_func=lambda idx: (
            f"{normalize_text(sample_df.loc[idx].get('title', 'Untitled'))[:65]} "
            f"— {'FAKE' if int(sample_df.loc[idx, 'fraudulent']) == 1 else 'LEGITIMATE'}"
        ),
        key="sample_row_index",
    )

    st.button(
        "Load selected example",
        on_click=load_selected_example,
        use_container_width=True,
    )

    st.divider()
    st.markdown("**Or enter a job manually**")

    title = st.text_input(
        "Job title",
        placeholder="e.g., Junior Cybersecurity Analyst",
        key="job_title",
    )
    company = st.text_input(
        "Company profile (optional)",
        placeholder="What does the company do?",
        key="job_company",
    )
    description = st.text_area(
        "Job description",
        height=180,
        placeholder="Paste the full job description...",
        key="job_description",
    )
    requirements = st.text_area(
        "Requirements / benefits (optional)",
        height=100,
        placeholder="Experience, salary, benefits, application steps...",
        key="job_requirements",
    )

    if st.button("Analyse posting", type="primary", use_container_width=True):
        user_text = " ".join([title, company, description, requirements]).strip()
        if len(user_text) < 25:
            st.warning("Please enter a job title and a more detailed description.")
        else:
            pred = int(model.predict([user_text])[0])
            probability = model.predict_proba([user_text])[0]
            classes = list(model.named_steps["classifier"].classes_)
            fake_prob = float(probability[classes.index(1)]) if 1 in classes else 0.0
            signals = warning_signals(user_text)
            left, right = st.columns(2)
            with left:
                if pred == 1:
                    st.error("Model result: Potentially fraudulent")
                else:
                    st.success("Model result: More similar to legitimate examples")
                st.metric("Model-estimated fake likelihood", f"{fake_prob:.1%}")
            with right:
                st.write("**Possible warning signs in the text**")
                if signals:
                    for signal in signals:
                        st.write(f"⚠️ {signal}")
                else:
                    st.write("No listed keyword pattern was detected. This does not prove the posting is safe.")
            st.caption("The percentage is a model score, not a verified real-world probability. Results depend on the training dataset and may be wrong.")
            st.markdown("**Before applying:** verify the company's official website and recruiter email, search for the job on the official careers page, and do not pay fees to obtain employment.")

with tab_data:
    st.subheader("Dataset overview")
    c1,c2,c3 = st.columns(3)
    c1.metric("Usable postings", f"{len(texts):,}")
    c2.metric("Legitimate examples", f"{int((labels==0).sum()):,}")
    c3.metric("Fake examples", f"{int((labels==1).sum()):,}")
    st.caption(f"Training source: {data_label}.")
    chart_df = pd.DataFrame({"Class":["Legitimate (0)","Fake (1)"],"Count":[int((labels==0).sum()),int((labels==1).sum())]})
    st.bar_chart(chart_df.set_index("Class"))
    if metrics:
        st.subheader("Held-out test metrics")
        cols = st.columns(4)
        for col, (name, val) in zip(cols, metrics.items()):
            col.metric(name, f"{val:.1%}")
        st.caption("Metrics are measured on a held-out split from the uploaded dataset. Performance on new websites may differ.")
        cm = confusion_matrix(y_test, test_pred, labels=[0,1])
        st.write("Confusion matrix (rows = actual, columns = predicted; class order 0, 1)")
        st.dataframe(pd.DataFrame(cm, index=["Actual legitimate","Actual fake"], columns=["Predicted legitimate","Predicted fake"]), use_container_width=True)
    else:
        st.info("The built-in examples are intentionally small for demonstrating the interface. Do not treat demo metrics or predictions as a validated model. Upload a larger labelled dataset for meaningful evaluation.")
    st.dataframe(df.head(10), use_container_width=True)

with tab_about:
    st.subheader("Machine-learning pipeline")
    st.markdown("""
1. **Data preprocessing:** missing text fields are replaced with empty strings; relevant job fields are joined into one text document.
2. **Feature extraction:** TF-IDF represents words and two-word phrases numerically. Terms that distinguish postings receive more weight than common words.
3. **Supervised classification:** Logistic Regression learns patterns from examples labelled `0` (legitimate) and `1` (fraudulent).
4. **Evaluation:** a stratified train/test split is used when enough labelled examples are available. Precision, recall and F1 help assess the fake class.
5. **Explainability aid:** a separate keyword scan highlights possible warning signs. These are clues, not proof and not the same thing as the ML prediction.
""")
    st.markdown("**Syllabus mapping:** Pandas/NumPy and preprocessing (Unit 1, Experiment 1); EDA and visualization (Experiment 2); supervised learning (Experiment 3); classification (Experiment 4); NLP concepts (Unit 3).")
    st.markdown("**Known limitations:** a small or biased dataset can misclassify jobs; keyword rules can miss scams or flag legitimate postings; scammers change their language. Always verify independently.")
