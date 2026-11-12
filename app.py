import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="ASD Traits Predictor", page_icon="🧩", layout="wide")

BASE = Path(__file__).parent


# AQ-10 items, worded so that "Yes" = the trait-positive answer (score 1 in the dataset)
AQ_QUESTIONS = [
    "I often notice small sounds that other people do not",
    "I focus on small details rather than the whole picture",
    "I find it hard to do more than one thing at once",
    "I find it hard to switch back to a task after being interrupted",
    "I find it hard to read between the lines when someone is talking",
    "I can't tell when the person I'm talking to is getting bored",
    "When reading a story, I find it hard to work out the characters' intentions",
    "I like collecting information about categories of things (cars, birds, trains...)",
    "I find it hard to tell what someone is thinking or feeling from their face",
    "I find it hard to work out people's intentions",
]

st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-top: 2.5rem;}
    div[data-testid="stForm"] {border: none; padding: 0;}
    .card-title {font-size: 1.05rem; font-weight: 600; margin-bottom: .25rem;}
    div[data-testid="stFormSubmitButton"] button {
        width: 100%; min-height: 3.6rem; background: #ff4b4b; color: white; border: none;
        font-weight: 600; padding: .7rem 0; border-radius: 10px;
    }
    div[data-testid="stFormSubmitButton"] button p {
        font-size: 1.5rem; font-weight: 700; letter-spacing: .05em;
    }
    div[data-testid="stFormSubmitButton"] button:hover {background: #e03e3e; color: white;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    with open(BASE / "best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE / "encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open(BASE / "model_config.json") as f:
        config = json.load(f)   # model name, decision threshold, feature order
    return model, encoders, config


def options_for(encoders, col):
    """Return {display label: raw value the encoder knows}."""
    return {str(c).strip(): c for c in encoders[col].classes_}


try:
    model, encoders, config = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Put best_model.pkl, encoders.pkl and model_config.json next to app.py.")
    st.stop()

FEATURES, THRESHOLD = config["features"], config["threshold"]

st.title("Autism Spectrum Disorder (ASD) Traits Predictor")
st.caption(
    "A screening-support tool that uses a trained machine learning model on demographic, "
    "clinical-history and AQ-10 behavioural answers. It is not a medical diagnosis."
)
st.divider()

gender_opts = {"Male": "m", "Female": "f"}
ethnicity_opts = options_for(encoders, "ethnicity")
country_opts = options_for(encoders, "contry_of_res")
relation_opts = {"Self": "Self", "Parent / relative / other": "Others"}
yes_no = ["Yes", "No"]

with st.form("asd_form"):
    left, right = st.columns(2, gap="large")

    with left.container(border=True):
        st.markdown('<div class="card-title">Demographics</div>', unsafe_allow_html=True)
        age = st.number_input("Age (years)", min_value=2, max_value=100, value=None, step=1,
                              placeholder="Enter age")
        gender = st.radio("Gender", list(gender_opts), index=None, horizontal=True)
        ethnicity = st.selectbox("Ethnicity", list(ethnicity_opts), index=None,
                                 placeholder="Choose an option")
        country = st.selectbox("Country of residence", list(country_opts), index=None,
                               placeholder="Choose an option")
        relation = st.radio("Who is completing this test?", list(relation_opts), index=None,
                            horizontal=True)

    with right.container(border=True):
        st.markdown('<div class="card-title">Clinical History</div>', unsafe_allow_html=True)
        jaundice = st.radio("Was the person born with jaundice?", yes_no, index=None, horizontal=True)
        family = st.radio("Is there a family member with autism?", yes_no, index=None, horizontal=True)
        used_app = st.radio("Has the person used a screening app before?", yes_no, index=None,
                            horizontal=True)

    with st.container(border=True):
        st.markdown('<div class="card-title">Autism Spectrum Quotient (AQ-10)</div>',
                    unsafe_allow_html=True)
        st.caption("Answer every statement: does it describe the person?")
        qa, qb = st.columns(2, gap="large")
        aq = []
        for i, q in enumerate(AQ_QUESTIONS):
            col = qa if i < 5 else qb
            aq.append(col.radio(f"**Q{i+1}.** {q}", yes_no, index=None, horizontal=True,
                                key=f"aq{i+1}"))

    submitted = st.form_submit_button("Predict", width= "stretch")

if submitted:
    required = [age, gender, ethnicity, country, relation, jaundice, family, used_app] + aq
    if any(v is None for v in required):
        st.warning("Please answer every question before predicting.")
        st.stop()

    aq_vals = [1 if a == "Yes" else 0 for a in aq]
    row = {f"A{i+1}_Score": v for i, v in enumerate(aq_vals)}
    row.update({
        "age": float(age),
        "gender": encoders["gender"].transform([gender_opts[gender]])[0],
        "ethnicity": encoders["ethnicity"].transform([ethnicity_opts[ethnicity]])[0],
        "jaundice": encoders["jaundice"].transform([jaundice.lower()])[0],
        "austim": encoders["austim"].transform([family.lower()])[0],
        "contry_of_res": encoders["contry_of_res"].transform([country_opts[country]])[0],
        "used_app_before": encoders["used_app_before"].transform([used_app.lower()])[0],
        "relation": encoders["relation"].transform([relation_opts[relation]])[0],
    })
    X = pd.DataFrame([row])[FEATURES]

    proba = float(model.predict_proba(X)[0][1])
    pred = int(proba >= THRESHOLD)   # threshold tuned in the notebook

    st.divider()
    if pred == 1:
        st.error("**ASD traits likely detected.** A clinical evaluation by a specialist is recommended.")
    else:
        st.success("**ASD traits not likely detected.**")
    st.metric("Model probability of ASD traits", f"{proba:.1%}")
    st.progress(min(max(proba, 0.0), 1.0))
    st.caption("This is a screening aid only and does not replace professional diagnosis.")
