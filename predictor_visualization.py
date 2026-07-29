import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Heart Failure Survival Predictor",
    page_icon="🫀",
    layout="wide"
)

st.markdown("""
<style>
    .stSlider > div > div > div > div { background: black; }
    .stSlider > div > div > div > div > div { background: black; }
    .stSlider p { color: black !important; }
    div[data-testid="stThumbValue"] { color: black !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    df.drop(columns=["time"], inplace=True)
    X = df.drop(columns=["DEATH_EVENT"])
    y = df["DEATH_EVENT"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=21)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=50, max_depth=10, max_features="sqrt", min_samples_split=5, random_state=21)
    model.fit(X_train_s, y_train)
    return model, scaler, X.columns.tolist()

model, scaler, feature_names = train_model()

st.title("🫀 Heart Failure Survival Predictor")
st.markdown(
    "Enter patient clinical measurements to predict 30-day survival probability. "
    "Built with Random Forest trained on the [Chicco & Jurman (2020)](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5) dataset."
)
st.divider()

st.sidebar.header("Patient Clinical Features")
st.sidebar.markdown("Adjust the sliders to match patient values.")

age                      = st.sidebar.slider("Age (years)", 40, 95, 60)
anaemia                  = st.sidebar.selectbox("Anaemia", [0, 1], format_func=lambda x: "Yes" if x else "No")
creatinine_phosphokinase = st.sidebar.slider("Creatinine Phosphokinase (mcg/L)", 23, 7861, 250)
diabetes                 = st.sidebar.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")
ejection_fraction        = st.sidebar.slider("Ejection Fraction (%)", 14, 80, 38)
high_blood_pressure      = st.sidebar.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x else "No")
platelets                = st.sidebar.slider("Platelets (kiloplatelets/mL)", 25000, 850000, 262000, step=1000)
serum_creatinine         = st.sidebar.slider("Serum Creatinine (mg/dL)", 0.5, 9.4, 1.1, step=0.1)
serum_sodium             = st.sidebar.slider("Serum Sodium (mEq/L)", 113, 148, 137)
sex                      = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x else "Female")
smoking                  = st.sidebar.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x else "No")

input_data = pd.DataFrame([[
    age, anaemia, creatinine_phosphokinase, diabetes,
    ejection_fraction, high_blood_pressure, platelets,
    serum_creatinine, serum_sodium, sex, smoking
]], columns=feature_names)

input_scaled  = scaler.transform(input_data)
proba         = model.predict_proba(input_scaled)[0]
survival_prob = proba[0] * 100
death_prob    = proba[1] * 100

col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction")
    if death_prob >= 50:
        st.error(f"⚠️ High Risk — Death probability: **{death_prob:.1f}%**")
    elif death_prob >= 30:
        st.warning(f"⚠️ Moderate Risk — Death probability: **{death_prob:.1f}%**")
    else:
        st.success(f"✅ Low Risk — Survival probability: **{survival_prob:.1f}%**")

    fig_prob, ax = plt.subplots(figsize=(5, 1.5))
    ax.barh([""], [survival_prob], color="#2ecc71", label="Survived")
    ax.barh([""], [death_prob], left=[survival_prob], color="#e74c3c", label="Died")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    ax.set_title("Survival vs Death Probability", fontsize=10)
    ax.axvline(50, color="black", linestyle="--", linewidth=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -1.1), ncol=2, fontsize=9)
    plt.subplots_adjust(bottom=0.6)
    st.pyplot(fig_prob)

with col2:
    st.subheader("Patient Summary")
    summary = {
        "Age": age,
        "Ejection Fraction (%)": ejection_fraction,
        "Serum Creatinine (mg/dL)": serum_creatinine,
        "Serum Sodium (mEq/L)": serum_sodium,
        "Creatinine Phosphokinase": creatinine_phosphokinase,
        "Platelets": f"{platelets:,}",
        "Anaemia": "Yes" if anaemia else "No",
        "Diabetes": "Yes" if diabetes else "No",
        "High Blood Pressure": "Yes" if high_blood_pressure else "No",
        "Sex": "Male" if sex else "Female",
        "Smoking": "Yes" if smoking else "No",
    }
    st.dataframe(pd.DataFrame(summary.items(), columns=["Feature", "Value"]), hide_index=True)

st.divider()

st.subheader("Feature Importance (Random Forest)")
st.markdown("Which clinical features matter most for predicting survival?")

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)

fig_imp, ax2 = plt.subplots(figsize=(8, 5))
colors = ["#e74c3c" if f in ["serum_creatinine", "ejection_fraction", "age"]
          else "#3498db" for f in importance_df["Feature"]]
ax2.barh(importance_df["Feature"], importance_df["Importance"], color=colors)
ax2.set_xlabel("Importance Score")
ax2.set_title("Random Forest Feature Importance")
ax2.axvline(importance_df["Importance"].mean(), color="gray", linestyle="--", linewidth=1, label="Mean importance")
ax2.legend()
plt.tight_layout()
st.pyplot(fig_imp)

st.caption("🔴 Red bars = top 3 predictors (serum creatinine, ejection fraction, age) 🔵 Blue = other features.")
st.divider()

st.markdown("### 📝 Notes")
st.markdown(
    """
    This predictor is built on a **Random Forest** model, selected after testing 8 classifiers:
    Logistic Regression, SVM (Linear & RBF), KNN, Naive Bayes, Gradient Boosting, LightGBM, and Random Forest.

    While boosting models like LightGBM and Gradient Boosting generally outperform Random Forest
    on large datasets, Random Forest was the best performing model on this particular dataset
    (299 patients) — outperforming all others across Accuracy, F1, ROC-AUC, and MCC.
    With small datasets, simpler ensemble methods tend to generalize better as boosting models
    are more prone to overfitting when data is scarce.

    The final model was optimized using GridSearchCV with 5-fold cross-validation
    (best params: n_estimators=50, max_depth=10, max_features='sqrt', min_samples_split=5).

    ⚠️ This tool is for **educational purposes only** and is not a clinical
    decision-making tool. It should not replace professional medical judgment.
    """
)