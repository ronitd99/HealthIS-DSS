import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from prep_data import X_train, X_test, y_train, y_test

# -----------------------------
# TRAIN FINAL MODEL
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=5000, class_weight="balanced")
model.fit(X_train_scaled, y_train)

# -----------------------------
# RISK CATEGORY FUNCTION
# -----------------------------
def risk_category(prob):
    if prob < 0.2:
        return "Low Risk"
    elif prob < 0.5:
        return "Medium Risk"
    else:
        return "High Risk"

# -----------------------------
# STREAMLIT APP
# -----------------------------
st.title("Atrial Fibrillation Risk DSS")
st.subheader("Post-MI In-Hospital AF Risk Assessment")

st.write(
    "This prototype estimates the risk of atrial fibrillation in hospitalized "
    "myocardial infarction patients using admission/history variables."
)

# Use one patient from test set as demo
patient_index = st.number_input(
    "Select a patient index from the test set",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0,
    step=1
)

if st.button("Assess Risk"):
    patient_data = X_test.iloc[[patient_index]]
    patient_scaled = scaler.transform(patient_data)

    prob = model.predict_proba(patient_scaled)[0, 1]
    risk = risk_category(prob)

    st.write(f"**Predicted Probability of AF:** {prob:.3f}")
    st.write(f"**Risk Category:** {risk}")

    if risk == "Low Risk":
        st.write("**Recommendation:** Routine monitoring.")
    elif risk == "Medium Risk":
        st.write("**Recommendation:** Increased rhythm surveillance and clinical review.")
    else:
        st.write("**Recommendation:** Close cardiac monitoring and consider early cardiology consultation.")

    st.write("### Patient Predictors")
    st.dataframe(patient_data.T.rename(columns={patient_index: "Value"}))