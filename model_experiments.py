from prep_data import X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# -----------------------------
# MODEL 1: BASELINE LOGISTIC
# -----------------------------
model1 = LogisticRegression(max_iter=5000)
model1.fit(X_train_scaled, y_train)

y_pred1 = model1.predict(X_test_scaled)
y_prob1 = model1.predict_proba(X_test_scaled)[:, 1]

print("\nBASELINE MODEL")
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Precision:", precision_score(y_test, y_pred1, zero_division=0))
print("Recall:", recall_score(y_test, y_pred1, zero_division=0))
print("AUC:", roc_auc_score(y_test, y_prob1))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred1))


# -----------------------------
# MODEL 2: BALANCED LOGISTIC
# -----------------------------
model2 = LogisticRegression(max_iter=5000, class_weight="balanced")
model2.fit(X_train_scaled, y_train)

y_pred2 = model2.predict(X_test_scaled)
y_prob2 = model2.predict_proba(X_test_scaled)[:, 1]

print("\nBALANCED MODEL")
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Precision:", precision_score(y_test, y_pred2, zero_division=0))
print("Recall:", recall_score(y_test, y_pred2, zero_division=0))
print("AUC:", roc_auc_score(y_test, y_prob2))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred2))

# -----------------------------
# FEATURE IMPORTANCE (LOG REG)
# -----------------------------
import pandas as pd

coeffs = pd.Series(model2.coef_[0], index=X_train.columns)

# sort by absolute value (strength)
coeffs = coeffs.reindex(coeffs.abs().sort_values(ascending=False).index)

print("\nTOP 10 FEATURES:")
print(coeffs.head(10))

print("\nBOTTOM 10 FEATURES:")
print(coeffs.tail(10))

# -----------------------------
# MODEL 3: BALANCED + THRESHOLD
# -----------------------------
threshold = 0.3

y_pred3 = (y_prob2 >= threshold).astype(int)

print(f"\nBALANCED MODEL (Threshold = {threshold})")
print("Accuracy:", accuracy_score(y_test, y_pred3))
print("Precision:", precision_score(y_test, y_pred3, zero_division=0))
print("Recall:", recall_score(y_test, y_pred3, zero_division=0))
print("AUC:", roc_auc_score(y_test, y_prob2))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred3))

# -----------------------------
# FEATURE REDUCTION
# -----------------------------
low_features = [
    "np07", "nr02", "SIM_GIPERT", "DLIT_AG",
    "nr01", "endocr_03", "zab_leg_06",
    "endocr_01", "np04", "endocr_02"
]

X_train_reduced = X_train.drop(columns=low_features)
X_test_reduced = X_test.drop(columns=low_features)

# Scale reduced data
scaler2 = StandardScaler()
X_train_reduced_scaled = scaler2.fit_transform(X_train_reduced)
X_test_reduced_scaled = scaler2.transform(X_test_reduced)

# -----------------------------
# MODEL 4: REDUCED FEATURES
# -----------------------------
model4 = LogisticRegression(max_iter=5000, class_weight="balanced")
model4.fit(X_train_reduced_scaled, y_train)

y_pred4 = model4.predict(X_test_reduced_scaled)
y_prob4 = model4.predict_proba(X_test_reduced_scaled)[:, 1]

print("\nREDUCED MODEL")
print("Accuracy:", accuracy_score(y_test, y_pred4))
print("Precision:", precision_score(y_test, y_pred4, zero_division=0))
print("Recall:", recall_score(y_test, y_pred4, zero_division=0))
print("AUC:", roc_auc_score(y_test, y_prob4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred4))