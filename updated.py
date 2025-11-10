import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
#                DATA LOADING STAGE
# ==================================================
# In this stage, we load our training and test datasets using pandas from the Downloads folder.
# Any columns with default names (such as "Unnamed")—which may result from previous CSV exports—are removed to keep only relevant features.

print("Loading data...")
import os

downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
train_path = os.path.join(downloads_path, 'Training.csv')
test_path = os.path.join(downloads_path, 'Testing.csv')

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train = train.loc[:, ~train.columns.str.contains('^Unnamed')]
test = test.loc[:, ~test.columns.str.contains('^Unnamed')]

print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")

# ==================================================
#               DATA CLEANING STAGE
# ==================================================
# Remove duplicate rows to avoid model bias and data leakage.
# Duplicated data can cause overestimation of model accuracy and make the classifier less generalizable.
print("\nCleaning data...")
train = train.drop_duplicates()
test = test.drop_duplicates()
print(f"After removing duplicates - Train: {train.shape}, Test: {test.shape}")

# ==================================================
#           FEATURE ENGINEERING STAGE
# ==================================================
# Feature engineering refines the raw symptom data:
# - 'symptom_sum' sums all symptom columns, giving an overall illness severity score.
# - 'respiratory_sum' groups key respiratory symptoms for additional clinical insight.
# Aggregated features based on medical domain knowledge improve model performance and interpretability.

def add_features(df):
    symptom_cols = [col for col in df.columns if col != 'prognosis']
    df['symptom_sum'] = df[symptom_cols].sum(axis=1)
    respiratory_cols = ['cough', 'high_fever', 'runny_nose', 'sinus_pressure', 'congestion']
    group_cols = [col for col in respiratory_cols if col in df.columns]
    if group_cols:
        df['respiratory_sum'] = df[group_cols].sum(axis=1)
    return df

print("\nEngineering features...")
train = add_features(train)
test = add_features(test)

# ==================================================
#              TRAIN-TEST PREPARATION
# ==================================================
# Split the dataset into features (X) and labels (y), ensuring the target variable ('prognosis') is separated.
# Labels are encoded as numbers with LabelEncoder since most classifiers require numeric targets.
# This step sets the stage for supervised learning.

X_train_full = train.drop(columns=['prognosis'])
y_train_raw = train['prognosis']

X_test_full = test.drop(columns=['prognosis'])
y_test_raw = test['prognosis']

le = LabelEncoder()
y_train_full = le.fit_transform(y_train_raw)
y_test_full = le.transform(y_test_raw)

print(f"\nNumber of classes: {len(le.classes_)}")
print(f"Sample classes: {le.classes_[:5]}")

# ==================================================
#             CLASS DISTRIBUTION CHECK
# ==================================================
# Understanding the frequency of each disease class:
# - Class imbalance can cause poor predictions for rare diseases.
# - Classes with only one sample may cause problems with stratified splitting or model generalization.
from collections import Counter

train_counts = Counter(y_train_full)
test_counts = Counter(y_test_full)

print("\nChecking class distribution...")
print(f"Classes with only 1 sample in train: {sum(1 for count in train_counts.values() if count == 1)}")
print(f"Classes with only 1 sample in test: {sum(1 for count in test_counts.values() if count == 1)}")

# ==================================================
#              TRAIN-VALIDATION SPLIT
# ==================================================
# Divide the training data into train and validation sets.
# Stratified splitting preserves class proportions if every class has enough samples; else, random splitting avoids errors.
# Validation set enables unbiased performance measurement and assists in model selection.

min_train_samples = min(train_counts.values())
min_test_samples = min(test_counts.values())

print(f"Minimum samples per class - Train: {min_train_samples}, Test: {min_test_samples}")

if min_train_samples >= 2:
    print("Using stratified split for train/val...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
else:
    print("Warning: Some classes have only 1 sample in train. Using random split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

X_test_split = X_test_full
y_test_split = y_test_full

print(f"\nData splits - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test_split.shape}")

# ==================================================
#                FEATURE SCALING STAGE
# ==================================================
# Standardize features (zero mean, unit variance) using StandardScaler.
# Feature scaling is essential for algorithms like SVM and logistic regression.
# It ensures optimal convergence and prevents features with larger ranges from dominating.

print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_split)
X_test_full_scaled = scaler.transform(X_test_full)

# ==================================================
#               MODEL TRAINING STAGE
# ==================================================
# Train multiple classifiers:
# - Random Forest: Handles unscaled data and offers robust feature importance.
# - SVM and Logistic Regression: Require scaled data, suitable for different data complexities.
# This multi-model approach is a best practice for robust selection.

print("\nTraining models...")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

svc_model = SVC(kernel='rbf', random_state=42, probability=True)
svc_model.fit(X_train_scaled, y_train)

logreg_model = LogisticRegression(max_iter=500, random_state=42)
logreg_model.fit(X_train_scaled, y_train)

# ==================================================
#             MODEL VALIDATION STAGE
# ==================================================
# Measure each model’s performance on the validation set.
# The highest accuracy selects the best candidate for test evaluation.
# Comparing multiple models ensures the selected algorithm generalizes well.

print("\n" + "=" * 50)
print("VALIDATION RESULTS")
print("=" * 50)

y_val_pred_rf = rf_model.predict(X_val)
y_val_pred_svc = svc_model.predict(X_val_scaled)
y_val_pred_logreg = logreg_model.predict(X_val_scaled)

acc_rf = accuracy_score(y_val, y_val_pred_rf)
acc_svc = accuracy_score(y_val, y_val_pred_svc)
acc_logreg = accuracy_score(y_val, y_val_pred_logreg)

print(f"Random Forest: {acc_rf:.4f}")
print(f"SVM: {acc_svc:.4f}")
print(f"Logistic Regression: {acc_logreg:.4f}")

best_acc = max(acc_rf, acc_svc, acc_logreg)
if best_acc == acc_rf:
    best_model = rf_model
    best_name = "Random Forest"
    use_scaled = False
elif best_acc == acc_svc:
    best_model = svc_model
    best_name = "SVM"
    use_scaled = True
else:
    best_model = logreg_model
    best_name = "Logistic Regression"
    use_scaled = True

print(f"\nBest model: {best_name} (Accuracy: {best_acc:.4f})")

# ==================================================
#             TEST EVALUATION STAGE
# ==================================================
# Test the selected model on entirely unseen test data.
# Print accuracy and a classification report (precision, recall, F1) for all diseases.
# This is the ultimate check for real-world performance before deployment.

print("\n" + "=" * 50)
print("TEST SET RESULTS")
print("=" * 50)

if use_scaled:
    y_test_pred = best_model.predict(X_test_scaled)
else:
    y_test_pred = best_model.predict(X_test_split)

test_acc = accuracy_score(y_test_split, y_test_pred)
print(f"Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_split, y_test_pred, target_names=le.classes_, zero_division=0))

# ==================================================
#             DATA VISUALIZATION STAGE
# ==================================================
# Use a confusion matrix heatmap to visualize the model’s prediction correctness for each disease class.
# The diagonal shows correct predictions; off-diagonal represents misclassifications.
# This helps you spot where the model struggles and may need further tuning.

cm = confusion_matrix(y_test_split, y_test_pred)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, cbar_kws={'label': 'Count'})
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Test Set Confusion Matrix", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================================================
#          FEATURE IMPORTANCE VISUALIZATION
# ==================================================
# For Random Forest, plot top 15 most influential features.
# This informs you which symptoms are most diagnostic, valuable for medical experts and further model refinement.

if best_name == "Random Forest":
    importances = best_model.feature_importances_
    indices = importances.argsort()[::-1][:15]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(indices)), importances[indices], color='steelblue')
    plt.xticks(range(len(indices)), X_train.columns[indices], rotation=45, ha='right')
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Importance", fontsize=12)
    plt.title("Top 15 Feature Importances", fontsize=14)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==================================================
#                MODEL SAVING STAGE
# ==================================================
# Save the trained model, the scaler, label encoder, and feature column order with joblib.
# Having all required artifacts means you can consistently reproduce predictions in deployment or APIs.

print("\nSaving models...")
joblib.dump(best_model, 'disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(X_train_full.columns.tolist(), 'symptom_columns.pkl')
print("Models saved successfully!")

# ==================================================
#             PREDICTION FUNCTION STAGE
# ==================================================
# Main API for predictions: receives a symptom dictionary, processes it, engineers features, scales as needed,
# and outputs diagnosis with class probabilities. Returns top-5 likely diseases.
# Useful for interactive tools, clinics, and web integrations.

def predict_disease(symptom_dict):
    model = joblib.load('disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    symptom_cols = joblib.load('symptom_columns.pkl')
    
    input_arr = []
    for col in symptom_cols:
        if col in ['symptom_sum', 'respiratory_sum']:
            continue
        input_arr.append(symptom_dict.get(col, 0))
    
    input_df = pd.DataFrame([input_arr], columns=[c for c in symptom_cols if c not in ['symptom_sum', 'respiratory_sum']])
    input_df = add_features(input_df)
    
    if use_scaled:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probs = model.predict_proba(input_scaled)[0]
    else:
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
    
    predicted_disease = le.inverse_transform([prediction])[0]
    top_indices = probs.argsort()[::-1][:5]
    top_diseases = [(le.classes_[i], probs[i]) for i in top_indices]
    
    return predicted_disease, top_diseases

# ==================================================
#             EXAMPLE PREDICTION STAGE
# ==================================================
# Demonstrate the prediction function with sample symptoms.
# This proves the pipeline is working and provides a clear template for usage in clinical or web settings.

print("\n" + "=" * 50)
print("EXAMPLE PREDICTION")
print("=" * 50)

example_symptoms = {
    'itching': 1,
    'skin_rash': 1,
    'nodal_skin_eruptions': 1
}

predicted, top_5 = predict_disease(example_symptoms)
print(f"\nPredicted Disease: {predicted}")
print("\nTop 5 Predictions:")
for disease, prob in top_5:
    print(f"  {disease}: {prob*100:.2f}%")

print("\n" + "=" * 50)
print("MODEL TRAINING COMPLETE!")
print("=" * 50)
