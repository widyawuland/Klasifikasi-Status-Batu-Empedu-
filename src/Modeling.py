# Modeling

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import time
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib

def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

results = {}

def eval_model(name, y_true, y_pred, y_prob, t):
    print(f"\n=== {name} ===")
    print(f"Training Time: {t:.4f} detik")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    plot_confusion(y_true, y_pred, f"Confusion Matrix - {name}")

    results[name] = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_true, y_prob),
        'Training Time (s)': t
    }

#@title Model 1: Logistic Regression

start = time.time()
logreg = LogisticRegression(max_iter=500)
logreg.fit(X_train_p, y_train)
end = time.time()

y_pred_lr = logreg.predict(X_test_p)
y_prob_lr = logreg.predict_proba(X_test_p)[:,1]

eval_model("Logistic Regression", y_test, y_pred_lr, y_prob_lr, end-start)

# Prediksi Test Set
pred_sample = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred_lr[:10],
    'Probability': y_prob_lr[:10]
})
print("\nContoh Prediksi Test Set (Logistic Regression):")
print(pred_sample)

#@title Model 2: Gradient Boosting

start = time.time()
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train_p, y_train)
end = time.time()

y_pred_gb = gb.predict(X_test_p)
y_prob_gb = gb.predict_proba(X_test_p)[:,1]

eval_model("Gradient Boosting", y_test, y_pred_gb, y_prob_gb, end-start)

# Feature Importance
feat_names = preprocessor.get_feature_names_out()
feat_imp = pd.Series(gb.feature_importances_, index=feat_names)
feat_imp = feat_imp.sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Top 10 Feature Importance - Gradient Boosting")
plt.show()

#@title Model 3: Deep Learning (MLP)
input_dim = X_train_p.shape[1]

early_stop = EarlyStopping(
    monitor='loss',
    patience=10,
    restore_best_weights=True
)

mlp = Sequential([
    Dense(16, activation='relu',
          input_shape=(input_dim,),
          kernel_regularizer=l2(0.001)),
    Dense(8, activation='relu',
          kernel_regularizer=l2(0.001)),
    Dense(1, activation='sigmoid')
])

# Model Summary
print("\n=== Model Summary (MLP) ===")
mlp.summary()

mlp.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

start = time.time()
history = mlp.fit(
    X_train_p, y_train,
    validation_data=(X_test_p, y_test),
    epochs=25,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)
end = time.time()

# evaluasi test set
y_prob_dl = mlp.predict(X_test_p).flatten()
y_pred_dl = (y_prob_dl > 0.5).astype(int)

eval_model("Deep Learning MLP", y_test, y_pred_dl, y_prob_dl, end-start)

# Training & Validation Loss per epoch
plt.figure(figsize=(12,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Training & Validation Accuracy per epoch
plt.figure(figsize=(12,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Prediksi Test Set
pred_sample = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred_dl[:10],
    'Probability': y_prob_dl[:10]
})
print("\nContoh Prediksi Test Set (MLP):")
print(pred_sample)

#perbandingan akhir model
results_df = pd.DataFrame(results).T.sort_values(by='F1-Score', ascending=False)
print("\n=== Perbandingan akhir model ===")
print(results_df.round(4))

#visualisasi perbandingan model
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']

plt.figure(figsize=(14, 6))
results_df[metrics].plot(kind='bar')

plt.title('Perbandingan Performa Model')
plt.ylabel('Score')
plt.xlabel('Model')
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

joblib.dump(logreg, "logistic_regression_model.pkl")
print("Model Logistic Regression berhasil disimpan")

joblib.dump(gb, "gradient_boosting_model.pkl")
print("Model Gradient Boosting berhasil disimpan")

mlp.save("mlp_deep_learning_model.h5")
print("Model MLP berhasil disimpan")