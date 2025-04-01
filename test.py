import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess data
file_path = r"C:\Users\hashr\OneDrive\Desktop\PBL_6th\scaled_dataset_2.csv"
df_sample = pd.read_csv(file_path, nrows=100000)

df_sample = df_sample.dropna(subset=["Attack"])
print("Total rows from dataset:", df_sample.shape[0])


print("Class Distribution Before Balancing:")
print(df_sample["Attack"].value_counts())

# Balance dataset
min_class_size = df_sample["Attack"].value_counts().min()
balanced_dfs = [df_sample[df_sample["Attack"] == cls].sample(n=min_class_size, random_state=42) for cls in df_sample["Attack"].unique()]
df_balanced = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)

print("\nClass Distribution After Balancing:")
print(df_balanced["Attack"].value_counts())

# Prepare data
X = df_balanced.iloc[:, :-1].values
y = df_balanced["Attack"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_cnn = np.expand_dims(X_train_scaled, axis=2)
X_test_cnn = np.expand_dims(X_test_scaled, axis=2)

# Build CNN Model
input_shape = (X_train_cnn.shape[1], 1)
input_layer = Input(shape=input_shape)

conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
drop1 = Dropout(0.3)(conv1)
conv2 = Conv1D(filters=32, kernel_size=3, activation='relu')(drop1)
drop2 = Dropout(0.3)(conv2)
flat = Flatten()(drop2)
dense1 = Dense(64, activation='relu')(flat)
drop3 = Dropout(0.3)(dense1)
feature_output = Dense(32, activation='relu')(drop3)

cnn_model = Model(inputs=input_layer, outputs=feature_output)
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_cnn, y_test), verbose=1)

# Extract Features using CNN
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.output)
X_train_features = feature_extractor.predict(X_train_cnn)
X_test_features = feature_extractor.predict(X_test_cnn)

# Train XGBoost Classifier
xgb_model = XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train_features, y_train)

# Evaluate Model
y_pred = xgb_model.predict(X_test_features)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nHybrid CNN + XGBoost Accuracy: {accuracy:.4f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))
