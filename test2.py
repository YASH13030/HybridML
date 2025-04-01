import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

file_path = r"C:\Users\hashr\OneDrive\Desktop\PBL_6th\scaled_dataset_2.csv"
df_sample = pd.read_csv(file_path, nrows=100000).dropna(subset=["Attack"])

print("Original Class Distribution:")
print(df_sample["Attack"].value_counts())

unique_classes = df_sample["Attack"].unique()
print("\nUnique classes:", unique_classes)


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_sample["Attack"].values)
X = df_sample.iloc[:, :-1].values

print("\nEncoded classes mapping:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"{cls} -> {i}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("\nClass distribution before SMOTE:", Counter(y_train))

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:", Counter(y_train))

num_classes = len(label_encoder.classes_)
assert all(0 <= label < num_classes for label in y_train), "Labels out of range!"
assert all(0 <= label < num_classes for label in y_test), "Labels out of range!"

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_cnn = np.expand_dims(X_train_scaled, axis=2)
X_test_cnn = np.expand_dims(X_test_scaled, axis=2)

input_shape = (X_train_cnn.shape[1], 1)

def create_cnn_model():
    inputs = Input(shape=input_shape)
    
    x = Conv1D(128, 5, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.4)(x)
    
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    features = Dense(128, activation='relu')(x)
    
    outputs = Dense(num_classes, activation='softmax')(features)
    
    return Model(inputs=inputs, outputs=outputs)

cnn_model = create_cnn_model()
cnn_model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train CNN
history = cnn_model.fit(X_train_cnn, y_train,
                        epochs=50,  
                        batch_size=128,
                        validation_data=(X_test_cnn, y_test),
                        callbacks=[EarlyStopping(patience=3)],
                        verbose=1)

feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-4].output)
X_train_features = feature_extractor.predict(X_train_cnn)
X_test_features = feature_extractor.predict(X_test_cnn)


xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    objective='multi:softmax', 
    num_class=num_classes,
    eval_metric='mlogloss',
    early_stopping_rounds=10,
    random_state=42
)

xgb_model.fit(X_train_features, y_train,
              eval_set=[(X_test_features, y_test)],
              verbose=1)

y_pred = xgb_model.predict(X_test_features)

y_test = np.array(y_test).ravel()
y_pred = np.array(y_pred).ravel()

print("\nFinal check before evaluation:")
print("y_test shape:", y_test.shape, "type:", type(y_test), "sample:", y_test[:5])
print("y_pred shape:", y_pred.shape, "type:", type(y_pred), "sample:", y_pred[:5])
print("Unique values in y_test:", np.unique(y_test))
print("Unique values in y_pred:", np.unique(y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))