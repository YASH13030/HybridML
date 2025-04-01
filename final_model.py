import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dropout, Dense, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------- Data Loading & Balancing --------------------
file_path = r"C:\Users\hashr\OneDrive\Desktop\PBL_6th\preprocessed_dataset.csv"
chunk_size = 100000  
sampled_chunks = []
sampling_frac = 0.1  

for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    sampled_chunk = chunk.sample(frac=sampling_frac, random_state=42)
    sampled_chunks.append(sampled_chunk)

df_sample = pd.concat(sampled_chunks, ignore_index=True)
print("Total sampled rows:", df_sample.shape[0])
print("Class Distribution Before Balancing:")
print(df_sample["Attack"].value_counts())

min_class_size = df_sample["Attack"].value_counts().min()
balanced_dfs = []
for cls in df_sample["Attack"].unique():
    df_cls = df_sample[df_sample["Attack"] == cls]
    df_cls_bal = df_cls.sample(n=min_class_size, random_state=42, replace=(len(df_cls) < min_class_size))
    balanced_dfs.append(df_cls_bal)

df_balanced = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)
print("\nClass Distribution After Balancing:")
print(df_balanced["Attack"].value_counts())

# -------------------- Prepare Data --------------------
# Assume dataset is already scaled.
# Convert features to float32 explicitly.
X = df_balanced.iloc[:, :-1].values.astype(np.float32)
y = df_balanced["Attack"].values.astype(np.int32)
print("Final Balanced Dataset Shape:", X.shape, y.shape)

# -------------------- Train/Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Reshape for CNN input (add channel dimension)
X_train_cnn = np.expand_dims(X_train, axis=2)  # shape: (num_samples, 22, 1)
X_test_cnn = np.expand_dims(X_test, axis=2)
print("Reshaped Training Data:", X_train_cnn.shape)
print("Reshaped Testing Data:", X_test_cnn.shape)

# -------------------- Create TensorFlow Datasets with Explicit Casting --------------------
batch_size = 32

def cast_fn(x, y):
    # Ensure that features are float32 and labels are int32.
    return tf.cast(x, tf.float32), tf.cast(y, tf.int32)

train_ds = tf.data.Dataset.from_tensor_slices((X_train_cnn, y_train))
train_ds = train_ds.map(cast_fn, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_test_cnn, y_test))
val_ds = val_ds.map(cast_fn, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# -------------------- Build CNN Model --------------------
input_shape = (X_train_cnn.shape[1], 1)
inputs = Input(shape=input_shape)

# Convolutional Block 1
x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Convolutional Block 2
x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Convolutional Block 3
x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Global Average Pooling
x = GlobalAveragePooling1D()(x)

# Dense Block with L2 regularization
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

# Feature extraction layer (penultimate layer)
feature_layer = Dense(64, activation='relu', name='feature_layer', kernel_regularizer=l2(0.001))(x)

# Final classification layer (binary output)
outputs = Dense(1, activation='sigmoid')(feature_layer)

cnn_model = Model(inputs=inputs, outputs=outputs)
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.summary()

# -------------------- Callbacks --------------------
checkpoint_path = "best_cnn_weights.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# -------------------- Train CNN --------------------
history = cnn_model.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# -------------------- Feature Extraction for XGBoost --------------------
if os.path.exists(checkpoint_path):
    cnn_model.load_weights(checkpoint_path)

# Create a feature extractor model from the penultimate layer
feature_extractor = Model(inputs=cnn_model.input,
                          outputs=cnn_model.get_layer('feature_layer').output)
X_train_features = feature_extractor.predict(X_train_cnn)
X_test_features = feature_extractor.predict(X_test_cnn)

# -------------------- Train XGBoost --------------------
from xgboost import XGBClassifier  # (imported earlier)
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train_features, y_train)
y_pred = xgb_model.predict(X_test_features)

accuracy = accuracy_score(y_test, y_pred)
print(f"Hybrid CNN + XGBoost Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
