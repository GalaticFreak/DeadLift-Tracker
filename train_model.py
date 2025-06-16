import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Load dataset
df = pd.read_csv("form_data.csv")
X = df[['knee_angle', 'hip_angle']]
y = LabelEncoder().fit_transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model path
print("Saving model in:", os.getcwd())
checkpoint = ModelCheckpoint("dl_model.h5", monitor='val_accuracy', save_best_only=False)

# Train
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[checkpoint])

# Manual Save
model.save("dl_model.h5")
print("✅ Model saved successfully.")
