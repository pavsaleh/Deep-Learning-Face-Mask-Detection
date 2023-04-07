import os
import cv2
import numpy as np

DATASET_DIR = "P:/Uni/Masters/Winter 23/ELG 6118/Project/dataset"
CATEGORIES = ["with_mask", "without_mask"]

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DATASET_DIR, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        data.append(image)
        labels.append(CATEGORIES.index(category))

data = np.array(data) / 255.0
labels = np.array(labels)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

head_model = base_model.output
head_model = AveragePooling2D(pool_size=(4, 4))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(64, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(1, activation="sigmoid")(head_model)

model = Model(inputs=base_model.input, outputs=head_model)

from tensorflow.keras.optimizers import Adam

INIT_LR = 0.001
EPOCHS = 10
BATCH_SIZE = 32

optimizer = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS)

from sklearn.metrics import classification_report

y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
y_pred = np.round(y_pred).astype(int)

print(classification_report(y_test, y_pred, target_names=CATEGORIES))
