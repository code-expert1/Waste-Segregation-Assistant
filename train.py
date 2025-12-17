# train.py
import os, json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, applications, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------
# 1. FIXED PATHS (THIS WAS THE MAIN PROBLEM)
# ---------------------------------------------------
BASE_DIR = r"C:\Users\Lenovo Yoga\Desktop\WasteProject"

# Your real folder is: WasteProject/dataset/wast_4class
DATA_DIR = os.path.join(BASE_DIR, "dataset", "wast_4class")

MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

IMG_SIZE = (224,224)
BATCH = 8
EPOCHS = 10

# ---------------------------------------------------
# 2. GENERATORS
# ---------------------------------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.12,
    height_shift_range=0.12,
    zoom_range=0.12,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_names = list(train_gen.class_indices.keys())
num_classes = len(class_names)
print("Classes found:", class_names)

# ---------------------------------------------------
# 3. MODEL (MobileNetV2)
# ---------------------------------------------------
base = applications.MobileNetV2(
    include_top=False,
    input_shape=IMG_SIZE + (3,),
    weights='imagenet'
)
base.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------------------------------------------
# 4. CALLBACKS
# ---------------------------------------------------
ckpt = callbacks.ModelCheckpoint(
    os.path.join(MODELS_DIR, "baseline_best.keras"),
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)
es = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# ---------------------------------------------------
# 5. TRAIN
# ---------------------------------------------------
hist = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[ckpt, es, rlr]
)

# ---------------------------------------------------
# 6. SAVE MODEL & CLASS MAPPING
# ---------------------------------------------------
final_path = os.path.join(MODELS_DIR, "waste_4class_baseline.keras")
model.save(final_path)

with open(os.path.join(MODELS_DIR, "class_mapping.json"), "w") as f:
    json.dump({"class_names": class_names}, f, indent=2)

# ---------------------------------------------------
# 7. SAVE TRAINING PLOTS
# ---------------------------------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(hist.history["accuracy"], label="train")
plt.plot(hist.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(hist.history["loss"], label="train")
plt.plot(hist.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(LOGS_DIR, "training_curves.png"))

print("Training complete.")
print("Saved model:", final_path)
