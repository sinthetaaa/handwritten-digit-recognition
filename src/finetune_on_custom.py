import os, tensorflow as tf
from tensorflow.keras import layers

MODEL_IN  = "models/mnist_advanced.keras"  
MODEL_OUT = "models/mnist_finetuned.keras"
DATA_DIR  = "my_digits"
BATCH     = 32
EPOCHS_WARM = 3  
EPOCHS_FULL = 8    

assert os.path.exists(MODEL_IN), f"Missing {MODEL_IN}"
for i in range(10):
    assert os.path.isdir(f"{DATA_DIR}/{i}"), f"Missing class folder {DATA_DIR}/{i}"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR, labels="inferred", label_mode="int", color_mode="grayscale",
    image_size=(28,28), batch_size=BATCH, validation_split=0.2, subset="training", seed=42
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR, labels="inferred", label_mode="int", color_mode="grayscale",
    image_size=(28,28), batch_size=BATCH, validation_split=0.2, subset="validation", seed=42
)

norm = tf.keras.Sequential([layers.Rescaling(1/255.)])
augment = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomTranslation(0.08, 0.08),
])
train_ds = train_ds.map(lambda x,y: (augment(norm(x), training=True), y)).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(lambda x,y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)

base = tf.keras.models.load_model(MODEL_IN)

for l in base.layers: l.trainable = False
base.layers[-1].trainable = True  
base.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
             loss="sparse_categorical_crossentropy", metrics=["accuracy"])
base.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_WARM, verbose=2)

unfreeze = False
for l in reversed(base.layers):
    l.trainable = True
    if isinstance(l, (layers.Conv2D, layers.DepthwiseConv2D)) and l.filters <= 32:
        break
    unfreeze = True

base.compile(optimizer=tf.keras.optimizers.AdamW(1e-4, weight_decay=1e-4),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(),
             metrics=["accuracy"])
cb = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy")
base.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FULL, callbacks=[cb], verbose=2)

os.makedirs("models", exist_ok=True)
base.save(MODEL_OUT)
print(f"Saved {MODEL_OUT}")