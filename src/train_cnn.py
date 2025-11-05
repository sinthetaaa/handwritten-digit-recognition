import numpy as np, tensorflow as tf
from tensorflow.keras import layers, models

tf.random.set_seed(42)

(Xtr, ytr), (Xte, yte) = tf.keras.datasets.mnist.load_data()
Xtr = (Xtr/255.0).astype("float32")[..., None]  
Xte = (Xte/255.0).astype("float32")[..., None]

BATCH = 128

data_augment = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomTranslation(0.08, 0.08),
])

def map_train(x, y):
    x = data_augment(x, training=True)
    return x, y

train_ds = (tf.data.Dataset.from_tensor_slices((Xtr, ytr))
            .shuffle(20_000).batch(BATCH).map(map_train).prefetch(tf.data.AUTOTUNE))
val_ds   = (tf.data.Dataset.from_tensor_slices((Xtr[-6000:], ytr[-6000:]))
            .batch(BATCH).prefetch(tf.data.AUTOTUNE))
test_ds  = tf.data.Dataset.from_tensor_slices((Xte, yte)).batch(BATCH)

model = models.Sequential([
    layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=(28,28,1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),               
    layers.Dropout(0.25),

    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),        
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
    tf.keras.callbacks.ModelCheckpoint("models/mnist_cnn.keras", save_best_only=True, monitor="val_accuracy"),
]
history = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks, verbose=2)

test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
