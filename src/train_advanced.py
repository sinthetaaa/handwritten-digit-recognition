import os, math, tensorflow as tf, numpy as np
from tensorflow.keras import layers, models

tf.random.set_seed(42)

(Xtr, ytr), (Xte, yte) = tf.keras.datasets.mnist.load_data()
Xtr = (Xtr/255.0).astype("float32")[..., None]
Xte = (Xte/255.0).astype("float32")[..., None]

ytr_oh = tf.keras.utils.to_categorical(ytr, 10)
yte_oh = tf.keras.utils.to_categorical(yte, 10)

VAL_SPLIT = 6000
Xval, yval = Xtr[-VAL_SPLIT:], ytr_oh[-VAL_SPLIT:]
Xtr, ytr_oh = Xtr[:-VAL_SPLIT], ytr_oh[:-VAL_SPLIT]

BATCH = 128

augment = tf.keras.Sequential([
    layers.RandomRotation(0.10),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.10),
])

def ds(x, y, train=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if train:
        ds = ds.shuffle(40_000).map(lambda a,b: (augment(a, training=True), b),
                                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

train_ds = ds(Xtr, ytr_oh, train=True)
val_ds   = ds(Xval, yval)
test_ds  = tf.data.Dataset.from_tensor_slices((Xte, yte_oh)).batch(BATCH)

def conv_block(x, filters, k=3, dw=False):
    if dw:
        x = layers.DepthwiseConv2D(k, padding="same", use_bias=False)(x)
        x = layers.Conv2D(filters, 1, use_bias=False)(x)
    else:
        x = layers.Conv2D(filters, k, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

inp = layers.Input((28,28,1))
x = conv_block(inp, 32); x = conv_block(x, 32, dw=True); x = layers.MaxPooling2D()(x); x = layers.Dropout(0.2)(x)
x = conv_block(x, 64);   x = conv_block(x, 64, dw=True); x = layers.MaxPooling2D()(x); x = layers.Dropout(0.3)(x)
x = conv_block(x, 96);   x = conv_block(x, 96, dw=True); x = layers.MaxPooling2D()(x); x = layers.Dropout(0.4)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(10, activation="softmax")(x)
model = models.Model(inp, out)

steps_per_epoch = math.ceil(len(Xtr)/BATCH)
lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=3e-3,
    first_decay_steps=steps_per_epoch*3,
    t_mul=2.0, m_mul=0.5, alpha=1e-5
)
opt = tf.keras.optimizers.Adam(learning_rate=lr_sched)

model.compile(
    optimizer=opt,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

os.makedirs("models", exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("models/mnist_advanced.keras",
                                       monitor="val_accuracy", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
]

model.summary()
model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks, verbose=2)
print("Test accuracy:", model.evaluate(test_ds, verbose=0)[1])
