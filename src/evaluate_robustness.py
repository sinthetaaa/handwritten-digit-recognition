import numpy as np, tensorflow as tf
from scipy.ndimage import gaussian_filter, rotate, shift

MODEL_PATH = "models/mnist_advanced.keras"  
model = tf.keras.models.load_model(MODEL_PATH)

(_, _), (Xte, yte) = tf.keras.datasets.mnist.load_data()
Xte = (Xte/255.0).astype("float32")[..., None]

def eval_acc(x, y, name):
    preds = model.predict(x, verbose=0).argmax(1)
    acc = float((preds == y).mean())
    print(f"{name:16s}: {acc:.4f}")
    return acc

print("=== Robustness Eval ===")
base = eval_acc(Xte, yte, "clean")

def add_noise(x, std): return np.clip(x + np.random.normal(0, std, x.shape).astype("float32"), 0, 1)
for std in [0.05, 0.10, 0.20, 0.30]:
    eval_acc(add_noise(Xte, std), yte, f"noise σ={std}")

def add_blur(x, sigma): return np.stack([gaussian_filter(img, sigma=sigma) for img in x], 0)
for sig in [0.8, 1.2, 1.6]:
    eval_acc(add_blur(Xte, sig), yte, f"blur σ={sig}")

def add_rot(x, deg): return np.stack([rotate(img, deg, reshape=False, order=1, mode="nearest") for img in x], 0)
for deg in [10, 15, 20]:
    eval_acc(add_rot(Xte, deg), yte, f"rot {deg}°")

def add_shift(x, pix): return np.stack([shift(img, (pix, pix, 0), order=1, mode="nearest") for img in x], 0)
for pix in [2, 3]:
    eval_acc(add_shift(Xte, pix), yte, f"shift {pix}px")

try:
    import cv2
    def thicken(img, k):
        x8 = (img*255).astype("uint8")
        kernel = np.ones((k,k), np.uint8)
        x8 = cv2.dilate(x8, kernel, iterations=1)
        return (x8/255.0).astype("float32")
    for k in [2,3,4]:
        Xt = np.stack([thicken(Xte[i,:,:,0], k) for i in range(len(Xte))], 0)[..., None]
        eval_acc(Xt, yte, f"thick k={k}")
except ImportError:
    print("OpenCV not installed — skipping thickness tests. `pip install opencv-python-headless`")
