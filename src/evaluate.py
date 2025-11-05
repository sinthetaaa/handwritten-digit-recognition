import numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

(_, _), (Xte, yte) = tf.keras.datasets.mnist.load_data()
Xte = (Xte/255.0).astype("float32")[..., None]

model = tf.keras.models.load_model("models/mnist_cnn.keras")
probs = model.predict(Xte, verbose=0)
pred  = probs.argmax(1)

cm = confusion_matrix(yte, pred)
ConfusionMatrixDisplay(cm).plot(values_format="d")
plt.title("MNIST Confusion Matrix")
plt.tight_layout(); plt.show()

wrong = np.where(pred != yte)[0][:25]
plt.figure(figsize=(8,8))
for i, idx in enumerate(wrong):
    plt.subplot(5,5,i+1); plt.imshow(Xte[idx,...,0], cmap="gray"); plt.axis("off")
    plt.title(f"y={yte[idx]}, ŷ={pred[idx]}")
plt.suptitle("Misclassified examples")
plt.tight_layout(); plt.show()
