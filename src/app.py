import os
import numpy as np, streamlit as st, tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Handwritten Digit Recognizer", page_icon="🔢", layout="centered")
st.title("Handwritten Digit Recognizer (Deep Learning)")

st.sidebar.header("Controls")

default_model = "models/mnist_advanced.keras" if os.path.exists("models/mnist_advanced.keras") else "models/mnist_cnn.keras"
model_choice = st.sidebar.selectbox(
    "Model file",
    options=[default_model, "models/mnist_cnn.keras", "models/mnist_advanced.keras", "models/mnist_finetuned.keras", "Custom…"],
    index=0
)
if model_choice == "Custom…":
    model_path = st.sidebar.text_input("Enter custom .keras path", value=default_model)
else:
    model_path = model_choice

brush_size   = st.sidebar.slider("Brush size", min_value=6, max_value=40, value=14, step=1)
dark_canvas  = st.sidebar.toggle("Dark canvas (black bg, white ink)", value=False)
show_gradcam = st.sidebar.toggle("Show Grad-CAM", value=True)

@st.cache_resource(show_spinner=True)
def load_model_cached(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return tf.keras.models.load_model(path)

try:
    model = load_model_cached(model_path)
except Exception as e:
    st.error(f"Could not load model at `{model_path}`:\n{e}")
    st.stop()

st.caption(f"Using model: `{model_path}`")

st.write("Draw a digit (0–9) in the box below, then click **Predict**.")

bg_color  = "#000000" if dark_canvas else "#FFFFFF"
ink_color = "#FFFFFF" if dark_canvas else "#000000"

canvas = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=brush_size,
    stroke_color=ink_color,
    background_color=bg_color,
    width=280, height=280, drawing_mode="freedraw",
    key=f"canvas_{'dark' if dark_canvas else 'light'}",
)

def preprocess(pil_img, dark=False):
    """RGBA -> 28x28 grayscale in [0,1]. Light canvas is inverted to match MNIST."""
    img = pil_img.convert("L")
    if not dark:
        img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    return arr[None, ..., None]  # (1, 28, 28, 1)

def _find_last_conv_layer(m: tf.keras.Model):
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def grad_cam(model: tf.keras.Model, x, class_index=None, last_conv_name=None):
    if class_index is None:
        class_index = int(np.argmax(model.predict(x, verbose=0)[0]))
    last_conv_name = last_conv_name or _find_last_conv_layer(model)
    if last_conv_name is None:
        raise ValueError("No Conv2D layer found for Grad-CAM.")
    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(last_conv_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        top = preds[:, class_index]
    grads   = tape.gradient(top, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam     = tf.reduce_sum(weights * conv_out, axis=-1)[0]
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()

def overlay_heatmap(base_28x28, heatmap_28x28, alpha=0.45, upscale=6):
    """Returns (heatmap_rgb, overlay_rgb). Falls back to matplotlib if OpenCV missing."""
    try:
        import cv2
    except ImportError:
        return None, None
    h = (heatmap_28x28 * 255).astype("uint8")
    h_color = cv2.applyColorMap(h, cv2.COLORMAP_JET)[:, :, ::-1]
    base = (base_28x28 * 255).astype("uint8")
    base_rgb = np.stack([base, base, base], axis=-1)
    overlay = (alpha * h_color + (1 - alpha) * base_rgb).astype("uint8")
    if upscale != 1:
        h_color = cv2.resize(h_color, (28 * upscale, 28 * upscale), interpolation=cv2.INTER_NEAREST)
        overlay = cv2.resize(overlay, (28 * upscale, 28 * upscale), interpolation=cv2.INTER_NEAREST)
    return h_color, overlay

col_clear, col_predict = st.columns(2)
with col_clear:
    if st.button("Clear Canvas"):
        st.rerun()
with col_predict:
    do_predict = st.button("Predict", type="primary")

if do_predict:
    if canvas.image_data is None:
        st.warning("Please draw a digit first.")
    else:
        pil = Image.fromarray((canvas.image_data).astype("uint8")).convert("RGB")
        x = preprocess(pil, dark=dark_canvas)

        probs = model.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))
        confidence = float(np.max(probs)) * 100.0

        st.subheader(f"Prediction: **{pred}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        top3_idx = np.argsort(-probs)[:3]
        top3_df = pd.DataFrame({
            "digit": top3_idx,
            "probability": probs[top3_idx]
        })
        st.write("Top-3 predictions")
        st.dataframe(top3_df.reset_index(drop=True), use_container_width=True)

        st.bar_chart(pd.DataFrame({"probability": probs}, index=list(range(10))))

        st.caption("Model Input (28×28 grayscale)")
        st.image(x[0, :, :, 0], width=150, clamp=True)

        if show_gradcam:
            try:
                cam = grad_cam(model, x, class_index=pred)
                heatmap_rgb, overlay_rgb = overlay_heatmap(x[0, :, :, 0], cam, alpha=0.45, upscale=6)
                st.markdown("**Grad-CAM (where the model looked)**")
                if heatmap_rgb is None:
                    fig = plt.figure()
                    plt.imshow(cam, cmap="jet"); plt.axis("off")
                    st.pyplot(fig)
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(heatmap_rgb, caption="Heatmap", use_column_width=True)
                    with c2:
                        st.image(overlay_rgb, caption="Overlay on input", use_column_width=True)
            except Exception as e:
                st.info(f"Grad-CAM not available: {e}")

st.caption("Tip: draw large and centered. Try the advanced or finetuned model from the sidebar.")
