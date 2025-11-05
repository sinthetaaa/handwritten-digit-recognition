# Handwritten Digit Recognition

An end-to-end MNIST digit recognizer that pairs interactive inference in Streamlit with reproducible TensorFlow training pipelines. The repository includes ready-to-use models, training scripts (from a scikit-learn baseline to a data-augmented CNN), evaluation helpers, and utilities for fine-tuning on custom handwriting samples.

## Features

- 🎨 **Interactive app**: Draw digits in the browser with `streamlit` + `streamlit-drawable-canvas`, view predictions, probabilities, and Grad-CAM heatmaps.
- 🧠 **Multiple model flavours**: Logistic regression baseline, a standard CNN, and an advanced convolutional architecture with depthwise blocks and cosine learning rate restarts.
- 🛠️ **Fine-tuning workflow**: Continue training the advanced model on your own dataset organized by digit.
- 📊 **Evaluation utilities**: Generate confusion matrices, inspect misclassifications, and probe robustness against noise, blur, rotation, shift, and stroke-thickness perturbations.

## Project Layout

```text
handwritten-digit-recognition/
├── models/
│   ├── mnist_cnn.keras           # CNN checkpoint trained on MNIST
│   └── mnist_advanced.keras      # Higher-accuracy architecture (fine-tuning base)
├── src/
│   ├── app.py                    # Streamlit UI for drawing + inference + Grad-CAM
│   ├── train_baseline.py         # scikit-learn logistic regression benchmark
│   ├── train_cnn.py              # CNN training script (TensorFlow/Keras)
│   ├── train_advanced.py         # Deeper CNN with augmentation & cosine LR restarts
│   ├── finetune_on_custom.py     # Transfer learning on a custom digit dataset
│   ├── evaluate.py               # Confusion matrix + misclassification inspection
│   └── evaluate_robustness.py    # Stress test saved model under corruptions
├── requirements-cloud.txt        # Dependency pin set for Linux/Cloud
├── requirements-mac.txt          # Dependency pin set for Apple Silicon (tensorflow-macos)
└── runtime.txt                   # Reference Python version (3.11)
```

## Getting Started

1. **Create a virtual environment** (Python 3.11 as per `runtime.txt`):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
2. **Install dependencies**:
   - Linux/Windows/Cloud:
     ```bash
     pip install -r requirements-cloud.txt
     ```
   - Apple Silicon (TensorFlow with Metal acceleration):
     ```bash
     pip install -r requirements-mac.txt
     ```
   - Optional baseline dependencies:
     ```bash
     pip install scikit-learn
     ```

## Run the Streamlit App

Launch the interactive digit recognizer (defaulting to `models/mnist_advanced.keras` if available):

```bash
streamlit run src/app.py
```

The sidebar lets you switch between saved models or load a custom `.keras` file, tweak the brush size, toggle a dark canvas, and enable/disable Grad-CAM visualizations.

## Training Pipelines

All training scripts expect to be run from the repository root (so that `models/` resolves correctly). They will create or update checkpoints under `models/`.

### 1. Logistic Regression Baseline

```bash
python src/train_baseline.py
```

- Uses scikit-learn’s `load_digits` (8×8 images) and `LogisticRegression`.
- Provides a quick sanity check before moving to deep learning approaches.

### 2. CNN on MNIST

```bash
python src/train_cnn.py
```

- Loads MNIST via `tf.keras.datasets`.
- Applies light data augmentation and trains a 4-layer convolutional network.
- Writes `models/mnist_cnn.keras` and reports held-out accuracy.

### 3. Advanced CNN with Depthwise Blocks

```bash
python src/train_advanced.py
```

- Adds stronger augmentation, depthwise separable convolutions, cosine decay with restarts, label smoothing, and model checkpoints.
- Produces `models/mnist_advanced.keras`, which powers the Streamlit app by default.

### 4. Fine-tune on Your Own Digits

Prepare a directory `my_digits/` with subfolders `0` through `9`, each containing grayscale PNG/JPEG samples resized or resizable to 28×28. Then run:

```bash
python src/finetune_on_custom.py
```

- Starts from `models/mnist_advanced.keras`, briefly warms up only the classifier head, then unfreezes deeper convolutional layers with AdamW.
- Saves the personalized model to `models/mnist_finetuned.keras`, selectable in the app.

## Evaluation and Robustness

- **Confusion matrix & misclassifications**:
  ```bash
  python src/evaluate.py
  ```
  Loads `models/mnist_cnn.keras` (adjust path as needed) and visualizes errors.

- **Robustness sweep**:
  ```bash
  python src/evaluate_robustness.py
  ```
  Uses `models/mnist_advanced.keras` to measure accuracy under additive noise, blur, rotations, translations, and optional stroke thickening (requires `opencv-python-headless`).

## Tips & Customization

- Grad-CAM overlays in the app require at least one convolutional layer; non-convolutional models will raise an informative warning.
- To try different models in Streamlit, drop additional `.keras` files into the `models/` directory and select them from the sidebar.
- When fine-tuning, keep class folders balanced to avoid bias; augmentations in the script help, but data quality matters most.
- GPU acceleration greatly speeds up training; configure TensorFlow accordingly for your platform (CUDA on Linux/Windows, TensorFlow-Metal on macOS).

## License

Specify your licensing terms here (e.g., MIT, Apache 2.0). If unspecified, consider adding a `LICENSE` file.

