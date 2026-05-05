"""
=======================================================
  AI OFFSIDE DETECTOR — Streamlit Web App

  What this file does:
  - Creates an interactive web app where you upload
    a football image and the AI classifies it
  - Loads the trained ResNet18 CNN from offside_cnn.pth
  - Runs the image through the model and shows the result

  Run from the project root:
      KMP_DUPLICATE_LIB_OK=TRUE streamlit run app.py --server.fileWatcherType none
=======================================================
"""

# ── Imports ────────────────────────────────────────────────────────────────────

import sys, os
sys.path.append(os.path.dirname(__file__))   # make sure Python can find our modules

import streamlit as st          # Streamlit — turns Python scripts into web apps
import torch                    # PyTorch — loads and runs our trained CNN
import torch.nn as nn           # neural network tools (we need nn.Linear to rebuild the model)
from torchvision import models, transforms   # ResNet18 architecture + image transforms
from PIL import Image           # Python Imaging Library — opens image files


# ══════════════════════════════════════════════════════
#  PAGE SETUP
# ══════════════════════════════════════════════════════

# Configure the browser tab — title, icon, layout
st.set_page_config(
    page_title="AI Offside Detector",
    page_icon="⚽",
    layout="centered"   # center the content on the page
)

# Main title and description shown at the top of the page
st.title("⚽ AI Offside Detector")
st.markdown("""
Upload a football match image and the AI will classify it as **Offside** or **Onside**
using a **ResNet18 CNN** trained on real match images.
""")
st.divider()   # draws a horizontal line


# ══════════════════════════════════════════════════════
#  LOAD THE TRAINED MODEL
# ══════════════════════════════════════════════════════

MODEL_PATH = "model/offside_cnn.pth"   # path to our saved model weights
IMG_SIZE   = 224                        # must match the size used during training

@st.cache_resource
# @st.cache_resource means: load the model ONCE and reuse it for every upload
# Without this, the model would reload from disk every time — very slow!
def load_model():
    """
    Rebuilds the exact same ResNet18 architecture we used for training,
    then loads the saved weights (offside_cnn.pth) into it.
    
    Why do we need to rebuild the architecture?
    PyTorch saves only the WEIGHTS, not the full model structure.
    So we must first recreate the same structure, then fill it with the saved weights.
    """

    # Recreate the ResNet18 architecture (same as in train_model.py)
    model = models.resnet18(weights=None)   # weights=None because we load our own
    model.fc = nn.Linear(model.fc.in_features, 2)   # same 2-class final layer

    # If no trained model exists yet, return None (app will show an error)
    if not os.path.exists(MODEL_PATH):
        return None, None

    # Load the saved weights from training
    # map_location="cpu" means load to CPU even if trained on GPU
    checkpoint  = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])   # fill model with saved weights
    class_names = checkpoint.get("class_names", ["offside", "onside"])

    model.eval()   # set to evaluation mode — disables training-only features like dropout
    return model, class_names

# Actually load the model when the app starts
model, class_names = load_model()

# If model file doesn't exist, stop the app and show an error
if model is None:
    st.error("⚠️ No trained model found. Run `python3 model/train_model.py` first.")
    st.stop()


# ══════════════════════════════════════════════════════
#  IMAGE PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════

# These transforms MUST match exactly what we used during training
# The model learned to expect images in this specific format
transform = transforms.Compose([

    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # Resize uploaded image to 224×224 (ResNet18 required input size)

    transforms.ToTensor(),
    # Convert image from PIL format (0-255) to PyTorch tensor (0.0-1.0)

    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    # Normalise using ImageNet statistics — required because we used
    # Transfer Learning from a model pretrained on ImageNet
])


# ══════════════════════════════════════════════════════
#  SIDEBAR — explanation panel on the left
# ══════════════════════════════════════════════════════

with st.sidebar:
    st.header("ℹ️ How it works")
    st.markdown("""
    **Model:** ResNet18 CNN  
    **Trained on:** 102 real football images  
    **Classes:** offside / onside  
    **Transfer Learning:** ImageNet → Football  

    **Pipeline:**
    ```
    Image
      ↓ Resize to 224×224
      ↓ Normalise pixels
      ↓ ResNet18 CNN
    OFFSIDE / ONSIDE + confidence
    ```
    """)


# ══════════════════════════════════════════════════════
#  MAIN — image upload and prediction
# ══════════════════════════════════════════════════════

# File uploader widget — accepts jpg and png images
uploaded_file = st.file_uploader(
    "Upload a football match image",
    type=["jpg", "jpeg", "png"]
)

# Only run the prediction if the user has uploaded something
if uploaded_file:

    # Open the uploaded image and convert to RGB
    # (some images are RGBA with transparency — we strip the alpha channel)
    image = Image.open(uploaded_file).convert("RGB")

    # Show the original uploaded image on the page
    st.subheader("📷 Your Image")
    st.image(image, width=700)

    # Run the CNN prediction
    with st.spinner("Analysing with CNN..."):

        # Apply our preprocessing pipeline to the image
        # Result: tensor of shape [3, 224, 224] (3 colour channels, 224×224 pixels)
        tensor = transform(image)

        # Add a batch dimension — models expect [batch_size, channels, height, width]
        # unsqueeze(0) changes shape from [3, 224, 224] to [1, 3, 224, 224]
        tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            # torch.no_grad() — we're not training, so we don't need gradients
            # This makes inference faster and uses less memory

            outputs = model(tensor)
            # Forward pass through ResNet18
            # outputs shape: [1, 2] — one raw score per class
            # Example: [[-0.8, 1.2]] → model leans toward class 1 (onside)

            probs = torch.softmax(outputs, dim=1)[0]
            # Softmax converts raw scores into probabilities that sum to 1
            # Example: [-0.8, 1.2] → [0.32, 0.68] (32% offside, 68% onside)

            pred_idx   = probs.argmax().item()
            # argmax picks the class with the highest probability
            # Example: argmax([0.32, 0.68]) = 1 (onside)

            confidence = probs[pred_idx].item()
            # The probability of the predicted class (our confidence score)

            prediction = class_names[pred_idx]
            # Convert class index to class name: 0 → 'offside', 1 → 'onside'

    # ── Display the result ────────────────────────────────────────────────────
    st.subheader("🔍 AI Result")

    if prediction == "offside":
        # Red error box for offside
        st.error(f"🚩 **OFFSIDE** — {confidence:.0%} confidence")
    else:
        # Green success box for onside
        st.success(f"✅ **ONSIDE** — {confidence:.0%} confidence")

    # Show probability bars for both classes
    st.markdown("**Confidence breakdown:**")
    for i, name in enumerate(class_names):
        # Progress bar from 0.0 to 1.0 for each class
        st.progress(float(probs[i]), text=f"{name}: {probs[i]:.0%}")

    # Debug info — expandable section showing technical details
    with st.expander("🔧 Debug info"):
        st.markdown(f"""
        - Original image size: `{image.size[0]} × {image.size[1]} px`
        - Model input size: `{IMG_SIZE} × {IMG_SIZE} px`
        - Raw model scores: `{[round(x, 3) for x in outputs[0].tolist()]}`
        - Probabilities: `offside={probs[0]:.3f}, onside={probs[1]:.3f}`
        """)

else:
    # Shown when no image is uploaded yet
    st.info("👆 Upload a football match image to get started.")
