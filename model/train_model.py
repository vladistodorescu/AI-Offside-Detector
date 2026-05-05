"""
=======================================================
  AI OFFSIDE DETECTOR — Model Training Script
  File: model/train_model.py
  
  What this script does:
  1. Downloads 102 labeled football images from Roboflow
  2. Loads them into PyTorch with augmentation
  3. Builds a ResNet18 CNN using Transfer Learning
  4. Trains the CNN to classify: offside vs onside
  5. Saves the best model to offside_cnn.pth
  
  Run from the project root:
      python3 model/train_model.py
=======================================================
"""

# ── Imports ────────────────────────────────────────────────────────────────────

import os                          # for file paths and directory checks
import sys                         # for exiting the script on errors
import torch                       # PyTorch — the main deep learning framework
import torch.nn as nn              # neural network building blocks (layers, loss functions)
from torch.utils.data import DataLoader      # loads images in batches during training
from torchvision import datasets, models, transforms  # image datasets, pretrained models, transforms
from roboflow import Roboflow      # Roboflow SDK — to download our labeled dataset


# ══════════════════════════════════════════════════════
#  CONFIGURATION — change these values to experiment
# ══════════════════════════════════════════════════════

API_KEY    = "voTL0eJRr4ZT5IYz59Vq"   # ⚠️ your Roboflow API key (keep this secret!)
WORKSPACE  = "vlad-istodorescu"    # your Roboflow workspace name
PROJECT    = "offside-dnedg-kntfo" # your Roboflow project name
VERSION    = 1                     # which version of the dataset to download

IMG_SIZE   = 224      # resize all images to 224×224 pixels (ResNet18 standard input size)
BATCH_SIZE = 8        # how many images to process at once (small = safer for limited memory)
EPOCHS     = 20       # how many times the model sees the full training set
LR         = 0.001    # learning rate — how big each learning step is (too big = unstable, too small = slow)

# where to save the trained model weights after training
MODEL_PATH = os.path.join(os.path.dirname(__file__), "offside_cnn.pth")

# where Roboflow will download the images to
DATA_DIR   = "data/dataset"


# ══════════════════════════════════════════════════════
#  STEP 1 — Download the dataset from Roboflow
# ══════════════════════════════════════════════════════

def download_dataset():
    """
    Connects to Roboflow and downloads our labeled dataset.
    The dataset contains 102 images labeled as 'offside' or 'onside'.
    Roboflow organises them into train/ valid/ test/ folders automatically.
    """
    print("\n[1/4] Downloading dataset from Roboflow...")

    rf      = Roboflow(api_key=API_KEY)          # authenticate with Roboflow
    project = rf.workspace(WORKSPACE).project(PROJECT)  # navigate to our project
    version = project.version(VERSION)            # select dataset version 1
    dataset = version.download("folder", location=DATA_DIR)  # download in folder format

    print(f"      → Downloaded to: {DATA_DIR}")
    return dataset


# ══════════════════════════════════════════════════════
#  STEP 2 — Load and prepare the images
# ══════════════════════════════════════════════════════

def load_data():
    """
    Loads the downloaded images into PyTorch DataLoaders.
    
    Applies two types of operations:
    - Transforms: resize, convert to tensor, normalise pixel values
    - Augmentations: randomly flip/adjust images to create artificial variety
      (important because we only have 102 images — augmentation helps prevent overfitting)
    """
    print("\n[2/4] Loading images...")

    # ── Training transforms (with augmentation) ──────────────────────────────
    # These are applied randomly during training to artificially expand the dataset
    train_transforms = transforms.Compose([

        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # Resize every image to 224×224 — ResNet18 requires a fixed input size

        transforms.RandomHorizontalFlip(),
        # Randomly mirror the image left-right (50% chance each time)
        # An offside situation looks similar whether viewed from left or right

        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        # Randomly adjust brightness and contrast
        # Helps the model handle different lighting (sunny day vs cloudy, night games)

        transforms.ToTensor(),
        # Convert PIL image (0-255 pixel values) to PyTorch tensor (0.0-1.0 float values)

        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        # Normalise using ImageNet mean and standard deviation
        # This is required because ResNet18 was pretrained on ImageNet
        # with these exact normalisation values — we must match them
    ])

    # ── Validation transforms (no augmentation) ──────────────────────────────
    # For validation we only resize and normalise — no random changes
    # We want consistent results so we can accurately measure performance
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # ── Load images from folders ──────────────────────────────────────────────
    # Roboflow creates this structure when you download in "folder" format:
    #   data/dataset/
    #       train/
    #           offside/    ← images labeled offside
    #           onside/     ← images labeled onside
    #       valid/
    #           offside/
    #           onside/
    train_dir = os.path.join(DATA_DIR, "train")
    valid_dir = os.path.join(DATA_DIR, "valid")

    if not os.path.exists(train_dir):
        print(f"ERROR: {train_dir} not found. Dataset may not have downloaded correctly.")
        sys.exit(1)

    # ImageFolder automatically reads class names from subfolder names
    # So it learns: offside/ → class 0, onside/ → class 1
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=val_transforms)

    # DataLoader feeds images to the model in batches during training
    # shuffle=True for training so the model doesn't memorise the order
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = train_dataset.classes  # ['offside', 'onside']
    print(f"      → Classes found: {class_names}")
    print(f"      → Training images: {len(train_dataset)}")
    print(f"      → Validation images: {len(valid_dataset)}")

    return train_loader, valid_loader, class_names


# ══════════════════════════════════════════════════════
#  STEP 3 — Build the CNN using Transfer Learning
# ══════════════════════════════════════════════════════

def build_model(num_classes=2):
    """
    Builds a ResNet18 model using Transfer Learning.

    Transfer Learning explained:
    - ResNet18 was originally trained on ImageNet (1.2 million images, 1000 classes)
    - It already knows how to detect: edges, shapes, textures, objects
    - Instead of training from scratch (which needs millions of images),
      we REUSE this knowledge and only teach it our specific task (offside vs onside)
    - This is like hiring an expert photographer and just teaching them football rules,
      rather than teaching someone to both photograph AND understand football from zero.
    """
    print("\n[3/4] Building ResNet18 model (Transfer Learning)...")

    # Load ResNet18 with pretrained ImageNet weights
    # These weights represent millions of images of learned knowledge
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # FREEZE all layers — we don't want to overwrite the ImageNet knowledge
    # requires_grad=False means these layers won't be updated during training
    for param in model.parameters():
        param.requires_grad = False

    # REPLACE only the final classification layer
    # Originally: 512 features → 1000 ImageNet classes
    # Our version: 512 features → 2 classes (offside / onside)
    # This is the only layer that will actually be trained
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # model.fc.in_features = 512 (ResNet18 output size)
    # num_classes = 2 (offside or onside)

    print("      → ResNet18 loaded with pretrained ImageNet weights")
    print("      → All layers frozen except the final classification layer")
    print(f"      → Final layer: 512 features → {num_classes} classes")

    return model


# ══════════════════════════════════════════════════════
#  STEP 4 — Train the model
# ══════════════════════════════════════════════════════

def train(model, train_loader, valid_loader):
    """
    The core training loop.
    
    How neural network training works:
    1. Feed a batch of images through the model → get predictions
    2. Compare predictions to correct labels → calculate loss (how wrong we are)
    3. Backpropagate: figure out which weights caused the error
    4. Update weights slightly in the right direction (optimizer step)
    5. Repeat for all batches → that's 1 epoch
    6. Repeat for all epochs
    
    After each epoch we check performance on the validation set
    (images the model has never seen) to measure real accuracy.
    """
    print(f"\n[4/4] Training for {EPOCHS} epochs...")

    # Use Apple Silicon GPU (MPS) if available, otherwise CPU
    # MPS = Metal Performance Shaders — Apple's GPU acceleration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model  = model.to(device)  # move model to the selected device

    # Loss function: CrossEntropyLoss is standard for classification problems
    # It measures how different our predictions are from the true labels
    criterion = nn.CrossEntropyLoss()

    # Optimizer: Adam adjusts the model weights to reduce the loss
    # We only optimize model.fc.parameters() — the final layer we unfroze
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

    # Scheduler: reduces learning rate by 90% every 7 epochs
    # Starts with bigger steps, then fine-tunes with smaller steps
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print(f"      → Training on: {device}")

    best_acc = 0.0  # track best validation accuracy to save the best model

    for epoch in range(EPOCHS):

        # ── Training phase ───────────────────────────────────────────────────
        model.train()  # set model to training mode (enables dropout, batch norm etc.)
        running_loss, running_correct = 0.0, 0

        for inputs, labels in train_loader:
            # Move data to the same device as the model
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            # Reset gradients from the previous batch
            # If we don't do this, gradients accumulate and corrupt training

            outputs = model(inputs)
            # Forward pass: feed images through the CNN, get raw scores (logits)
            # outputs shape: [batch_size, 2] — one score per class per image

            loss = criterion(outputs, labels)
            # Calculate how wrong our predictions are compared to true labels

            loss.backward()
            # Backpropagation: calculate gradients for each weight
            # This tells us "which weights contributed to the error?"

            optimizer.step()
            # Update weights using the gradients — take one step toward better predictions

            running_loss    += loss.item()
            running_correct += (outputs.argmax(1) == labels).sum().item()
            # argmax(1) picks the class with the highest score as our prediction

        train_acc = running_correct / len(train_loader.dataset)

        # ── Validation phase ─────────────────────────────────────────────────
        model.eval()   # set model to evaluation mode (disables dropout etc.)
        val_correct = 0

        with torch.no_grad():
            # torch.no_grad() disables gradient calculation — we don't need it for evaluation
            # This saves memory and speeds up the validation loop
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs      = model(inputs)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(valid_loader.dataset)

        scheduler.step()  # update the learning rate according to our schedule

        print(f"  Epoch {epoch+1:02d}/{EPOCHS}  "
              f"Loss: {running_loss/len(train_loader):.3f}  "
              f"Train Acc: {train_acc:.0%}  "
              f"Val Acc: {val_acc:.0%}")

        # Save the model whenever we beat the previous best validation accuracy
        # We save based on VALIDATION accuracy (not training) to avoid overfitting
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),  # all the learned weights
                "class_names": ["offside", "onside"],
                "img_size":    IMG_SIZE,
            }, MODEL_PATH)
            print(f"             ✅ New best model saved! (val acc: {val_acc:.0%})")

    print(f"\n  Training complete!")
    print(f"  Best validation accuracy: {best_acc:.0%}")
    print(f"  Model saved → {MODEL_PATH}")


# ══════════════════════════════════════════════════════
#  ENTRY POINT — runs when you execute this script
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("   AI Offside Detector — CNN Training (ResNet18)")
    print("=" * 55)

    download_dataset()                                      # Step 1
    train_loader, valid_loader, class_names = load_data()  # Step 2
    model = build_model(num_classes=len(class_names))      # Step 3
    train(model, train_loader, valid_loader)                # Step 4
