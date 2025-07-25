import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import os
import time
import copy
import numpy as np
from PIL import Image

# --- Configuration ---
st.set_page_config(page_title="Cheating Detector", layout="wide")
st.title("📸 Exam Cheating Detector")

data_dir = 'c:/Users/Tinova/Documents/test projects/ExamCheatingDataset'
train_dir = os.path.join(data_dir, 'train')
test_image_dir = os.path.join(data_dir, 'test', 'images')
model_path = "cheating_model.pth"

# --- Data Transforms ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# --- Load Train Dataset ---
if not os.path.exists(train_dir):
    st.error(f"Training directory not found: {train_dir}")
    st.stop()

train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
class_names = train_dataset.classes
st.write("🔢 **Classes:**", class_names)

# --- Compute Class Weights ---
labels = [y for _, y in train_dataset.imgs]
class_weights = torch.tensor(
    compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels),
    dtype=torch.float
)

# --- Define Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
for n, p in model.named_parameters():
    p.requires_grad = ("layer4" in n) or ("fc" in n)
model = model.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# --- Train Model ---
if st.button("▶️ Train Model"):
    with st.spinner("Training..."):
        since = time.time()
        best_wts, best_acc = copy.deepcopy(model.state_dict()), 0
        for epoch in range(10):
            model.train()
            running_loss = running_corrects = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x); _, preds = out.max(1)
                loss = criterion(out, y)
                loss.backward(); optimizer.step()
                running_loss += loss.item() * x.size(0)
                running_corrects += torch.sum(preds == y)
            acc = running_corrects.double() / len(train_dataset)
            st.write(f"Epoch {epoch+1}, Acc: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_wts)
        torch.save(model.state_dict(), model_path)
        st.success(f"✅ Training done. Best train accuracy: {best_acc:.4f}")
        st.write(f"⏱️ Time elapsed: {time.time() - since:.0f}s")

# --- Load Model if Already Trained ---
elif os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    st.info("📦 Loaded trained model from file.")

# --- Predict Test Images from Folder ---
if st.button("🔍 Predict Test Images"):
    if not os.path.exists(test_image_dir):
        st.error(f"Test folder not found: {test_image_dir}")
        st.stop()
    test_files = [f for f in os.listdir(test_image_dir) if f.lower().endswith(('.jpg', '.png'))]
    if not test_files:
        st.warning("No test images found.")
    else:
        cols = st.columns(4)
        model.eval()
        for i, fname in enumerate(test_files):
            img = Image.open(os.path.join(test_image_dir, fname)).convert("RGB")
            x = data_transforms['test'](img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(x)
                pred = out.argmax().item()
                prob = torch.nn.functional.softmax(out, dim=1)[0][pred].item()
            cols[i % 4].image(img, caption=f"{fname}\n→ {class_names[pred]} ({prob:.2%})", use_column_width=True)

# --- Predict from Uploaded Image ---
st.markdown("### 📤 Upload Image for Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)
    x = data_transforms['test'](img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        pred = out.argmax().item()
        prob = torch.nn.functional.softmax(out, dim=1)[0][pred].item()
    st.success(f"Prediction: **{class_names[pred]}** ({prob:.2%})")