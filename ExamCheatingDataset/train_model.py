import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import os
import time
import copy
import numpy as np
from PIL import Image

# ✅ Use absolute path to avoid FileNotFoundError
data_dir = 'c:/Users/Tinova/Documents/test projects/ExamCheatingDataset'
train_dir = os.path.join(data_dir, 'train')
test_image_dir = os.path.join(data_dir, 'test', 'images')

# ✅ Parameters
num_classes = 5
batch_size = 16
num_epochs = 10
learning_rate = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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

# ✅ Load training dataset
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found: {train_dir}")

train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
class_names = train_dataset.classes
print("Class Names:", class_names)

# ✅ Compute class weights
labels = [label for _, label in train_dataset.imgs]
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# ✅ Custom dataset for test images
class TestImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [os.path.join(image_dir, fname)
                            for fname in os.listdir(image_dir)
                            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)

# ✅ Load test data (unlabeled)
if not os.path.exists(test_image_dir):
    raise FileNotFoundError(f"Test image directory not found: {test_image_dir}")

test_dataset = TestImageDataset(test_image_dir, transform=data_transforms['test'])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ✅ Initialize model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# ✅ Freeze all layers except last layer and layer4
for name, param in model.named_parameters():
    param.requires_grad = ("layer4" in name) or ("fc" in name)

model = model.to(device)

# ✅ Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# ✅ Training function
def train_model(model, criterion, optimizer, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\nTraining complete. Best train accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model

# ✅ Train the model
model = train_model(model, criterion, optimizer, num_epochs)

# ✅ Run inference on test images
model.eval()
all_filenames = []
all_preds = []

with torch.no_grad():
    for inputs, filenames in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_filenames.extend(filenames)
        all_preds.extend(preds.cpu().numpy())

# ✅ Show predictions
print("\nPredictions on test images:")
for fname, pred in zip(all_filenames, all_preds):
    print(f"{fname}: {class_names[pred]}")
