import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchsummary import summary

# Tentukan path ke folder train dan test
train_path = 'data/train'
test_path = 'data/test'

# Contoh transformasi yang dapat Anda gunakan
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Muat dataset train
train_dataset = ImageFolder(root=train_path, transform=transform)
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Muat dataset test
test_dataset = ImageFolder(root=test_path, transform=transform)
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Gunakan pre-trained ResNet18 sebagai dasar model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features

# Ganti output layer dengan nn.Linear berisi action
num_actions = 4  # misalnya, pelumasan, pembersihan cetakan, pembukaan baut, perancangan baut
model.fc = nn.Linear(num_ftrs, num_actions)

# Pilih device (CPU atau GPU jika tersedia)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Tentukan fungsi kerugian dan optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Scheduler untuk mengurangi learning rate setiap beberapa epoch
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training model
num_epochs = 13
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_data_loader)}")

# Evaluasi model pada dataset test
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Simpan model jika diperlukan
torch.save(model.state_dict(), 'pose_recognition_model_Vdua.pth')
