

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

class LeafClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 128, 3, padding=1)

        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 38)
    def forward(self, x):
      x = torch.relu(self.conv1(x))
      x = torch.max_pool2d(x, 2)

      x = torch.relu(self.conv2(x))
      x = torch.max_pool2d(x, 2)

      x = torch.relu(self.conv3(x))
      x = torch.max_pool2d(x, 2)

      x = x.view(x.size(0), -1)

      x = torch.relu(self.fc1(x))
      x = self.fc2(x)
      return x
    
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(net:LeafClassifier, train_loader:DataLoader, num_epochs:int=1, learning_rate:float=0.001):
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    loss_list=[]
    accuracy_list=[]
    for epoch in range(num_epochs):
        total = 0
        correct = 0
        batch_count = 0
        epoch_loss = 0.0
        for i, (images, labels) in tqdm(enumerate(train_loader, 0)):
            try:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                epoch_loss += loss.item()
                batch_count += 1
            except Exception as e:
                # Keep training other batches but surface the failure
                print(f"Error in training batch {i}: {e}")
                continue
            except KeyboardInterrupt:
                break

        print()
        torch.save(net.state_dict(), "model_state.pth")
        avg_loss = epoch_loss / batch_count if batch_count else 0.0
        accuracy = (correct / total) if total else 0.0
        loss_list.append(avg_loss)
        accuracy_list.append(accuracy)
        print(f"Epoch {epoch+1} loss: {avg_loss}, accuracy: {100 * accuracy:.2f}%")
    if num_epochs==1:return
    
    epochs = range(1, num_epochs+1)
    plt.plot(epochs, loss_list, label='Training Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, accuracy_list, label='Training Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



def test_model(net:LeafClassifier,test_loader:DataLoader):
    net.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader, 0):
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the {total} test images: {100 * correct / total :.2f}%")
    return correct