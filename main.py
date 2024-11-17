import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import torchvision.transforms as transforms

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 28*28),
            nn.ReLU(),
            nn.Linear(28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = Generator()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist, batch_size=32, shuffle=True)

    epochs = 10
    model.train()
    for epoch in range(epochs):
        for images, labels in tqdm(train_loader):
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(torch.float).view(-1, 1).to(device)
            optimizer.zero_grad()
            output = model(labels)
            loss = criterion(output, images)
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch + 1}/{epochs}, Loss: {loss.item()}")
    torch.save(model.state_dict(), "model.pth")

    model.eval()
    with torch.no_grad():
        while (x := input("Enter a number...")) != "exit":
            x = torch.tensor([int(x)], dtype=torch.float32).to(device)
            output = model(x)
            output = output.view(28, 28).cpu().numpy()
            plt.imshow(output, cmap='gray')
            plt.show()