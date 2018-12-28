import torch
from torchvision import datasets
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

training_data = torch.utils.data.DataLoader(datasets.MNIST('../data/', train=True, download=True, transform=transform),
                                            shuffle=True, batch_size=32)
test_data = torch.utils.data.DataLoader(datasets.MNIST('../data/', train=False, download=True, transform=transform),
                                            shuffle=True, batch_size=32)




