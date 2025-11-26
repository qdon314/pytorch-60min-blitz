import torch
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
data = torch.randn(1, 3, 64, 64)
labels = torch.randn(1, 1000)

predictions = model(data) #forward pass
loss = (predictions - labels).sum()
loss.backward() #backward pass

print(model.conv1.weight.grad)

optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

optim.step() #update weights

# Differentiation

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
Q = 3*a**3 - b**2

x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does `a` require gradients?: {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")

# Freezing model parameters
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False
    
model.fc = torch.nn.Linear(512, 10) # New fc layer for 10 classes

# Optimize only the classifier
optim = torch.optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)