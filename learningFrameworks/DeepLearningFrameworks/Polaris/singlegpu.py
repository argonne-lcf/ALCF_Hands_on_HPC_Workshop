import torch
#if torch 
print(f"PyTorch Version: {torch.__version__}")

device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
print(f"Using device: {device}")

#scripts based on https://github.com/argonne-lcf/ALCF_Hands_on_HPC_Workshop/blob/bd0d804c2701107840d6a3343200943bb13c8e43/learningFrameworks/PyTorch.ipynb

import torch
import torchvision
import torchvision.transforms as transforms

class ResidualBlock(torch.nn.Module):

    def __init__(self):
        # Call the parent class's __init__ to make this class functional with training loops:
        super().__init__()
        self.conv1  = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3,3], padding=[1,1])
        self.conv2  = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3,3], padding=[1,1])

    def forward(self, inputs):
    
        # Apply the first weights + activation:
        outputs = torch.nn.functional.relu(self.conv1(inputs))
        
        # Apply the second weights:
        outputs = self.conv2(outputs)

        # Perform the residual step:
        outputs = outputs + inputs

        # Second activation layer:
        return torch.nn.functional.relu(outputs)

class MyModel(torch.nn.Module):
    
    def __init__(self):
        # Call the parent class's __init__ to make this class functional with training loops:
        super().__init__()
        
        self.conv_init = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        
        self.res1 = ResidualBlock()
        
        self.res2 = ResidualBlock()
        
        # 10 filters, one for each possible label (classification):
        self.conv_final = torch.nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1)
        
        self.pool = torch.nn.AvgPool2d(32,32)
        
    def forward(self, inputs):
        
        x = self.conv_init(inputs)
        
        x = self.res1(x)
        
        x = self.res2(x)
        
        x = self.conv_final(x)
        
        return self.pool(x).reshape((-1,10))


model = MyModel()

print(model)
_num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of Trainable Parameters: {:d}".format(_num_trainable_parameters))

#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context
#https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

imagenet_data = torchvision.datasets.CIFAR10('cifar10', download=True,train=True,transform=transform)
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=2)

#TODO timer here
def gradient_step():
    img, label = next(iter(data_loader))
    logits = model(img)
    loss = torch.nn.functional.cross_entropy(logits, label.flatten())
    gradients = torch.autograd.grad(loss, model.parameters())
    return gradients

gradient_step()


#Compiled version of the model
model_c = torch.compile(model)

#TOOO timer here
def gradient_step_c():
    img, label = next(iter(data_loader))
    logits = model_c(img)
    loss = torch.nn.functional.cross_entropy(logits, label.flatten())
    gradients = torch.autograd.grad(loss, model.parameters())
    return gradients
gradient_step_c()
