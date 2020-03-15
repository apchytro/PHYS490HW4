import torch
import torch.nn as nn
import torch.nn.functional as func

# VAE from nn_gen.py
# Encodes and decodes MNIST data

class VAE(nn.Module):

    # Initialize the net 
    def __init__(self):
        super(VAE, self).__init__()

        self.cn = nn.Conv2d(1, 3, kernel_size=3, stride=1,padding = 1)
        
        self.fc1 = nn.Linear(3 * 14 * 14, 500)
        self.fc21 = nn.Linear(500, 5)
        self.fc22 = nn.Linear(500, 5)
        self.fc3 = nn.Linear(5, 500)
        self.fc4 = nn.Linear(500, 196)
    
    # Encoding method. Ouputs mu and logvar
    def encode(self, x):
        h1= func.relu(self.cn(x))
        h1= h1.view(h1.size(0), -1)
        h1 = func.relu(self.fc1(h1))
        return self.fc21(h1), self.fc22(h1)

    # Method to sample through reparameterization 
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + torch.mul(std,eps)

    # Decoding method. Outputs generated images
    def decode(self, z):
        h3 = func.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    # Forward propagation by encoding then decoding
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # Back propagation method
    def backprop(self, data, optimizer):
        # Train model
        self.train()
        # Initialize input
        inputs= torch.from_numpy(data.x_train)

        # Check if GPU available
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        # Zero gradient optimizer
        optimizer.zero_grad()
        # Calculate outputs
        recon, mu, logvar = self(inputs)
        # Calculate losses
        obj_val= self.loss(inputs,recon,mu,logvar)
        # Back prop and optimizer step
        obj_val.backward()
        optimizer.step()
        # Return loss
        return obj_val.item()
    
    # Loss function for Gaussian distribution normalized with batch size
    def loss(self, x, recon, mu, logvar):
        BCE = func.binary_cross_entropy(recon, x.view(-1,196), reduction = 'sum')
        KLD = -0.5 * torch.sum(1 + logvar - torch.pow(mu,2) - torch.exp(logvar))
        return (BCE + KLD)/ x.size(0)
    
