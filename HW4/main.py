import json, torch, argparse
import torch.optim as optim
from nn_gen import VAE
from data_gen import Data
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image

# Name: Anthony Chytros 
# ID: 20624286

# A VAE in pytorch
# Modules include : nn_gen.py and data_gen.py
# Additional files include: even_mnist.csv and param.json

if __name__ == '__main__':
    # Parser arguments
    parser = argparse.ArgumentParser(description='VAE For MNIST Data')
    parser.add_argument('-o', default = 'result_dir', metavar = 'result_dir', help = 'results directory (default: result_dir)' )
    parser.add_argument('-n', default = 100, metavar = 'result_num', help = 'number of results (default: 100)')
    parser.add_argument('--v', type=int, default=1, metavar='N', help='verbosity (default: 1)')
    parser.add_argument('--param', default = 'data\param.json', metavar='param.json', help='parameter file name')
    parser.add_argument('--data', default='data\even_mnist.csv', metavar='data_dir', help='data directory (default: even_mnist.csv)')
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    try:
        os.mkdir(args.o)
    except:
        pass
    
    # Read in json parameters 
    with open(args.param) as paramfile:
        param = json.load(paramfile)
    
    # Initialize VAE
    model = VAE()
    # Read in data
    data = Data(datafile = args.data)

    # Define an optimizer
    # Optimizer uses Adam
    optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])
    
    # Check if GPU is available 
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Initialize list of training losses and number of epochs
    train_loss= []
    num_epochs= int(param['num_epochs'])
    
    # Train the model and calculate losses
    for epoch in range(1, num_epochs + 1):

        train_val= model.backprop(data,optimizer)
        train_loss.append(train_val)
        
        # Print more detail if verbosity is high
        if args.v>=2:
            if not ((epoch + 1) % param['display_epochs']):
                print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+ '\tTraining Loss: {:.4f}'.format(train_val))
    
    # Print low verbosity final values 
    if args.v:
        print('Final training loss: {:.4f}'.format(train_loss[-1]))
        
    # Plot loss and save as loss.pdf
    fig = plt.figure()
    plt.plot(list(range(0,num_epochs)), train_loss)
    plt.xlabel('Number of Epochs')  
    plt.ylabel('Loss')  
    plt.savefig('{0}\loss.pdf'.format(args.o))
    plt.close(fig)
    
    # Generate n z tensors and save their generated images as n.pdf
    for i in range (0, args.n):
         with torch.no_grad():
            # Sample from distribution
            sample = torch.randn(1, 5)
            if torch.cuda.is_available():
               sample = sample.cuda()
            # Decode sample
            sample = model.decode(sample)
            # Save image
            save_image(sample.view(1, 1, 14, 14),'{0}\{1}.pdf'.format(args.o,(i + 1)))
    
    