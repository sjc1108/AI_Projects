import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    #img to pytorch tensor than normalize using Fashion-MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    #https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    #got tutorial for writing mnist_Dataset such as donwload = true 
    mnist_dataset = datasets.FashionMNIST('./data', train = training, download = True, transform = transform) #opt bool arg w/ default of True 

    data_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size = 64, shuffle = training) ##dataloading object for the data w/ b size 64, shuffling for training.

    return data_loader



def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    #https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
    #tutorial for seq container ex

    model = nn.Sequential(
        nn.Flatten(),   #2-D 28x28 to 1-D tensor of 784 feat
        nn.Linear(784,128),  #connected layers
        nn.ReLU(),  #use ReLU act fxn
        nn.Linear(128,64), #conn layer
        nn.ReLU(), 
        nn.Linear(64,10) #10 class of fashion-MNIST data
    )

    return model



def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """

    #https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
    #got tutorial for writing the train and test network

    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9) #init sgd
    model.train()
    
    for epoch in range(T): #epoch looping over, like reading books many times to understand the matl, ml the same way
        tot_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad() #reset to zero so doesnt accumulate
            outputs = model(images) 
            loss = criterion(outputs, labels)
            loss.backward() #calc grad
            optimizer.step() #small steps
            
            tot_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
        accuracy = 100 * correct/total
        avg_loss = tot_loss /total
        print(f"Train Epoch: {epoch} Accuracy: {correct}/{total}({accuracy:.2f}%) Loss: {avg_loss:.3f}")



def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()

    tot_loss = 0.0
    correct = 0
    total = 0

    #https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
    #tutorial for the writing of torch.no_grad
    with torch.no_grad(): #no store grad

        for ima_data, labels in test_loader:
            outputs = model(ima_data)
            loss = criterion(outputs,labels)
            tot_loss += loss.item()*ima_data.size(0)
            _, predicted = torch.max(outputs.data,1) #1 = class, _ catches values
            total += labels.size(0) #tot num samples processed
            correct += (predicted ==labels).sum().item() 
            
    accuracy = 100 * correct/total
    avg_loss = tot_loss/total

    if show_loss:
        print(f"Average loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")



def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    #https://pytorch.org/docs/stable/generated/torch.topk.html
    #tutorial provided for writing torch topk

    model.eval()

    image = test_images[index].unsqueeze(0) #unsqueeze(0) adds batch d > batch w/ single image
    logits = model(image)

    prob = F.softmax(logits, dim = 1) #softmax function to logits for prob/ dim=1 so softmax is applied acros classes
    top_proba, top_i = prob.topk(3, dim = 1)  #top 3 prob and class labales

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    for idx in range(3):
        class_name = class_names[top_i[0][idx].item()]
        probability = top_proba[0][idx].item() * 100
        print(f"{class_name}: {probability:.2f}%")


