import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision.datasets as datasets
from torchvision import transforms

from common.cnn import ConvolutionalNeuralNetwork


def aufgabe10():
    #
    # In dieser Aufgabe sollen Sie eine CNN architektur mithilfe des PyTorch Frameworks entwickeln.
    # Implementieren Sie die drei Funktionen init, forward und test im modul common.cnn
    #
    # Initialisieren Sie in der init Funktion die Convolutional und Fully Connected Schichten
    # mit den folgenden Hyperparametern:
    # erste conv2d : in_channel:1, out_channel:10, kernel_size=5
    # zweite conv2d : in_channel:10, out_channel:20, kernel_size=5
    # erste linear: 4*4*20, 10
    # zweite linear: 10, 2
    #
    # In der forward Funktion soll der Vorwaertsdurchlauf durch die Architektur implementiert werden,
    # basierend auf dem Ablauf, welcher in Abbildung 1 auf dem Uebungsblatt dargestellt ist
    # Beachten Sie, dass Sie sowohl torch.nn als auch torch.nn.functional verwenden koennen.
    # Fuer die Schichten mit trainierbaren Parametern ist torch.nn sinnvoller.
    #

    # Fuer weitete Informationen lesen Sie die PyTorch Dokumentation fuer torch.nn.Conv2d und torch.nn.Linear:
    # https://pytorch.org/docs/stable/nn.html#conv2d
    # https://pytorch.org/docs/stable/nn.html#linear
    # fuer functional ReLU and MaxPool2d:
    # https://pytorch.org/docs/stable/nn.functional.html#max-pool2d
    # https://pytorch.org/docs/stable/nn.functional.html#relu
    #
    # Beantworten Sie zusaetzlich die folgenden Fragen:
    #
    # Warum sollten MLPs nicht zur Klassifikation von Bildern verwendet werden?
    # Warum ist es sinnvoll als Aktivierungsfunktion der versteckten Schichten die
    # Rectified Linear Unit zu verwenden?

    # Laden des MNIST Datensatzes
    dataset = datasets.MNIST(root='../../../data/torchvision',
                             transform=transforms.ToTensor(),
                             train=True,
                             download=True)

    # Zwei Klassen des MNIST Datensaztes auswählen (0 und 1)
    idx = (dataset.targets==0) | (dataset.targets==1)
    dataset.targets = dataset.targets[idx]
    dataset.data = dataset.data[idx]

    batch_size = 10
    train_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Visualisierung einer Minibatch des ausgewählten Teils des MNIST Datensatzes, der klassifiziert werden soll
    batch, target = next(iter(train_loader))
    _, ax_arr = plt.subplots(2,5)
    ax_arr = ax_arr.flatten()
    for idx, img in enumerate(batch):
        ax_arr[idx].imshow(np.squeeze(img), cmap='Greys_r')
        ax_arr[idx].axis('off')
    plt.show()

    # Initialisierung des Netzes
    cnn = ConvolutionalNeuralNetwork()

    learning_rate = 0.0001
    log_interval = 10
    test_interval = 100

    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    # Training des Netzes fure eine Epoche
    cnn.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = cnn(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Iteration: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                   batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))
        if batch_idx % test_interval == 0:
            print('Evaluatiere nach {} Iterationen...'.format(batch_idx *len(data)))
            test_loss, accuracy, correct = cnn.test(test_loader)
            print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    test_loss, correct, len(test_loader.dataset),
                    100 * accuracy))

    print('Finale Evalauierung:')
    test_loss, accuracy, correct = cnn.test(test_loader)
    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100 * accuracy))


if __name__ == '__main__':
    aufgabe10()
