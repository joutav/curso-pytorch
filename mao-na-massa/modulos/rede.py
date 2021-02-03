import torch
from torch import nn
from torch import optim
import time

import numpy as np

class ModeloNeural(nn.Module):

    def __init__(self, input_size, hidden_size, out_size):
        super(ModeloNeural, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, out_size),
            nn.ReLU()
        )

        self.define_device()
        self.to(self.device)

    def define_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
    def forward(self, x):

        hidden = self.features(x)
        output = self.regressor(hidden)

        return output

    def otimizador(self, learning_rate, weight_decay):
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate, weight_decay=weight_decay)

    def perda(self):
        self.loss = nn.MSELoss().to(self.device)

    
    def treinamento(self, train_lodaer : torch.utils.data.DataLoader, epoch : int):
        start = time.time()
        self.train()

        epoch_loss = []

        for batch in train_lodaer:
            dado, rotulo = batch

            # cast para gpu
            dado = dado.to(self.device)
            rotulo = rotulo.to(self.device)

            # forward
            predicao = self(dado)
            loss = self.loss(predicao, rotulo)
            epoch_loss.append(loss.cpu().data)

            # backward
            loss.backward()
            self.optimizer.step()
        
        epoch_loss = np.asarray(epoch_loss)
        end = time.time()
    
        print(f'Época {epoch} - loss {epoch_loss.mean() : 0.2f} +/-{epoch_loss.std() : 0.2f} - Tempo {end-start : 0.2f}', end=' ')

        return epoch_loss.mean()

    
    def validacao(self, test_loader : torch.utils.data.DataLoader, epoch : int):
        start = time.time()
        self.eval()

        epoch_loss = []

        with torch.no_grad():
            for batch in test_loader:
                dado, rotulo = batch

                # cast para gpu
                dado = dado.to(self.device)
                rotulo = rotulo.to(self.device)

                # forward
                predicao = self(dado)
                loss = self.loss(predicao, rotulo)
                epoch_loss.append(loss.cpu().data)

        
        epoch_loss = np.asarray(epoch_loss)
        end = time.time()
    
        print(f'- val_loss {epoch_loss.mean() : 0.2f} - Tempo {end-start : 0.2f}')

        return epoch_loss.mean()

    def fit(self, quantidade_epocas, train_loader, test_loader, learning_rate, weight_decay):
        '''
        Treinamento da rede neural

        Input:
            quantidade_epocas : {int} Número de épocas para o treinamento da rede
            train_loader : {DataLoader} Conjuto de dados para treinamento
            test_loader : {DataLoader} Conjunto de dados para validacao
            learning_rate : {float} Taxa de aprendizado do modelo
            weight_decay : {float} Reduz a complexidade do modelo punindo pesos altos.

        return:
            array : Contem loss médio para cada época

        '''

        self.perda()
        self.otimizador(learning_rate, weight_decay)

        train_loss = []
        val_loss = []

        for i in range(1, quantidade_epocas+1):
            mean_loss = self.treinamento(train_loader, i)
            train_loss.append(mean_loss)
            mean_loss = self.validacao(test_loader, i)
            val_loss.append(mean_loss)

        return train_loss, val_loss


