import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class AutoEncoder(object):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
    ):
        """This is autoencoder fitter\.

        Args:
            model: torch model.
            optimzier: torch optimzier.
            criterion: autoencoder criterion.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
    
    
    def fit(
        self,
        EPOCH,
        trainloader,
        testloader=None,
        validation_mode=True,
        scheduler=None,
        device="cuda:0",
    ):
        losses = {"train":[], "test":[]}
        for epoch in range(EPOCH):
            print(f"epoch:{epoch+1}")
            self.train(trainloader, device)
            if validation_mode:
                print("Training data results-----------------------------")
                loss = self.test(trainloader, device)
                losses["train"].append(loss)
                if testloader is not None:
                    print("Test data results---------------------------------")
                    loss = self.test(testloader, device)
                    losses["test"].append(loss)
            if scheduler is not None:
                scheduler.step()
        return losses
            
        
    def train(
        self,
        dataloader,
        device="cuda:0",
    ):
        device = torch.device(device)
        self.model.train()
        for (inputs, labels) in tqdm(dataloader):
            self.optimizer.zero_grad()
            inputs = inputs.to(device)

            outputs = self.model(inputs)

            loss = self.criterion(outputs, inputs)
            
            loss.backward()
            self.optimizer.step()
            
            
    def test(
        self,
        dataloader,
        device="cuda:0",
    ):
        sum_loss = 0.
        
        self.model.eval()
        for (inputs, labels) in tqdm(dataloader):
            inputs = inputs.to(device)
            
            outputs = self.model(inputs)

            loss = self.criterion(outputs, inputs)
            
            sum_loss += loss.item()*inputs.shape[0]
        
        sum_loss /= len(dataloader.dataset)
        
        print(f"mean_loss={sum_loss}")
        
        return sum_loss
    
    
    def getOutputs(
        self,
        dataloader,
        divece="cuda:0",
    ):
        self.model.eval()
        results = []
        for (inputs, labels) in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = self.model(inputs)
            
            results.append(outputs)
        return torch.vstack(results)
                    
        
    def setModel(self, model):
        self.model = model
    
    
    def setOptimizer(self, optimizer):
        self.optimizer = optimizer
        
        
    def setCriterion(self, criterion):
        self.criterion = criterion
        
    