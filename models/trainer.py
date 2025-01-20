import torch

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def generate_one_step(self, x):
        x = x.to(self.device)
        y = self.model(x)
        return y
    
    def train_one_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                total_loss += loss.item()
        return total_loss / len(dataloader)
    
    def train(self, train_dataloader, valid_dataloader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for x, y in train_dataloader:
                loss = self.train_one_step(x, y)
                total_loss += loss
            train_loss = total_loss / len(train_dataloader)
            valid_loss = self.evaluate(valid_dataloader)
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
        return self.model