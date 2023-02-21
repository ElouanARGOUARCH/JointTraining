import torch
from tqdm import tqdm
from torch import nn

class BinaryClassifier(nn.Module):
    def __init__(self, label_0_samples, label_1_samples, hidden_dims):
        super().__init__()
        self.label_0_samples = label_0_samples
        self.N_0 = label_0_samples.shape[0]
        self.label_1_samples = label_1_samples
        self.N_1 = label_0_samples.shape[0]
        self.p = label_0_samples.shape[-1]

        network_dimensions = [self.p] + hidden_dims + [1]
        network = []
        for h0, h1 in zip(network_dimensions, network_dimensions[1:]):
            network.extend([nn.Linear(h0, h1), nn.SiLU(), ])
        network.pop()
        self.logit_r = nn.Sequential(*network)

        self.loss_values = []

    def loss(self,label_0_samples,label_1_samples):
        log_sigmoid = nn.LogSigmoid()
        return -torch.sum(log_sigmoid(self.logit_r(label_1_samples)))-torch.sum(log_sigmoid(-self.logit_r(label_0_samples)))

    def train(self, epochs, lr = 5e-3):
        self.para_list = list(self.parameters())

        self.optimizer = torch.optim.Adam(self.para_list, lr)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        pbar = tqdm(range(epochs))
        for t in pbar:
            self.optimizer.zero_grad()
            loss = self.loss(self.label_0_samples.to(device),self.label_1_samples.to(device))
            loss.backward()
            self.optimizer.step()
            self.loss_values.append(loss)
            pbar.set_postfix_str('loss = ' + str(round(loss.item(),6))+ '; device = ' + str(device))
        self.to(torch.device('cpu'))

    def train_batch(self, epochs, batch_size, lr = 5e-3):
        assert self.label_0_samples.shape[0]==self.label_1_samples.shape[0], 'mismatch in number samples'
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if batch_size is None:
            batch_size = self.label_1_samples.shape[0]
        dataset_0 = torch.utils.data.TensorDataset(self.label_0_samples)
        dataset_1 = torch.utils.data.TensorDataset(self.label_1_samples)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        pbar = tqdm(range(epochs))
        for t in pbar:
            dataloader_0 = torch.utils.data.DataLoader(dataset_0, batch_size=batch_size, shuffle=True)
            dataloader_1 = torch.utils.data.DataLoader(dataset_1, batch_size=batch_size, shuffle=True)
            for batch_0, batch_1 in zip(dataloader_0, dataloader_1):
                label_0_batch = batch_0[0].to(device)
                label_1_batch = batch_1[0].to(device)
                optimizer.zero_grad()
                batch_loss = self.loss(label_1_batch,label_0_batch)
                batch_loss.backward()
                optimizer.step()
            with torch.no_grad():
                iteration_loss = torch.tensor([self.loss(batch_1[0].to(device),batch_0[0].to(device)) for batch_0, batch_1 in zip(dataloader_0, dataloader_1)]).sum().item()
            self.loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(round(iteration_loss,6)) + '; device = ' + str(device))
        self.to(torch.device('cpu'))

class KClassifier(nn.Module):
    def __init__(self, K, samples, labels, hidden_dimensions =[]):
        super().__init__()
        self.K = K
        assert samples.shape[0] == labels.shape[0],'number of samples does not match number of samples'
        self.samples = samples
        self.labels = labels
        self.p = samples.shape[-1]
        self.network_dimensions = [self.p] + hidden_dimensions + [self.K]
        network = []
        for h0, h1 in zip(self.network_dimensions, self.network_dimensions[1:]):
            network.extend([nn.Linear(h0, h1),nn.Tanh(),])
        self.f = nn.Sequential(*network)

    def log_prob(self, samples):
        temp = self.f.forward(samples)
        return temp - torch.logsumexp(temp, dim = -1, keepdim=True)

    def loss(self, samples,labels):
        return -torch.sum((self.log_prob(samples))[range(samples.shape[0]), labels])

    def train(self, epochs, lr = 5e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        pbar = tqdm(range(epochs))
        for _ in pbar:
            optimizer.zero_grad()
            loss = self.loss(self.samples.to(device), self.labels.to(device))
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str('loss = ' + str(round(loss.item(),4)) + '; device = ' + str(device))
        self.cpu()

    def train_batch(self, epochs,batch_size=None, lr = 5e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if batch_size is None:
            batch_size = self.samples.shape[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        dataset = torch.utils.data.TensorDataset(self.samples, self.labels)

        pbar = tqdm(range(epochs))
        for _ in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for _,batch in enumerate(dataloader):
                optimizer.zero_grad()
                loss = self.loss(batch[0].to(device), batch[1].to(device))
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                iteration_loss = torch.tensor([self.loss(batch[0].to(device), batch[1].to(device)) for _, batch in  enumerate(dataloader)]).sum().item()
            pbar.set_postfix_str('loss = ' + str(round(iteration_loss,4)) + '; device = ' + str(device))
        self.cpu()


