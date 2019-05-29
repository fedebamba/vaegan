import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Weighted_net(nn.Module):
    def __init__(self):
        super(Weighted_net, self).__init__()

    def init_weights(self, n):
        def initf(m):
            if m.__class__.__name__.find("Linear") != -1 or m.__class__.__name__.find("Conv2d") != -1:
                m.weight.data.normal_(0, 1/np.sqrt(n))
                m.bias.data.fill_(0)
        self.apply(initf)


class VAE_net(nn.Module): # questa si potrebbe buttare via
    def __init__(self):
        super(VAE_net, self).__init__()
        self.fc1_encode = nn.Linear(784, 400)
        self.fc2_encode_mean = nn.Linear(400, 20)
        self.fc2_encode_sigma = nn.Linear(400, 20)
        self.fc1_decode = nn.Linear(20, 400)
        self.fc2_decode = nn.Linear(400, 784)

    def encode(self, x):
        x = F.relu(self.fc1_encode(x))
        x_mean, x_sigma = self.fc2_encode_mean(x), self.fc2_encode_sigma(x)
        return x_mean, x_sigma

    def gaussian_sampling(self, mean, sigma):
        std = torch.exp(sigma*.5)
        es = torch.randn_like(std)
        return mean + es*std

    def decode(self, sample):
        sample = F.relu(self.fc1_decode(sample))
        return torch.sigmoid(self.fc2_decode(sample))

    def forward(self, x):
        x = x.view(-1, 784)
        mean, var = self.encode(x)
        z = self.gaussian_sampling(mean, var)
        return self.decode(z), mean, var


class VAE_conv_net(Weighted_net):
    def __init__(self):
        super(VAE_conv_net, self).__init__()
        self.conv1_encode = nn.Conv2d(1, 20, 5) # 24x24x20
        self.bn_conv1 = nn.BatchNorm2d(20)
        self.conv2_encode = nn.Conv2d(20, 64, 5)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.fc1_encode = nn.Linear(1024, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2_encode_mean = nn.Linear(256, 20)
        self.fc2_encode_sigma = nn.Linear(256, 20)
        self.fc1_decode = nn.Linear(20, 256)
        self.bn_fc1_decode = nn.BatchNorm1d(256)
        self.fc2_decode = nn.Linear(256, 400)
        self.bn_fc2_decode = nn.BatchNorm1d(400)
        self.fc3_decode = nn.Linear(400, 784)

    def encode(self, x):
        x = F.relu(self.conv1_encode(x))
        x = self.bn_conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2_encode(x))
        x = self.bn_conv2(x)
        x = F.max_pool2d(x, 2)

        x = self.fc1_encode(x.view(-1, 1024))
        x = self.bn_fc1(x)
        return self.fc2_encode_mean(x), self.fc2_encode_sigma(x)

    def gaussian_sampling(self, mean, sigma):
        std = torch.exp(sigma*.5)
        es = torch.randn_like(std)
        return mean + es*std

    def decode(self, sample):
        sample = F.relu(self.fc1_decode(sample))
        # if len(sample) > 1:
        #     sample = self.bn_fc1_decode(sample)
        sample = F.relu(self.fc2_decode(sample))
        # if len(sample) > 1:
        #       sample = self.bn_fc2_decode(sample)
        res = torch.sigmoid(self.fc3_decode(sample))
        return res

    def decode_single_image(self, sample):
        sample = sample.reshape(1, 20)
        x =  self.decode(sample)
        return x.reshape(28,28)

    def decode_image(self, sample):
        x = torch.Tensor().to("cuda")
        for i in range(len(sample)):
            x = torch.cat((x, self.decode(sample[i]).reshape(1,1,28,28)), 0)
        return x

    def forward(self, x):
        mean, var = self.encode(x)
        z = self.gaussian_sampling(mean, var)
        return self.decode(z), mean, var


class Discriminator_net(Weighted_net):
    def __init__(self):
        super(Discriminator_net, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 5)
        self.bn_conv1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 40, 5)
        self.bn_conv2 = nn.BatchNorm2d(40)
        self.fc1 = nn.Linear(640, 1)
        self.final = nn.Sigmoid()

    def forward(self, x, hidden=False):
        x = self.bn_conv1(F.relu(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x1 = self.bn_conv2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x1, 2)

        x = x.view(-1, 640)
        x = self.fc1(x)
        if hidden:
            return self.final(x), x1
        else:
            return self.final(x)
