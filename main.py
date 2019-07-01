import torch
import torch.optim
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import os
import math
import sys

import networks
import losses
import trainer as dtrain
import immys
import utils


# parameters...............................................
train_again_VAE = False
mode = "vaegan"
load = "model_vae"
epochs = 100
starting_epoch = 50

# main..................................................
dataset_train = datasets.MNIST("/data", train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor()
]))
dataset_test = datasets.MNIST("/data", train=False, download=False, transform=transforms.Compose([
    transforms.ToTensor()
]))
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=100)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=100)

#========================================


if mode == "vae":
    net = networks.VAE_conv_net().to("cuda")
    trainer = dtrain.VAE_Trainer(model=net, dataloader=dataloader_train, loss=losses.vae_loss_function, device="cuda", optimizer= torch.optim.Adam(net.parameters(), lr=5e-4), testloader=dataloader_test)

    for i in range(epochs):
        trainer.train(i)
        trainer.test(i)
        if i % 10 == 0:
            immys.gen_random_figs(net, n=(4, 4))
    immys.gen_random_figs(net)


elif mode == "gan":
    net_G = networks.VAE_conv_net().to("cuda:0")
    net_D = networks.Discriminator_net().to("cuda:0")
    trainer = dtrain.GAN_trainer(d_model=net_D, g_model=net_G, dataloader=dataloader_train, d_optimizer= torch.optim.Adam(net_D.parameters(), lr=5e-4), g_optimizer= torch.optim.Adam(net_G.parameters(), lr=5e-4), device="cuda")
    # todo -> trainer : rmsprop

    for i in range(epochs):
        trainer.train(i)
        trainer.test(i)
        if i % 10 == 0:
            immys.gen_random_figs(net_G, n=(4, 4))
    immys.gen_random_figs(net_G)

elif mode == "vaegan":
    net_G = networks.VAE_conv_net().to("cuda:0")
    # net_G.init_weights(len(dataset_train))
    optim_g = torch.optim.RMSprop(net_G.parameters(), lr=5e-4) #torch.optim.Adam(net_G.parameters(), lr=5e-4)

    utils.load(namefile="model_gen", model=net_G, optimizer=optim_g)

    net_D = networks.Discriminator_net().to("cuda:0")
    net_D.init_weights(len(dataset_train)) # weight init
    optim_d = torch.optim.RMSprop(net_D.parameters(), lr=5e-4) #torch.optim.Adam(net_D.parameters(), lr=5e-4)

    trainer = dtrain.VAE_GAN_trainer(d_model=net_D, g_model=net_G, dataloader=dataloader_train, d_optimizer=optim_d, g_optimizer=optim_g, device="cuda:0", g_loss=losses.vae_loss_function, testLoader=dataloader_test)
    # todo -> trainer : rmsprop

    other_trainer = dtrain.VAE_Trainer(model=net_G, dataloader=dataloader_train, loss=losses.vae_loss_function,
                                       device="cuda:0", optimizer=optim_g,
                                       testloader=dataloader_test)

    immys.gen_random_figs(net_G)
    if train_again_VAE:
        for i in range(40):
            other_trainer.train(i)
        immys.gen_random_figs(net_G)

    immys.show_latent_space(other_trainer)

    models = [os.path.splitext(y)[0] for y in os.listdir() if os.path.splitext(y)[1] == ".pt"]
    numbers = [x.split("_")[-1] for x in models if x.split("_")[-2] == "D"]

    if starting_epoch is None:
        starting_epoch = 0
        if len(numbers) > 0:
            eps = max([x.split("_")[-1] for x in models if x.split("_")[-2] == "D"])
            startingepoch, net_G, optim_g = utils.load("model_vae_G_" + str(eps), model=net_G, optimizer=optim_g)
            _, net_D, optim_d = utils.load("model_vae_D_" + str(eps), model=net_D, optimizer=optim_d)

    # utils.save(model=net_G, namefile="model_gen", optimizer=optim_g)
    for i in range(starting_epoch, epochs):
        trainer.train(i)
        trainer.test(i)
        if i % 10 == 0 :
            utils.save("model_vae_D_" +str(i), model=net_D, optimizer=optim_d, epoch=i)
            utils.save("model_vae_G_" + str(i), model=net_G, optimizer=optim_g, epoch=i)
            immys.show_latent_space(trainer)
        if i % 10 == 0:
            immys.gen_random_figs(net_G, n=(4, 4))
    immys.gen_random_figs(net_G)



