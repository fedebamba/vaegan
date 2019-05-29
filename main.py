import torch
import torch.optim
from torchvision import datasets, transforms


import networks
import losses
import trainer as dtrain
import immys

mode = "vaegan"
epochs = 100



# main..................................................
dataset_train = datasets.MNIST("/data", train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor()
]))
dataset_test = datasets.MNIST("/data", train=False, download=False, transform=transforms.Compose([
    transforms.ToTensor()
]))
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=32)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=32)

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
    net_G = networks.VAE_conv_net().to("cuda")
    net_D = networks.Discriminator_net().to("cuda")
    trainer = dtrain.GAN_trainer(d_model=net_D, g_model=net_G, dataloader=dataloader_train, d_optimizer= torch.optim.Adam(net_D.parameters(), lr=5e-4), g_optimizer= torch.optim.Adam(net_G.parameters(), lr=5e-4), device="cuda")
    # todo -> trainer : rmsprop

    for i in range(epochs):
        trainer.train(i)
        trainer.test(i)
        if i % 10 == 0:
            immys.gen_random_figs(net_G, n=(4, 4))
    immys.gen_random_figs(net_G)

elif mode == "vaegan":
    net_G = networks.VAE_conv_net().to("cuda")
    # net_G.init_weights(len(dataset_train))
    optim_g = torch.optim.RMSprop(net_G.parameters(), lr=5e-4) #torch.optim.Adam(net_G.parameters(), lr=5e-4)

    net_D = networks.Discriminator_net().to("cuda")
    net_D.init_weights(len(dataset_train)) # weight init
    optim_d = torch.optim.RMSprop(net_D.parameters(), lr=5e-4) #torch.optim.Adam(net_D.parameters(), lr=5e-4)

    trainer = dtrain.VAE_GAN_trainer(d_model=net_D, g_model=net_G, dataloader=dataloader_train, d_optimizer=torch.optim.Adam(net_D.parameters(), lr=5e-4), g_optimizer=torch.optim.Adam(net_G.parameters(), lr=5e-4), device="cuda", g_loss=losses.vae_loss_function)
    # todo -> trainer : rmsprop

    other_trainer = dtrain.VAE_Trainer(model=net_G, dataloader=dataloader_train, loss=losses.vae_loss_function,
                                       device="cuda", optimizer=torch.optim.Adam(net_G.parameters(), lr=5e-4),
                                       testloader=dataloader_test)
    for i in range(1):
        other_trainer.train(i)
    immys.gen_random_figs(net_G)

    for i in range(epochs):
        trainer.train(i)
        trainer.test(i)
        if i % 40 == 0:
            immys.gen_random_figs(net_G, n=(4, 4))
    immys.gen_random_figs(net_G)
