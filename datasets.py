import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt


import networks
import time
import losses


path = "C:\\Users\\Federico\\Downloads\\celeba-dataset\\img_align_celeba\\"

trans = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()])

class Celeba:
    def __init__(self):
        self.dataset = ImageFolder(path, transform=trans)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=40)

c = Celeba()

model = networks.VAE_CELEBA().to("cuda")
optimizer = torch.optim.RMSprop(model.parameters(), lr=5e-4)



for i in range(10):
    mega = 2**20

    model.train()
    train_loss = 0
    start_time = time.time()
    for index, (data, target) in enumerate(c.dataloader):
        data = data.to("cuda")
        optimizer.zero_grad()
        res, mean, logvar = model(data)
        loss = losses.vae_loss_function_mse(data, res, mean, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        print("Train epoch {0} ({1:.1f}% - {4}/{5}) >> loss: {2:.2f} {3:>70}\r".format(
                i,
                100*(index/(len(c.dataset)/c.dataloader.batch_size)),
                loss/c.dataloader.batch_size,
                "GPU: {0:.2f} / {1:.2f}".format(torch.cuda.memory_allocated()/mega,torch.cuda.max_memory_allocated()/mega),
                index*c.dataloader.batch_size,
                len(c.dataset)),
            end="\r")
    print("Train epoch {0} (--.-%) >> avg.loss: {1:.2f} {2:^20} ".format(
            i,
            train_loss / len(c.dataset),
            "elapsed time: {0:.2f} s.".format(time.time() - start_time)))











    # show images
    n = (3,4)
    imgs = [torch.randn(2048).to('cuda') for x in range(n[0] * n[1])]
    imgs_rec = [model.decode_single_image(t).cpu().detach() for t in imgs]

    fig = plt.figure()
    for i in range(len(imgs_rec)):
        fig.add_subplot(n[0], n[1], i + 1)
        plt.imshow(imgs_rec[i])
    plt.show()

    t = transforms.ToPILImage()
    for el in data:
        plt.imshow(t(el))
        plt.show()



