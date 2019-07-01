import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def gen_random_figs(net, n=(3,4), size=28):
    imgs = [torch.randn(20).to('cuda') for x in range(n[0] * n[1])]
    imgs_rec = [net.decode_single_image(t).cpu().detach() for t in imgs]

    fig = plt.figure()
    for i in range(len(imgs_rec)):
        fig.add_subplot(n[0], n[1], i + 1)
        plt.imshow(imgs_rec[i])
    plt.show()

def show_images(images):
    n = (int(len(images) /4 )+1, 4)
    imgs_rec = [t.detach() for t in images]

    fig = plt.figure()
    for i in range(len(imgs_rec)):
        fig.add_subplot(n[0], n[1], i + 1)
        plt.imshow(imgs_rec[i])
    plt.show()


def show_latent_space(other_trainer):
    X, c = other_trainer.single_pass()
    X, c = X.cpu().numpy(), c.cpu().numpy()

    X_embeddes = TSNE().fit_transform(X)

    print(X_embeddes[:, 0].shape)
    plt.scatter(X_embeddes[:, 0], X_embeddes[:, 1], c=c, label=c)
    plt.legend(loc=2)
    plt.show()

