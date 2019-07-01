import torch
import pickle


def save(namefile="model", model=None, epoch=0, optimizer=None):
    if model is None or optimizer is None:
        print("Save error: expected non-None parameters but got {0} = None instead".format("model" if model is None else "optimizer"))
        return

    namefile = namefile + ".pt"
    torch.save({
        'epoch':epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, namefile)



def load(namefile="model", model=None, optimizer=None):
    checkpoint = torch.load(namefile + ".pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    return checkpoint['epoch'], model, optimizer
