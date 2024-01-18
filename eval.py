import torch as pt
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from train import sigmoid, softmax, forward


def main():
    pt.set_printoptions(precision=10)
    model = np.load("models/model.npz", allow_pickle=True)
    weights = [pt.from_numpy(w) for w in model["weights"]]
    biases = [pt.from_numpy(b) for b in model["biases"]]
    activations = [sigmoid, sigmoid, softmax]

    mnist = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())

    amount = 10
    start = pt.randint(len(mnist) - amount, (1,))
    mnist = pt.utils.data.Subset(mnist, range(start, start + 10))

    dl = DataLoader(dataset=mnist, batch_size=amount)
    images = [img for img, _ in dl][0]

    x = images.squeeze()
    x = x.reshape(x.shape[0], -1)

    pred = forward(x, weights, biases, activations)[-1].cpu()
    pred_idx = pt.argmax(pred, dim=1)

    columns = 5
    rows = amount // columns
    fig, axs = plt.subplots(rows, columns)
    for i in range(rows):
        for j in range(columns):
            ax = axs[i][j]
            ax.set_axis_off()
            k = i * columns + j
            ax.imshow(images[k].squeeze())
            idx = pred_idx[k].item()
            ax.set_title(f"P: {idx}")
            ax.text(6, 34, f"{round(pred[k][idx].item() * 100, 2)}%")
    plt.show()
    print()


if __name__ == "__main__":
    main()
