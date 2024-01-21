import torch as pt
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets

from model import Model
from train import sigmoid, softmax, forward, transform_image


def main():
    model = Model(activations=[sigmoid, sigmoid, softmax])
    model.load("models/model.pt")

    mnist = datasets.MNIST("data", train=False, download=True, transform=transform_image)

    amount = 10
    start = pt.randint(len(mnist) - amount, (1,))
    mnist = pt.utils.data.Subset(mnist, range(start, start + 10))

    dl = DataLoader(dataset=mnist, batch_size=amount)
    images = [img for img, _ in dl][0]

    pred = forward(images, model)[-1].cpu()
    pred_idx = pt.argmax(pred, dim=1)

    images = images.reshape((amount, 1, 28, 28))

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
