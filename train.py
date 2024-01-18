import os
from typing import List, Tuple, Callable

import numpy as np
import torch as pt
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

DEV = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")
DTYPE = pt.float32


def create_network(sizes: list[int]) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Creates weights and biases
    :param sizes: a list of sizes for each layer including input and output
    :return:
    """

    weights = []
    biases = []
    for i in range(len(sizes) - 1):
        weights.append(pt.normal(mean=0., std=0.02, size=(sizes[i], sizes[i + 1]), device=DEV))
        biases.append(pt.zeros(sizes[i + 1], device=DEV))
    return weights, biases


def mse_loss(y_hat: Tensor, y: Tensor):
    return pt.mean(pt.sum((y - y_hat) ** 2, dim=1), dim=0)


def mse_grad(y_hat: Tensor, y: Tensor):
    return 2. * (y_hat - y)


def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + pt.exp(-x))


def relu(x: Tensor) -> Tensor:
    return pt.maximum(pt.zeros(1, device=DEV), x)


def leaky_relu(x: Tensor, alpha: float) -> Tensor:
    return pt.maximum(alpha * x, x)


def softmax(x: Tensor):
    """
    Calculates the softmax on the input tensor x.
    :param x: a tensor of shape B x T with B being the batch size and
    T being the length of the vector
    :return: a tensor of the shape B x T
    """
    exp = pt.exp(x)
    return exp / pt.sum(exp, dim=1)[:, None]


def softmax_grad(y: Tensor):
    outer = pt.einsum('ij,ik->ijk', y, y)
    grad = pt.stack([pt.diagflat(y[i]) - outer[i] for i in range(len(y))])
    return grad


def sigmoid_grad(sig: pt.Tensor) -> pt.Tensor:
    """
    Calculates the sigmoid gradient.
    :param sig: a tensor of a sigmoid values
    :return:
    """
    return sig * (1. - sig)


def forward(batch: Tensor, weights: List[Tensor], biases: List[Tensor],
            activations: List[Callable[[pt.Tensor], pt.Tensor]]) -> List[Tensor]:
    """
    Will perform the forward propagation on the given mini-batch.
    :param activations: a list of activation function for each layer
    :param batch: a tensor of inputs (x)
    :param weights: a list of weights
    :param biases: a list of biases
    :return: a list of tensors containing the outputs of each layer for each batch
    """
    result = [batch]
    for i in range(len(weights)):
        z = result[i] @ weights[i] + biases[i]
        a = activations[i](z)
        result.append(a)

    return result


def backward(y_hat: List[Tensor], y: Tensor, weights: List[Tensor],
             biases: List[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    """
    Calculates the backpropagation
    :param y: label
    :param y_hat: neuron outputs with activation function
    :param weights: weights of the network
    :param biases: biases of the network
    :return: the mean of the weight gradients and bias gradients
    """
    w_grad_means: List[pt.Tensor] = [pt.empty_like(w, device=DEV) for w in weights]
    b_grad_means: List[pt.Tensor] = [pt.empty_like(b, device=DEV) for b in biases]

    # gradient of the mse times the gradient of the softmax
    sm = softmax_grad(y_hat[-1])
    mse = mse_grad(y_hat[-1], y).unsqueeze(-1)
    delta = pt.bmm(sm, mse).squeeze()
    b_grad_means[-1][:] = pt.mean(delta, dim=0)
    w_grad_means[-1][:] = pt.mean(pt.einsum("bi,bj->bij", y_hat[-2], delta), dim=0)

    # inner neurons
    for i in range(len(weights) - 2, -1, -1):
        # weight times previous delta
        d0: pt.Tensor = delta @ weights[i + 1].T
        # sigmoid derivative
        d1: pt.Tensor = sigmoid_grad(y_hat[i + 1])
        delta = d0 * d1

        # delta times the output of the neurons from the previous layer
        b_grad_means[i][:] = pt.mean(delta, dim=0)
        w_grad_means[i][:] = pt.mean(pt.einsum("bi,bj->bij", y_hat[i], delta), dim=0)

    return w_grad_means, b_grad_means


def train():
    print(f"Using device: {DEV}")

    # create weights and biases
    architecture = [28 * 28, 32, 16, 10]
    activations = [sigmoid, sigmoid, softmax]
    weights, biases = create_network(architecture)

    # set hyperparameters
    batch_size = 128
    max_epochs = 50
    alpha = 0.2  # learning rate
    beta = 0.5  # adam scaler
    classes = 10
    round_to = 4

    dataset_path = "data/"
    t_set = datasets.MNIST(dataset_path, train=True, download=True, transform=transforms.ToTensor())
    v_set = datasets.MNIST(dataset_path, train=False, download=True, transform=transforms.ToTensor())

    t_loader = DataLoader(dataset=t_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2,
                          pin_memory=True)

    v_loader = DataLoader(dataset=v_set,
                          batch_size=batch_size * 4,
                          shuffle=True,
                          num_workers=2,
                          pin_memory=True)

    one_hot_enc = pt.zeros((classes ** 2,)).reshape((classes, classes))
    one_hot_enc.fill_diagonal_(1.)
    one_hot_enc.to(DEV)

    val_loss = pt.zeros(max_epochs)
    train_loss = pt.zeros(max_epochs)

    weight_backup = []
    bias_backup = []

    for epoch in range(max_epochs):

        # learning rate velocity for adam
        w_vel = [pt.zeros_like(w) for w in weights]
        b_vel = [pt.zeros_like(b) for b in biases]

        x: pt.Tensor  # type hint
        y: pt.Tensor  # type hint
        for batch, (x, y) in enumerate(t_loader):

            x = x.to(DEV, non_blocking=True)
            x = x.reshape(x.shape[0], -1)
            y = pt.stack([one_hot_enc[i] for i in y])
            y = y.to(DEV, non_blocking=True)

            # forward propagation
            y_hat = forward(x, weights, biases, activations)

            # calculate batch loss
            # loss = pt.nn.functional.mse_loss(y_hat[-1], y)
            loss = mse_loss(y_hat[-1], y).cpu()

            train_loss[epoch] += loss

            # backward propagation
            w_grad, b_grad = backward(y_hat, y, weights, biases)

            # update weights with adam
            for i in range(len(weights)):
                w_vel[i] = beta * w_vel[i] + (1. - beta) * w_grad[i]
                weights[i] -= alpha * w_vel[i]

            # update bias with adam
            for i in range(len(biases)):
                b_vel[i] = beta * b_vel[i] + (1. - beta) * b_grad[i]
                biases[i] -= alpha * b_vel[i]

            print(f"Epoch: {epoch} | Batch {batch + 1}/{len(t_loader)} loss: ~{round(loss.item(), round_to)}")

        print("")
        train_loss[epoch] /= len(t_loader)

        for batch, (x, y) in enumerate(v_loader):
            x = x.to(DEV, non_blocking=True)
            x = x.reshape(x.shape[0], -1)
            y = pt.stack([one_hot_enc[i] for i in y])
            y = y.to(DEV, non_blocking=True)

            # forward propagation
            y_hat = forward(x, weights, biases, activations)

            # calculate batch loss
            # loss = pt.nn.functional.mse_loss(y_hat[-1], y)
            loss = mse_loss(y_hat[-1], y).cpu()
            val_loss[epoch] += loss

            print(f"Epoch: {epoch} | Validation Batch {batch + 1}/{len(t_loader)}")

        val_loss[epoch] /= len(v_loader)
        print(f"\nValidation loss: ~{round(val_loss[epoch].item(), round_to)}")

        if epoch > 0 and val_loss[epoch] < pt.min(val_loss[:epoch]):
            weight_backup = [w.cpu().numpy() for w in weights]
            bias_backup = [b.cpu().numpy() for b in biases]

        # draw plot
        ar = pt.arange(epoch + 1)
        plt.plot(ar, train_loss[:epoch+1], "r")
        plt.plot(ar, val_loss[:epoch+1], "b")
        plt.legend(["train loss", "validation loss"], loc="upper right")
        plt.pause(0.01)

    os.makedirs("models", exist_ok=True)
    np.savez("models/model.npz", weights=np.array(weight_backup, dtype=object), biases=np.array(bias_backup, dtype=object))


if __name__ == "__main__":
    with pt.no_grad():
        train()
