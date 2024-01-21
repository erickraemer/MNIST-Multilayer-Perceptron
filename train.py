from typing import List, Tuple

import numpy as np
import torch as pt
from PIL import Image
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms.functional import pil_to_tensor

from model import Model
from one_hot_encoder import OneHotEncoder

DEV = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")
DTYPE = pt.float32


def mse_loss(y_hat: Tensor, y: Tensor):
    return pt.mean(pt.sum((y - y_hat) ** 2, dim=1), dim=0)


def mse_grad(y_hat: Tensor, y: Tensor):
    return 2. * (y_hat - y)


def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + pt.exp(-x))


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


def sigmoid_grad(sig: Tensor) -> Tensor:
    """
    Calculates the sigmoid gradient.
    :param sig: a tensor of a sigmoid values
    :return:
    """
    return sig * (1. - sig)


def forward(batch: Tensor, model: Model) -> List[Tensor]:
    """
    Will perform the forward propagation on the given mini-batch.
    :param model: the model
    :param batch: a tensor of inputs (x)
    :return: a list of tensors containing the outputs of each layer for each batch
    """

    result = [batch]
    for i in range(len(model.weights)):
        z = result[i] @ model.weights[i] + model.biases[i]
        a = model.activations[i](z)
        result.append(a)

    return result


def backward(y_hat: List[Tensor], y: Tensor, model: Model) -> tuple[list[Tensor], list[Tensor]]:
    """
    Calculates the backpropagation
    :param y: label
    :param y_hat: neuron outputs with activation function
    :param weights: weights of the network
    :param biases: biases of the network
    :return: the mean of the weight gradients and bias gradients
    """
    w_grad_means: List[Tensor] = [pt.empty_like(w, device=DEV) for w in model.weights]
    b_grad_means: List[Tensor] = [pt.empty_like(b, device=DEV) for b in model.biases]

    # gradient of the mse times the gradient of the softmax
    sm = softmax_grad(y_hat[-1])
    mse = mse_grad(y_hat[-1], y).unsqueeze(-1)
    delta = pt.bmm(sm, mse).squeeze()
    b_grad_means[-1][:] = pt.mean(delta, dim=0)
    w_grad_means[-1][:] = pt.mean(pt.einsum("bi,bj->bij", y_hat[-2], delta), dim=0)

    # inner neurons
    for i in range(len(model.weights) - 2, -1, -1):
        # weight times previous delta
        d0: Tensor = delta @ model.weights[i + 1].T
        # sigmoid derivative
        d1: Tensor = sigmoid_grad(y_hat[i + 1])
        delta = d0 * d1

        # delta times the output of the neurons from the previous layer
        b_grad_means[i][:] = pt.mean(delta, dim=0)
        w_grad_means[i][:] = pt.mean(pt.einsum("bi,bj->bij", y_hat[i], delta), dim=0)

    return w_grad_means, b_grad_means


def propagate(x: Tensor, y: Tensor, model: Model):
    # forward propagation
    y_hat = forward(x, model)

    # calculate batch loss
    loss = mse_loss(y_hat[-1], y).cpu()

    return y_hat, loss


def prepare_dataloader(dataset_path: str, batch_size: int, classes: int, num_workers: int = 2):
    target_transform = OneHotEncoder(classes)
    t_set = datasets.MNIST(dataset_path,
                           train=True, download=True,
                           transform=transform_image,
                           target_transform=target_transform
                           )

    v_set = datasets.MNIST(dataset_path,
                           train=False,
                           download=True,
                           transform=transform_image,
                           target_transform=target_transform
                           )

    t_loader = DataLoader(dataset=t_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          pin_memory=True)

    v_loader = DataLoader(dataset=v_set,
                          batch_size=batch_size * 4,
                          shuffle=True,
                          num_workers=num_workers,
                          pin_memory=True)

    return t_loader, v_loader


def plot_loss(train_loss: List[float], validation_loss: List[float], accuracy: List[float], epoch: int,
              ax: np.ndarray[plt.Axes]):
    ar = pt.arange(epoch + 1)
    ax[0].plot(ar, train_loss[:epoch + 1], "r")
    ax[0].plot(ar, validation_loss[:epoch + 1], "b")
    ax[0].legend(["Train loss", "Validation loss"], loc="upper right")
    ax[1].plot(ar, accuracy[:epoch + 1], "g")
    ax[1].legend(["Validation accuracy"], loc="lower right")
    plt.draw()
    plt.pause(0.01)


def transform_image(x: Image) -> Tensor:
    x = pil_to_tensor(x).type(pt.float32)
    x /= 255.
    x = x.reshape(-1)
    return x


def calculate_accuracy(pred: Tensor, label: Tensor):
    pred[pred <= 0.5] = 0.
    pred[pred > 0.5] = 1.

    p0 = (pred == 0.)
    p1 = ~ p0
    l0 = (label == 0.)
    l1 = ~ l0

    tp = pt.sum((p1 & l1).int())  # noqa
    fp = pt.sum((p1 & l0).int())  # noqa
    fn = pt.sum((p0 & l1).int())  # noqa
    tn = pt.sum((p0 & l0).int())  # noqa

    # calculate accuracy from the confusion matrix
    accuracy = ((tp + tn) / (tp + fp + fn + tn)).item()
    return accuracy


def propagate_model(model: Model, loader: DataLoader, forward_only=False) -> Tuple[float, float]:
    label = []
    predictions = []
    total_loss = 0

    # train network on the trainings set
    x: Tensor  # type hint
    y: Tensor  # type hint
    for batch, (x, y) in enumerate(loader):

        print(f"\r\tProcessing Batch: {round(100. / len(loader) * (batch + 1), 1)}% [ {batch + 1} / {len(loader)} ]",
              end='', flush=True)

        # send tensors to gpu if available
        x = x.to(DEV, non_blocking=True)
        y = y.to(DEV, non_blocking=True)

        # calculate forward propagation and loss
        y_hat, loss = propagate(x, y, model)

        # save loss for plotting
        total_loss += loss

        label.append(y)
        predictions.append(y_hat[-1])

        if forward_only:
            continue

        # backward propagation
        w_grad, b_grad = backward(y_hat, y, model)

        # update weights with adam
        for i in range(len(model.weights)):
            model.adam_w_vel[i] = model.adam_beta * model.adam_w_vel[i] + (1. - model.adam_beta) * w_grad[i]
            model.weights[i] -= model.learning_rate * model.adam_w_vel[i]

        # update bias with adam
        for i in range(len(model.biases)):
            model.adam_b_vel[i] = model.adam_beta * model.adam_b_vel[i] + (1. - model.adam_beta) * b_grad[i]
            model.biases[i] -= model.learning_rate * model.adam_b_vel[i]

    print(f"\r\tProcessing Batch: 100% [ {batch + 1} / {len(loader)} ]")

    label = pt.cat(label, dim=0)
    predictions = pt.cat(predictions, dim=0)
    accuracy = calculate_accuracy(predictions, label)
    total_loss /= len(loader)
    total_loss = total_loss.item()

    return total_loss, accuracy


def train():
    print(f"Using device: {DEV}")

    model = Model(
        layer=[28 * 28, 32, 16, 10],
        activations=[sigmoid, sigmoid, softmax],
        learning_rate=0.2,
        adam_beta=0.5
    )
    model.init()

    # set hyperparameters
    batch_size = 128
    max_epochs = 200
    classes = 10
    round_to = 4

    # create subplot
    fig, ax = plt.subplots(2)
    plt.ion()
    plt.show()

    # prepare image loader
    t_loader, v_loader = prepare_dataloader("data", batch_size, classes)

    # create tensor to store the loss
    val_loss = []
    train_loss = []
    accuracys = []

    model.save("models/model.pt")

    # train model
    for epoch in range(max_epochs):

        print(f".:::::::::::::::: Epoch {epoch} ::::::::::::::::.")

        # train model
        print("Training Model:")
        t_loss, _ = propagate_model(model, t_loader, forward_only=(epoch == 0))
        train_loss.append(t_loss)

        # evaluate model
        print(f"Evaluating Model:")
        v_loss, accuracy = propagate_model(model, v_loader, forward_only=True)
        accuracys.append(accuracy)
        val_loss.append(v_loss)

        # plot loss
        print(f"Validation Loss: ~{round(v_loss, round_to)} | Accuracy: ~{round(accuracy, round_to)}\n")
        plot_loss(train_loss, val_loss, accuracys, epoch, ax)

        # backup weights and biases
        model.backup()

        # stop training if validation loss is increasing again
        if epoch > 0 and val_loss[-1] > val_loss[-2]:
            break

    model.restore()
    model.save("models/model.pt")


if __name__ == "__main__":
    with pt.no_grad():
        train()
