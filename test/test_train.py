import pytest

from model import Model
from train import softmax, sigmoid, softmax_grad, sigmoid_grad, forward, mse_loss, mse_grad
import torch as pt


def test_softmax():
    x = pt.arange(4).reshape((2, 2))
    y = pt.exp(x)
    y = pt.tensor([
        [y[0, 0] / (y[0, 0] + y[0, 1]), y[0, 1] / (y[0, 0] + y[0, 1])],
        [y[1, 0] / (y[1, 0] + y[1, 1]), y[1, 1] / (y[1, 0] + y[1, 1])]
    ])
    y_hat = softmax(x)

    assert pt.equal(y_hat, y)


def test_sigmoid():
    x = pt.tensor([0., -2., 2.]).reshape((3, 1))
    y = pt.tensor([[1. / 2.],
                   [1. / (1 + pt.exp(pt.tensor([2.])))],
                   [1. / (1 + pt.exp(pt.tensor([-2.])))]
                   ])

    y_hat = sigmoid(x)

    assert pt.equal(y_hat, y)


def test_softmax_grad():
    x = pt.arange(4).reshape((2, 2)).type(pt.float32)
    y = pt.tensor([
        [0., 0., 0., 0.],
        [-2., -6., -6., -6],
    ]).reshape((2, 2, 2))

    y_hat = softmax_grad(x)

    assert pt.equal(y_hat, y)


def test_sigmoid_grad():
    x = pt.tensor([0., -2., 2.]).reshape((3, 1))

    def sigmoid(x):
        return 1. / (1. + pt.exp(-x))

    y = pt.tensor([
        [sigmoid(pt.tensor(0.)) * (1. - sigmoid(pt.tensor(0.)))],
        [sigmoid(pt.tensor(2.)) * (1. - sigmoid(pt.tensor(2.)))],
        [sigmoid(pt.tensor(-2.)) * (1. - sigmoid(pt.tensor(-2.)))]
    ])

    y_hat = sigmoid_grad(sigmoid(x))

    assert pt.allclose(y_hat, y)


def test_mse():
    y_hat = pt.tensor([
        [1., 2.],
        [3., 4.]
    ])

    y = pt.tensor([
        [1., 1.],
        [2., 2.]
    ])

    loss = mse_loss(y_hat, y)
    target_loss = pt.tensor([3.])

    assert pt.allclose(loss, target_loss)


def test_mse_grad():
    y_hat = pt.tensor([
        [1., 2.],
        [3., 4.]
    ])

    y = pt.tensor([
        [1., 1.],
        [2., 2.]
    ])

    grad = mse_grad(y_hat, y)
    target_loss = pt.tensor([
        [0., 2.],
        [2., 4.]
    ])

    assert pt.allclose(grad, target_loss)


def test_forward():
    x = pt.tensor([
        [0, 1],
        [2, 3],
    ], dtype=pt.float32)

    w = pt.tensor([
        [-0.5, -0.25],
        [0.75, 0.3]
    ], dtype=pt.float32)

    b = pt.tensor([0., 1.], dtype=pt.float32)

    weights = [w, w]
    biases = [b, b]

    model = Model(
        activations=[sigmoid, softmax],
        weights=weights,
        biases=biases)

    out = forward(x, model)

    # Logits, hidden layer:
    # lh = pt.tensor([[0.7500000000, 1.2999999523],
    #         [1.2500000000, 1.4000000954]])

    # Sigmoid, hidden layer:
    sh = pt.tensor([[0.6791787148, 0.7858350277],
                    [0.7772998810, 0.8021839261]], dtype=pt.float32)

    # Logits, output layer:
    # lo = pt.tensor([[0.2497869134, 1.0659558773],
    #         [0.2129880190, 1.0463302135]])

    # Softmax, output layer:
    so = pt.tensor([[0.3065774739, 0.6934224963],
                    [0.3029388189, 0.6970611811]], dtype=pt.float32)

    assert pt.allclose(out[0], x), f"Diff: {out[0] - x}"
    assert pt.allclose(out[1], sh), f"Diff: {out[1] - sh}"
    assert pt.allclose(out[2], so), f"Diff: {out[2] - so}"
