from typing import List, Callable, Optional

import torch as pt
from torch import Tensor

DEV = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")


def init_weights(sizes: list[int], mean: float = 0.0, std: float = 0.2) -> List[Tensor]:
    """
    Create a list of weight tensors. Weight will be initialized by a normal distribution.
    :param sizes: an ordered list with the number of input, hidden and output layer neurons.
    :param mean: the mean to initialize the weights with.
    :param std: the std to initialize the weights with.
    :return: an ordered list of weights.
    """
    weights = [pt.normal(mean=mean, std=std, size=(sizes[i], sizes[i + 1]), device=DEV) for i in range(len(sizes) - 1)]
    return weights


def init_biases(sizes: list[int]) -> List[Tensor]:
    """
    Create an ordered list of bias tensors. Biases will be initialized by a normal distribution.
    :param sizes: a list with the number of input, hidden and output layer neurons.
    :return: an ordered list of biases.
    """
    biases = [pt.zeros(sizes[i + 1], device=DEV) for i in range(len(sizes) - 1)]
    return biases


class Model:
    """
    Represents a multilayer perceptron and holds data specific to the network.
    """
    def __init__(self,
                 layer: Optional[List[int]] = None,
                 activations: Optional[List[Callable[[Tensor], Tensor]]] = None,
                 learning_rate: float = 1.,
                 adam_beta: float = 0.5,
                 adam_decay: float = 0.999,
                 weights: Optional[List[Tensor]] = None,
                 biases: Optional[List[Tensor]] = None):
        """
        :param activations: an ordered list of activation functions.
        :param learning_rate: learning rate of the network.
        :param adam_beta: ADAM beta value. Should be in the range of [0, 1). Zero represents zero momentum from.
        previous gradients. One will only look at the previous momentum.
        :param adam_decay: ADAM decay rate. How fast the momentum will decay after each update.
        :param weights: an ordered list of weights.
        :param biases: an ordered list of biases.
        """
        self.layer: List[int] = layer if layer is not None else []
        self.activations: List[Callable[[Tensor], Tensor]] = activations if activations is not None else []
        self.learning_rate: float = learning_rate
        self.adam_beta: float = adam_beta
        self.adam_decay: float = adam_decay
        self.weights: List[Tensor] = [] if weights is None else weights
        self.biases: List[Tensor] = [] if biases is None else biases
        self._init_adam()
        self.w_backup = []
        self.b_backup = []

    def _init_adam(self):
        """
        Will reset ADAM's momentum to zero w.r.t. the current weights and biases.
        """
        self.adam_w_vel = [pt.zeros_like(w) for w in self.weights]
        self.adam_b_vel = [pt.zeros_like(b) for b in self.biases]

    def init(self, layer: List[int]):
        """
        Will initialize the network with new weights and biases according to
        description of the parameter layer.
        :param layer: an ordered list with the number of input, hidden and output layer neurons.
        """
        self.weights = init_weights(layer)
        self.biases = init_biases(layer)
        self._init_adam()
        self.layer = layer

    def backup(self):
        """
        Will internally back up weights and biases to the cpu.
        """
        self.w_backup = [w.cpu() for w in self.weights]
        self.b_backup = [b.cpu() for b in self.biases]

    def restore(self):
        """
        Will restore the internal back up of the weights and biases.
        """
        self.weights = self.w_backup
        self.biases = self.b_backup

    def save(self, path: str):
        """
        Will save the weights and biases to the given path as a torch file.
        :param path: path to save the model.
        """
        d = {
            "weights": self.weights,
            "biases": self.biases
        }
        pt.save(d, f=path)

    def load(self, path: str):
        """
        Load a model from the given torch file.
        :param path: path to the torch file (example: model.pt).
        """
        d = pt.load(path)
        self.weights = d["weights"]
        self.biases = d["biases"]
        self.layer = [w.shape[0] for w in self.weights] + [self.weights[-1].shape[-1]]
        self._init_adam()
