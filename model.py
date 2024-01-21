from typing import List, Callable, Optional

import torch as pt
from torch import Tensor

DEV = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")


def init_weights(sizes: list[int], mean: float = 0.0, std: float = 0.2) -> List[Tensor]:
    weights = [pt.normal(mean=mean, std=std, size=(sizes[i], sizes[i + 1]), device=DEV) for i in range(len(sizes) - 1)]
    return weights


def init_biases(sizes: list[int]) -> List[Tensor]:
    biases = [pt.zeros(sizes[i + 1], device=DEV) for i in range(len(sizes) - 1)]
    return biases


class Model:
    def __init__(self,
                 layer: Optional[List[int]] = None,
                 activations: Optional[List[Callable[[Tensor], Tensor]]] = None,
                 learning_rate: float = 1.,
                 adam_beta: float = 0.5,
                 weights: Optional[List[Tensor]] = None,
                 biases: Optional[List[Tensor]] = None):
        self.layer: List[int] = layer if layer is not None else []
        self.activations: List[Callable[[Tensor], Tensor]] = activations if activations is not None else []
        self.learning_rate: float = learning_rate
        self.adam_beta: float = adam_beta
        self.weights: List[Tensor] = [] if weights is None else weights
        self.biases: List[Tensor] = [] if biases is None else biases
        self._init_adam()
        self.w_backup = []
        self.b_backup = []

    def _init_adam(self):
        self.adam_w_vel = [pt.zeros_like(w) for w in self.weights]
        self.adam_b_vel = [pt.zeros_like(b) for b in self.biases]

    def init(self):
        self.weights = init_weights(self.layer)
        self.biases = init_biases(self.layer)
        self._init_adam()

    def backup(self):
        self.w_backup = [w.cpu() for w in self.weights]
        self.b_backup = [b.cpu() for b in self.biases]

    def restore(self):
        self.weights = self.w_backup
        self.biases = self.b_backup

    def save(self, path: str):
        d = {
            "weights": self.weights,
            "biases": self.biases
        }
        pt.save(d, f=path)

    def load(self, path: str):
        d = pt.load(path)
        self.weights = d["weights"]
        self.biases = d["biases"]
        self.layer = [w.shape[0] for w in self.weights] + [self.weights[-1].shape[-1]]
        self._init_adam()
