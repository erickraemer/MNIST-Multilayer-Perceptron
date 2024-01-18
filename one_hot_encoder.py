from torch import Tensor
import torch as pt


def create_one_hot_encoding(classes: int) -> Tensor:
    ohe = pt.zeros((classes ** 2,)).reshape((classes, classes))
    ohe.fill_diagonal_(1.)
    return ohe


class OneHotEncoder:
    def __init__(self, classes: int):
        self._classes = classes
        self._enc = create_one_hot_encoding(classes)

    def __call__(self, t: int) -> Tensor:
        if isinstance(t, Tensor):
            t = t.item()
        return self._enc[t]
