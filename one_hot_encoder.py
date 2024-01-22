from torch import Tensor
import torch as pt


def create_one_hot_encoding(classes: int) -> Tensor:
    """
    Creates a tensor of one hot encodings.
    :param classes: the amount of one hot encodings to generate.
    :return: a tensor of the shape (N x N):
    N = amount of classes
    """

    ohe = pt.zeros((classes ** 2,)).reshape((classes, classes))
    ohe.fill_diagonal_(1.)
    return ohe


class OneHotEncoder:
    """
    Class used to convert labels to a one hot encoding.
    Usage example:
    ohe = OneHotEncoder(2)
    enc = ohe(0)
    """

    def __init__(self, classes: int):
        """
        :param classes: the amount of one hot encodings to generate.
        """
        self.classes = classes
        self._enc = create_one_hot_encoding(classes)

    def __call__(self, i: int) -> Tensor:
        """
        Used to convert the input to a one hot encoding.
        :param i: input to convert.
        :return: a tensor of the shape (N,) representing a one hot encoding:
        N = number of classes
        """

        if isinstance(i, Tensor):
            i = i.item()
        return self._enc[i]
