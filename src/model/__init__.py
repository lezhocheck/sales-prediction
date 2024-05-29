from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def predict(self) -> None:
        pass