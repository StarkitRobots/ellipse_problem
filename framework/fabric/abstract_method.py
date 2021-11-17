from abc import ABC, abstractmethod

class AbstractMethod(ABC):
    # общий метод, который будут использовать все наследники этого класса
    def draw(self):
        print("Drew a chess piece")

    # абстрактный метод, который будет необходимо переопределять для каждого подкласса
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass

