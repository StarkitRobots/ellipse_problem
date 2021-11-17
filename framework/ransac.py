from .fabric.abstract_method import AbstractMethod
from abc import ABC, abstractmethod

class RANSAC(ABC):
    # общий метод, который будут использовать все наследники этого класса
    # абстрактный метод, который будет необходимо переопределять для каждого подкласса
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass