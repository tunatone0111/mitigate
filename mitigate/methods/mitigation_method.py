from abc import ABC, abstractmethod

from PIL import Image


class MitigationMethod(ABC):
    @abstractmethod
    def mitigate(self, prompt: str, seed: int) -> Image.Image:
        raise NotImplementedError
