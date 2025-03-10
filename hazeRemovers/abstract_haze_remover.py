from abc import ABC, abstractmethod

class AbstractHazeRemover(ABC):
    @abstractmethod
    def remove_haze(self):
        """
        This is an abstract method that should be implemented by all the
        haze removers. This method will be responsible for removing the haze.
        """
        pass