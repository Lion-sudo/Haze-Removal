from abc import ABC, abstractmethod

class AbstractHazeRemover(ABC):
    @abstractmethod
    def remove_haze(self, image):
        """
        This is an abstract method that should be implemented by all the
        haze removers. This method will be responsible for removing the haze.
        """
        pass

    @abstractmethod
    def get_method_name(self):
        """
        This is an abstract method that should be implemented by all the
        haze removers. This method will return the name of the method used.
        """
        pass
