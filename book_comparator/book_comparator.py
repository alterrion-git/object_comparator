from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

class BookComparator(ABC):
    """Abstract base class for book comparison engines."""
    
    @abstractmethod
    def compare_images(self, query_image: NDArray[np.uint8], 
                      train_image: NDArray[np.uint8]) -> bool:
        """
        Compare two book cover images to determine if they're the same edition.
        
        Args:
            query_image: First image as numpy array
            train_image: Second image as numpy array
            
        Returns:
            bool: True if images represent the same book edition
        """
        pass