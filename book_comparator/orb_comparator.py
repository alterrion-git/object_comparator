from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import cv2
from PIL import Image
import google.generativeai as genai

from .book_comparator import BookComparator
from .config import get_settings


class ORBBookComparator(BookComparator):
    def __init__(self, threshold: float = 0.35, prompt_template: str = None):
        settings = get_settings()
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(settings.MODEL)
        self.threshold = threshold

        if prompt_template is None:
            prompt_template = (
                "Are these two images of the same book? Consider the cover, whether it is part of a set, "
                "if it is instead a sequel/different part etc. "
                "Respond only with a single word, yes/no. Respond yes only if you are 100% certain."
            )
        self.prompt_template = prompt_template

        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def compare_images(
        self, query_image: NDArray[np.uint8], train_image: NDArray[np.uint8]
    ) -> Tuple[bool, float, int, str]:
        kp1, des1 = self.orb.detectAndCompute(query_image, None)
        kp2, des2 = self.orb.detectAndCompute(train_image, None)

        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            score_orb = 0.0
        else:
            matches = self.bf.match(des1, des2)
            score_orb = len(matches) / max(len(kp1), len(kp2))
        
        return score_orb
