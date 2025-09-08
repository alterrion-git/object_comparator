from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import cv2
from PIL import Image
import google.generativeai as genai

from .book_comparator import BookComparator
from .config import get_settings


class GeminiBookComparator(BookComparator):
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
        img1_pil = Image.fromarray(query_image)
        img2_pil = Image.fromarray(train_image)

        response = self.model.generate_content([self.prompt_template, img1_pil, img2_pil])
        text = response.text.strip().lower()
        
        score_gemini = 1 if "yes" in text else 0

        return score_gemini