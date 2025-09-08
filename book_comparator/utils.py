import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from .config import Settings


def load_pairs(settings: Settings) -> Dict[str, List[Tuple[Path, Path]]]:
    def _load_from_dir(base_dir: Path) -> List[Tuple[Path, Path]]:
        pairs = []
        if not base_dir.exists():
            return pairs

        for subdir in base_dir.iterdir():
            if subdir.is_dir():
                imgs = sorted(list(subdir.glob("*.*")))
                if len(imgs) >= 2:
                    pairs.append((imgs[0], imgs[1]))
        return pairs

    return {
        "positive_images": _load_from_dir(settings.POSITIVE_PAIRS_DIR),
        "negative_images": _load_from_dir(settings.NEGATIVE_PAIRS_DIR),
    }