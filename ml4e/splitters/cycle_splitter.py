import librosa
from typing import List, Tuple


def CycleSplitter(path: str, cycles: List[Tuple[int]]):
    y, sr = librosa.load(path)
    splits = []
    for i, cycle in enumerate(cycles):
        print(f"Splitting cycle {i}")
        start, end = cycle
        splits.append(y[int(sr*start):int(sr*end)])
    return splits
