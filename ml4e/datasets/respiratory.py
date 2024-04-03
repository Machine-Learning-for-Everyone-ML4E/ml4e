import os
from ml4e.datasets.data_loaders import kaggleDataset

def getKaggleRespiratoryData():
    path = kaggleDataset('vbookshelf/respiratory-sound-database')
    return path
