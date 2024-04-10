import os
import pathlib
storage = pathlib.Path(__file__).parent.parent.resolve()

def kaggleDataset(dataset_name):
    os.system('kaggle datasets download -d '+ dataset_name + ' -p '+ str(storage)+'/data/archive')
    zipPath = str(storage)+'/data/archive/'+dataset_name.split('/')[1]+'.zip'
    # Unzip
    os.system('unzip '+zipPath+' -d '+str(storage)+'/data/'+dataset_name.split('/')[1])
    return str(storage)+'/data/'+dataset_name.split('/')[1]