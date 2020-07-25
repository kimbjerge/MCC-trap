import json
from pathlib import Path
import os

#TODO - REF HER
def readconfig(path):
    with open(path) as file:
        data = json.load(file)
    print('config loaded')
    if data['datareader']['datapath'] == "":
            base_path = Path(__file__).parent.parent.parent
            data['datareader']['datapath'] = os.path.join(base_path, '../data')
    if data['blobdetector']['backgroundpath'] == "":
            base_path = Path(__file__).parent.parent.parent
            data['blobdetector']['backgroundpath'] = os.path.join(base_path, '../config/sample_background.jpg')
    if data['classifier']['modeltype'] == "":
            base_path = Path(__file__).parent.parent.parent
            data['classifier']['modeltype'] = os.path.join(base_path, '../config/default128')
    if data['classifier']['unknown_dir'] == "":
            base_path = Path(__file__).parent.parent.parent
            data['classifier']['unknown_dir'] = os.path.join(base_path, '../unknowns')
    if data['moviemaker']['resultdir'] == "":
            base_path = Path(__file__).parent.parent.parent
            data['moviemaker']['resultdir'] = os.path.join(base_path, '../results')
    
    return data