#!/.venv

#from IPython import display
#display.clear_output()

from ultralytics import YOLO
import os
from ultralytics.models.yolo.segment import SegmentationTrainer
import yaml




def main():
    
    with open("train_cfg.yaml", 'r') as stream:
        params = yaml.safe_load(stream)
        
    model = SegmentationTrainer(overrides=params)
    results = model.train()
    return 0

if __name__ == '__main__':
    main()