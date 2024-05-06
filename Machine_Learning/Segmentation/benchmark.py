from ultralytics import YOLO
import yaml
from ultralytics.utils.benchmarks import benchmark



def main():
    
    with open("benchmark_cfg.yaml", 'r') as stream:
        params = yaml.safe_load(stream)
    
    #model = YOLO()  # load a custom trained model

    #    Export the model
    #model.export(format='engine')
    # Benchmark on GPU
    benchmark(model=params['model'], data=params['data'], imgsz=640, half=False, device=0)
    
    return 0

if __name__ == '__main__':
    main()