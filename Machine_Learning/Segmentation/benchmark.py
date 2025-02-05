from ultralytics import YOLO
import yaml
from ultralytics.utils.benchmarks import benchmark



def main():
    
    with open("benchmark_cfg.yaml", 'r') as stream:
        params = yaml.safe_load(stream)
    
    # Benchmark on GPU
    benchmark(model=params['model'], 
              data=params['data'],
              imgsz=640,
            #   half=params['half'], 
              device=params['device']
            )
    
    return 0

if __name__ == '__main__':
    main()