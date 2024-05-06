from ultralytics import YOLO
import yaml



def main():
    
    with open("val_cfg.yaml", 'r') as stream:
        params = yaml.safe_load(stream)
    
    model = YOLO(params['model'],task='segment')
    
    validation_results = model.val(data=params['data'],
                                conf=params['threshold'],
                                project=params['save_dir']
    )
    print(
        'map50-95: ', validation_results.box.map,'\n',    # map50-95
        'map50: ', validation_results.box.map50,'\n',  # map50
        'map75: ', validation_results.box.map75,'\n',  # map75
        'list map50-95: ', validation_results.box.maps)   # a list contains map50-95 of each category)
    
    return 0

if __name__ == '__main__':
    main()