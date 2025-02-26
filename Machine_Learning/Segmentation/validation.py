from ultralytics import YOLO
import yaml



def main():
    
    with open("val_cfg.yaml", 'r') as stream:
        params = yaml.safe_load(stream)
    
    model = YOLO(params['model'],task='segment')
    
    validation_results = model.val(data=params['data'],
                                project=params['project'],
                                name= params['name'],
                                split= params['split'],
                                # half= params['half'],
                                device= params['device'],
                                batch= params['batch'],
                                save_json= params['save_json'],
                                plots= params['plots'],
                                imgsz = params['imgsz'],
                                verbose= True
    )
    inference_time_ms = validation_results.speed["inference"]  # Replace with the actual value
    fps = 1000 / inference_time_ms
    print(f"Inference Time: {inference_time_ms} ms")
    print(f"Frames Per Second (FPS): {fps:.3f}")
    
    print("-------------------------------------")
    print(validation_results.results_dict)
    print("-------------------------------------")
    print(validation_results.speed)
    # print(
    #     'map50-95: ', validation_results.box.map,'\n',    # map50-95
    #     'map50: ', validation_results.box.map50,'\n',  # map50
    #     'map75: ', validation_results.box.map75,'\n',  # map75
    #     'list map50-95: ', validation_results.box.maps)   # a list contains map50-95 of each category)
    
    return 0

if __name__ == '__main__':
    main()