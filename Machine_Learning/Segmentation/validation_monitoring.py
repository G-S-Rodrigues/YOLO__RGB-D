from ultralytics import YOLO
import yaml
import time
import threading
from jtop import jtop


# --- Monitoring function ---
cpu_usage = []
gpu_usage = []
stop_monitoring = False


def main():
    
    with open("val_cfg.yaml", 'r') as stream:
        params = yaml.safe_load(stream)
    
    model = YOLO(params['model'],task='segment')
    
# --- Start the monitoring thread ---
    monitor_thread = threading.Thread(target=monitor_resources)
    monitor_thread.start()

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
    # --- Stop monitoring ---
    global stop_monitoring
    stop_monitoring = True
    monitor_thread.join()
    
    inference_time_ms = validation_results.speed["inference"]  # Replace with the actual value
    fps = 1000 / inference_time_ms
    print(f"Inference Time: {inference_time_ms} ms")
    print(f"Frames Per Second (FPS): {fps:.3f}")
    
    print("-------------------------------------")
    print(validation_results.results_dict)
    print("-------------------------------------")
    print(validation_results.speed)

    # Compute average resource usage
    avg_cpu_usage = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
    avg_gpu_usage = sum(gpu_usage) / len(gpu_usage) if gpu_usage else 0

    # print(f"gpu length: {len(gpu_usage):.2f}")
    print(f"Average CPU Usage: {avg_cpu_usage:.2f}%")
    print(f"Average GPU Usage: {avg_gpu_usage:.2f} %")
    
    return 0

def monitor_resources(interval=0.005):
    time.sleep(0.7)
    # It uses jtop to get real-time CPU and GPU utilization.
    with jtop() as jetson:
        while not stop_monitoring:
            stats = jetson.stats
            for i in range(1,13):
                cpu_util = stats[f"CPU{i}" ]
                cpu_usage.append(cpu_util)
            gpu_util = stats["GPU"]
            # print(f"Appending: CPU: {cpu_util}%, GPU: {gpu_util}%")
            gpu_usage.append(gpu_util)
            time.sleep(interval)
        
if __name__ == '__main__':
    main()