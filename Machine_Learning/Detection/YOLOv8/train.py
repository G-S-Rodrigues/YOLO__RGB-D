#!/.venv

#from IPython import display
#display.clear_output()

from ultralytics import YOLO
import os

data_path = 'trunk_dataset.yaml'
model_path = '/home/guilh/Vineyard_autonomous_robot/DL_Trunk_Detection/YOLOv8/runs_n/train7/weights/last.pt'
#Número total de épocas de treino. Cada época representa uma passagem completa por todo o conjunto de dados. O ajuste deste valor pode afetar a duração do treino e o desempenho do modelo.
n_epochs = 50
#Tamanho do lote para treino, indicando quantas imagens são processadas antes de os parâmetros internos do modelo serem actualizados. AutoBatch (batch=-1) ajusta dinamicamente o tamanho do lote com base na disponibilidade de memória da GPU.
bs = -1 # 16
gpu_id = 0
#Ativa a saída detalhada durante o treino, fornecendo registos detalhados e actualizações de progresso. É útil para depurar e monitorizar de perto o processo de formação
verbose = False
validate = True
# Reinicia o treinamento a partir do último ponto de verificação salvo. Carrega automaticamente os pesos do modelo, o estado do otimizador e a contagem de épocas, continuando o treinamento sem problemas.
resume= True
patience=25 # 100 Número de épocas a aguardar sem melhoria nas métricas de validação antes de parar o treino antecipadamente. Ajuda a evitar o sobreajuste, interrompendo o treino quando o desempenho atinge um patamar.

# Specify the save directory for training runs
save_dir = '/home/guilh/Vineyard_autonomous_robot/DL_Trunk_Detection/YOLOv8/runs_n'
os.makedirs(save_dir, exist_ok=True)

# Load and train the model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO(model_path)
results = model.train(
    data=data_path,
    epochs=n_epochs,
    batch=bs,
    device=gpu_id,
    verbose=verbose,
    val=validate,
    project=save_dir,
    resume=resume
)
