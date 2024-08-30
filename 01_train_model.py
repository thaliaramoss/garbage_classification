from ultralytics import YOLO
import yaml

# Carregar o arquivo YAML
with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Acessar os caminhos dos diretórios
data_path = config['directories']['data_path']

# Carregar o modelo YOLOv8 pré-treinado para classificação
model = YOLO('yolov8n-cls.pt') 

# Treinar o modelo
model.train(
    data= data_path,  # Caminho para o diretório contendo as pastas 'train' e 'val'
    epochs=100,                    # Número de épocas
    imgsz=224,                    # Tamanho das imagens (224x224 é padrão para classificação)
    batch=32,                     # Tamanho do batch
    patience=10                                           
)

metrics = model.val()
print(metrics)
