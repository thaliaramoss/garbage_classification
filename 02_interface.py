import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
from torchvision import transforms
import yaml
import os

# Carregar o arquivo YAML
with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Acessar os caminhos dos diretórios
model_path = os.getenv('MODEL_PATH', './runs/classify/train/weights/best.pt')

# Carregar o modelo YOLOv8 para classificação
model = YOLO(model_path)

st.header('Classificação de Resíduos com YOLOv8', divider='rainbow')

# Upload da imagem ou captura pela câmera
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    uploaded_file = st.camera_input("Tire uma foto")

if uploaded_file is not None:
    # Ler a imagem usando PIL
    image = Image.open(uploaded_file).convert('RGB')  # Garantir que está em RGB
    
    # Redimensionamento e conversão para tensor
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Tamanho comum para YOLO
        transforms.ToTensor(),
    ])
    img_tensor = transform(image)
    
    # Adicionar uma dimensão de batch
    img_tensor = img_tensor.unsqueeze(0)
    
    # Fazer a classificação
    results = model(img_tensor)
    
    # Exibir a imagem carregada
    st.image(image, caption='Imagem carregada.', use_column_width=True)

    # Mostrar as previsões
    for result in results:
        if result.probs:
            probs_tensor = result.probs.data  # Acesso ao tensor de probabilidades diretamente
            probs_array = probs_tensor.cpu().numpy()  # Converter para array NumPy, se necessário
            
            # Encontrar o índice da classe com a maior probabilidade
            max_prob_idx = probs_array.argmax()
            max_prob = probs_array[max_prob_idx]
            class_name = result.names[max_prob_idx]
            
            # Exibir a classe com a maior probabilidade
            st.subheader(f"O objeto detectado é da classe: :blue[**{class_name}**] com probabilidade de :blue[{max_prob:.2%}]")
            "---"
            st.balloons()
            st.write("Previsões:")
            # Exibir a barra de progresso para cada classe
            for idx, prob in enumerate(probs_array):
                if prob > 0:  # Mostrar apenas classes com probabilidade maior que zero
                    st.write(f"{result.names[idx]}: {prob:.2%}")
                    st.progress(int(prob * 100))