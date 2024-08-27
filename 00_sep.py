import os
import shutil
import random
import yaml

# Carregar o arquivo YAML
with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Acessar os caminhos dos diretórios
dataset_path = config['directories']['dataset_dir']
data_path = config['directories']['data_path']

dataset_dir = dataset_path
output_dir = data_path

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

classes = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper','plastic', 'shoes', 'trash', 'white-glass']


for split in ['train', 'val', 'test']:
    for class_name in classes:
        os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

for class_name in classes:
    source_dir = os.path.join(dataset_dir, class_name)
    files = os.listdir(source_dir)
    random.shuffle(files)
    
    train_files = files[:int(len(files) * train_ratio)]
    val_files = files[int(len(files) * train_ratio):int(len(files) * (train_ratio + val_ratio))]
    test_files = files[int(len(files) * (train_ratio + val_ratio)):]
    
    for file_name in train_files:
        shutil.move(os.path.join(source_dir, file_name), os.path.join(output_dir, 'train', class_name, file_name))
    
    for file_name in val_files:
        shutil.move(os.path.join(source_dir, file_name), os.path.join(output_dir, 'val', class_name, file_name))
    
    for file_name in test_files:
        shutil.move(os.path.join(source_dir, file_name), os.path.join(output_dir, 'test', class_name, file_name))
