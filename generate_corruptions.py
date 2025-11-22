import os
from PIL import Image
import numpy as np
from endoscopycorruptions import corrupt, get_corruption_names
from tqdm import tqdm

# Rutas base
input_root = "/workspace/datasets/hamlyn"
output_root = "/workspace/datasets/hamlyn/hamlyn_corruptions"

# Obtener todas las corrupciones disponibles
corruption_types = get_corruption_names()
severities = [1, 2, 3, 4, 5]

# Recorrer datasets
for dataset in os.listdir(input_root):
    dataset_path = os.path.join(input_root, dataset)
    if not os.path.isdir(dataset_path):
        continue

    for keyframe in os.listdir(dataset_path):
        keyframe_path = os.path.join(dataset_path, keyframe)
        data_path = os.path.join(keyframe_path, "data")
        if not os.path.isdir(data_path):
            continue

        image_filenames = [f for f in os.listdir(data_path) if f.endswith(".jpg")]

        for img_name in tqdm(image_filenames, desc=f"{dataset}/{keyframe}"):
            img_path = os.path.join(data_path, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img_np = np.asarray(img)

                for corr in corruption_types:
                    for sev in severities:
                        # Aplicar corrupción
                        try:
                            img_corr = corrupt(img_np, corruption_name=corr, severity=sev)
                        except Exception as e:
                            print(f"Error con {img_name} corrupción {corr} nivel {sev}: {e}")
                            continue

                        # Ruta de salida
                        out_dir = os.path.join(output_root, corr, f"severity_{sev}", dataset, keyframe, "data")
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, img_name)

                        Image.fromarray(img_corr).save(out_path)

            except Exception as e:
                print(f"Error cargando {img_path}: {e}")
