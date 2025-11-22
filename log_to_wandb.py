import os
import random
import wandb
from PIL import Image

# Configuración
CORRUPTED_ROOT = "/workspace/datasets/hamlyn/hamlyn_corruptions"
EXAMPLES_PER_COMBO = 3  # Número de imágenes por corrupción/nivel
SEED = 42

# Iniciar sesión
wandb.init(
    project="hamlun-corruption-viewer",
    name="hamlun-visual-inspection",
    config={"examples_per_combination": EXAMPLES_PER_COMBO}
)

random.seed(SEED)

# Recorremos todas las corrupciones
for corruption in sorted(os.listdir(CORRUPTED_ROOT)):
    corr_path = os.path.join(CORRUPTED_ROOT, corruption)
    if not os.path.isdir(corr_path):
        continue

    for severity in sorted(os.listdir(corr_path)):
        sev_path = os.path.join(corr_path, severity)
        if not os.path.isdir(sev_path):
            continue

        collected = []

        # Recorrer aleatoriamente los subdirectorios para tomar ejemplos
        for root, _, files in os.walk(sev_path):
            image_files = [f for f in files if f.endswith(".jpg")]
            random.shuffle(image_files)

            for img_file in image_files:
                if len(collected) >= EXAMPLES_PER_COMBO:
                    break
                try:
                    img_path = os.path.join(root, img_file)
                    img = Image.open(img_path).convert("RGB")
                    caption = f"{corruption}/{severity} - {os.path.relpath(img_path, CORRUPTED_ROOT)}"
                    collected.append(wandb.Image(img, caption=caption))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
            if len(collected) >= EXAMPLES_PER_COMBO:
                break

        if collected:
            wandb.log({f"{corruption}/{severity}": collected})

wandb.finish()
