import argparse
import os
import sys
from contextlib import asynccontextmanager
from typing import List

import cv2
import numpy as np
import uvicorn
import yaml
from easydict import EasyDict
from fastapi import FastAPI, File, UploadFile

# =====================================================================
# DOPPIO BINARIO (DEV vs PROD)
# =====================================================================
DEV_PATH = "/workspaces/darknet"  # Cartella montata da tkTrain / DevContainer
PROD_PATH = "/opt/darknet"  # Cartella sicura compilata nell'immagine Docker

if os.path.exists(os.path.join(DEV_PATH, "libdarknet.so")):
    print("Modalità SVILUPPO: trovata compilazione locale. Uso il codice host!")
    sys.path.insert(0, DEV_PATH)
else:
    print("Modalità PRODUZIONE: nessuna compilazione locale. Uso Darknet integrato!")
    sys.path.insert(0, PROD_PATH)

# Ora che il path è impostato correttamente, possiamo importare il wrapper di AlexeyAB
try:
    import darknet
except ImportError:
    sys.exit(
        "ERRORE: Impossibile trovare darknet.py o libdarknet.so! Assicurati di aver compilato correttamente."
    )

parser = argparse.ArgumentParser(description="YOLO Darknet Inference Server")
parser.add_argument("--cfg", type=str, required=True, help="Path al file .cfg")
args, unknown = parser.parse_known_args()

NETWORK = None
CLASS_NAMES = None
CLASS_COLORS = None


def load_config(path):
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return EasyDict(data)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config(args.cfg)
    global NETWORK, CLASS_NAMES, CLASS_COLORS

    print(f"🚀 Caricamento rete Darknet in C/CUDA...")
    print(f"   CFG: {cfg.config_file}")
    print(f"   WEIGHTS: {cfg.weights_file}")

    # Carica la rete in memoria RAM/VRAM usando la funzione C nativa
    NETWORK, CLASS_NAMES, CLASS_COLORS = darknet.load_network_inference(
        cfg.config_file, cfg.weights_file, cfg.label_file, batch_size=1
    )
    print("✅ Modello Darknet caricato con successo!")

    yield

    print("🛑 Spegnimento server Darknet...")


app = FastAPI(title="Darknet YOLO Inference Server", lifespan=lifespan)


@app.post("/predict_batch")
async def predict_batch(thr: float = 0.5, files: List[UploadFile] = File(...)):
    global NETWORK, CLASS_NAMES

    all_annotations = []

    for file in files:
        raw_bytes = await file.read()
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        frame_annotations = []

        if img_cv2 is not None:
            img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img_rgb.shape[:2]

            # Allocazione memoria C
            darknet_image = darknet.make_image(orig_w, orig_h, 3)
            darknet.copy_image_from_bytes(darknet_image, img_rgb.tobytes())

            # Inferenza C-Core
            detections = darknet.detect_image(
                NETWORK, CLASS_NAMES, darknet_image, thresh=thr
            )

            # De-allocazione per prevenire Memory Leak
            darknet.free_image(darknet_image)

            for label, confidence, bbox in detections:
                conf_val = float(confidence)
                # Darknet restituisce la confidenza in percentuale (es. 98.5)
                score = conf_val / 100.0 if conf_val > 1.0 else conf_val

                frame_annotations.append(
                    {
                        "class_name": label,
                        "score": score,
                        "box": bbox,  # Darknet bbox formato: (x_center, y_center, w, h)
                    }
                )

        all_annotations.append(frame_annotations)

    return {"annotations": all_annotations}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
