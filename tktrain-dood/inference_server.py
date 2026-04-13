import sys
import os
import cv2
import numpy as np
import argparse
from fastapi import FastAPI, File, UploadFile
import uvicorn
import yaml
from easydict import EasyDict

# =====================================================================
# DOPPIO BINARIO (DEV vs PROD)
# =====================================================================
DEV_PATH = "/workspaces/darknet"  # Cartella montata da tkTrain / DevContainer
PROD_PATH = "/opt/darknet"        # Cartella sicura compilata nell'immagine Docker

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
    sys.exit("ERRORE: Impossibile trovare darknet.py o libdarknet.so! Assicurati di aver compilato correttamente.")

parser = argparse.ArgumentParser(description="YOLO Darknet Inference Server")
parser.add_argument("--cfg", type=str, required=True, help="Path al file .cfg")
args, unknown = parser.parse_known_args()

app = FastAPI(title="Darknet YOLO Inference Server")
NETWORK = None
CLASS_NAMES = None
CLASS_COLORS = None

def load_config(path):
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return EasyDict(data)

@app.on_event("startup")
async def startup_event():
    cfg = load_config(args.cfg)
    global NETWORK, CLASS_NAMES, CLASS_COLORS
    print(f"Caricamento rete Darknet in C/CUDA...")
    print(f"   CFG: {cfg.config_file}")
    print(f"   WEIGHTS: {cfg.weights_file}")
    print(f"   DATA: {cfg.data_file}")
    print(f"   LABELS: {cfg.label_file}")

    # Carica la rete in memoria RAM/VRAM usando la funzione C nativa
    NETWORK, CLASS_NAMES, CLASS_COLORS = darknet.load_network(
        cfg.config_file,
        cfg.data_file,
        cfg.weights_file,
        batch_size=1
    )
    print("Modello Darknet caricato con successo!")

# =====================================================================
# 4. ENDPOINT DI INFERENZA
# =====================================================================
@app.post("/predict_stream")
async def predict_stream(file: UploadFile = File(...), thr: float = 0.5):
    global NETWORK, CLASS_NAMES

    raw_bytes = await file.read()
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_cv2 is None:
        return {"status": "error", "message": "Immagine non valida o corrotta"}

    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]
    darknet_image = darknet.make_image(orig_w, orig_h, 3)
    darknet.copy_image_from_bytes(darknet_image, img_rgb.tobytes())
    detections = darknet.detect_image(NETWORK, CLASS_NAMES, darknet_image, thresh=thr)
    darknet.free_image(darknet_image)
    annotations = []
    for label, confidence, bbox in detections:
        conf_val = float(confidence)
        score = conf_val / 100.0 if conf_val > 1.0 else conf_val

        annotations.append({
            "class_name": label,
            "score": score,
            "box": bbox # bbox è già una tupla (x, y, w, h)
        })

    return {"status": "success", "annotations": annotations}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)