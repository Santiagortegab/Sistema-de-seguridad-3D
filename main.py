from os import read
import threading
import time
import cv2
import torch
import numpy as np

from ultralytics import YOLO

from vda_core.video_depth_anything.video_depth_stream import VideoDepthAnything

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Cargando modelo YOLOv8...")
detector_yolo = YOLO("weights/yolov8n.pt")

print("Cargando modelo Depth Anything...")
model_configs = {
    'vits' : {'encoder': 'vits', 'features': 64, 'out_channels':[48,96,192,384]}
}

depth_model = VideoDepthAnything(**model_configs['vits'])
ruta_pesos_vda = "weights/video_depth_anything_vits.pth"
estado_modelo = torch.load(ruta_pesos_vda, map_location=DEVICE)
depth_model.load_state_dict(estado_modelo)

depth_model = depth_model.to(DEVICE).eval()

print("Modelo YOLOv8 y Depth Anything cargados correctamente.")

frame_actual = None
ultimo_mapa = None
lock= threading.Lock()

def thread_profundidad():
    global frame_actual, ultimo_mapa
    while True:
        frame_process = None
        with lock:
            if frame_actual is not None:
                frame_process = frame_actual.copy()
        if frame_process is None:
            time.sleep(0.01)
            continue
        frame_rgb = cv2.cvtColor(frame_process, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            mapa = depth_model.infer_video_depth_one(
                frame_rgb,
                input_size=250,
                device=DEVICE,
                fp32=True)
        with lock:
            ultimo_mapa = mapa.copy()

bg_thread = threading.Thread(target=thread_profundidad, daemon=True)
bg_thread.start()
print("Procesamiento de profundidad listo")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir el archivo de video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fin del video")
        break

    frame_process = cv2.resize(frame, (854, 480))

    with lock:
        frame_actual = frame_process.copy()

    resultados_yolo = detector_yolo(frame_process, classes=[0], verbose=False)
    mapa_actual = None
    with lock:
        if ultimo_mapa is not None:
            mapa_actual = ultimo_mapa.copy()

    for resultado in resultados_yolo:
        for caja in resultado.boxes:
            x1, y1, x2, y2 = caja.xyxy[0].int().tolist()

            distancia_label = "Calculando profundidad"
            if mapa_actual is not None:
                region = mapa_actual[y1:y2, x1:x2]
                if region.size > 0:
                    distancia_profundidad = np.max(region)
                    distancia_label = f"Profundidad:{distancia_profundidad}"
            cv2.rectangle(frame_process, (x1,y1), (x2,y2), (0, 255, 0), 2)
            cv2.putText(frame_process, distancia_label, (x1, y1+100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.imshow("Sistema Seguridad 3D", frame_process)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()