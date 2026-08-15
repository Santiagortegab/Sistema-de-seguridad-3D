import cv2
import torch
from ultralytics import YOLO

from vda_core.video_depth_anything.video_depth import VideoDepthAnything

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


cap = cv2.VideoCapture("example.mp4")

if not cap.isOpened():
    print("Error al abrir el archivo de video.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Fin del video.")
        break

    frame = cv2.resize(frame, (854,480))

    resultados = detector_yolo(frame, classes=[0], verbose=False)

    for resultado in resultados:
        cajas = resultado.boxes
        for caja in cajas:

            x1, y1, x2, y2 = caja.xyxy[0].int().tolist()
            confianza = caja.conf[0].item()
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            label = f"Operador: {confianza:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Sistema de seguridad 3D - Detección", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

