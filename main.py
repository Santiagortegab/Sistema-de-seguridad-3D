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


cap = cv2.VideoCapture("example2.mp4")

if not cap.isOpened():
    print("Error al abrir el archivo de video.")
    exit()

contador_frames = 0
salto_frames = 8
resolucion_vda = (350)


while True:
    ret, frame = cap.read()

    if not ret:
        print("Fin del video.")
        break

    contador_frames += 1

    if contador_frames % salto_frames != 0:
        continue

    frame = cv2.resize(frame, (854,480))

    resultados = detector_yolo(frame, classes=[0], verbose=False)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        profundidad = depth_model.infer_video_depth_one(
            frame_rgb,
            input_size=resolucion_vda,
            device=DEVICE,
            fp32=True
        )

    mapa_norm = cv2.normalize(profundidad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mapa_color = cv2.applyColorMap(mapa_norm, cv2.COLORMAP_INFERNO)

    for resultado in resultados:
        cajas = resultado.boxes
        for caja in cajas:

            x1, y1, x2, y2 = caja.xyxy[0].int().tolist()
            region_profundidad_humano = mapa_norm[y1:y2, x1:x2]
            if region_profundidad_humano.size > 0:
                profundidad_estimada = np.median(region_profundidad_humano)
            else:
                profundidad_estimada = 0

            confianza = caja.conf[0].item()
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            label_conf = f"Operador: {confianza:.2f}"
            label_prof = f"Profundidad: {profundidad_estimada:.2f}"
            cv2.putText(frame, label_prof, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            

    cv2.imshow("Sistema de seguridad 3D - Deteccion RGB", frame)
    cv2.imshow("Sistema de seguridad 3D - Mapa de profundidad", mapa_color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

