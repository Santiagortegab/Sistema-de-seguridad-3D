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

