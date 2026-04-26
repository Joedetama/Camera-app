from ultralytics import YOLO
import time

model = YOLO("yolov8n.pt")

start = time.time()
results = model("https://ultralytics.com/images/bus.jpg", device=0)
end = time.time()

print(results[0].boxes)
print("Tempo:", end - start)
