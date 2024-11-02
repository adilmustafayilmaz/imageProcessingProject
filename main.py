import os
import cv2
import numpy as np
import pytesseract
import easyocr
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

reader = easyocr.Reader(['en'])
tensorrt_model = YOLO("best.engine")

video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
results_dict = {}
frame_nmr = 0
vehicle_track_ids = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.rectangle(frame, (0, height-100), (width, height-400), (256, 256, 256), 2)
    results = tensorrt_model(frame)
    frame_results = {}

    for result in results:
        boxes = result.boxes
        annotator = Annotator(frame, line_width=2, example=str(tensorrt_model.names))

        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = box.conf[0]
            cls = box.cls[0]

            if conf > 0.5:
                # !!! NEED HELP HERE !!! (Program is not able to detect the license plate text perfectly)
                label = f"Plate ({conf:.2f})"
                annotator.box_label(xyxy, label, color=colors(int(cls), True))
                license_plate_crop = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                license_plate_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
                license_plate_thresh = cv2.resize(license_plate_thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                
                output = reader.readtext(license_plate_thresh)
                for out in output:
                    text_bbox, license_text, text_score = out
                    if text_score > 0.5:

                        frame_results[frame_nmr] = {
                            "car": {"bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]},
                            "license_plate": {
                                "bbox": xyxy,
                                "bbox_score": conf,
                                "text": license_text,
                                "text_score": text_score
                            }
                        }
    
    results_dict[frame_nmr] = frame_results
    frame_nmr += 1

    cv2.imshow('Video', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

output_csv_path = "license_plate_results.csv"
import csv

with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Car BBox", "License Plate BBox", "BBox Score", "Text", "Text Score"])
    
    for frame_id, frame_data in results_dict.items():
        for car_id, data in frame_data.items():
            writer.writerow([
                frame_id,
                data["car"]["bbox"],
                data["license_plate"]["bbox"],
                data["license_plate"]["bbox_score"],
                data["license_plate"]["text"],
                data["license_plate"]["text_score"]
            ])
