from ultralytics import YOLO
import cv2
import easyocr
from ultralytics.utils.plotting import Annotator, colors
from util import get_car, read_license_plate, write_csv

reader = easyocr.Reader(['en'])
tensorrt_model = YOLO("best.engine")

video_path = "video.mp4"  # For videos
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

results_dict = {}

frame_nmr = 0  # Frame counter
vehicle_track_ids = []  # Store car bounding boxes and IDs if needed for tracking

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = tensorrt_model(frame)
    frame_results = {}  # Store frame-specific results

    for result in results:
        boxes = result.boxes
        annotator = Annotator(frame, line_width=2, example=str(tensorrt_model.names))

        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = box.conf[0]
            cls = box.cls[0]

            if conf > 0.5:
                license_plate_crop = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                license_text, text_score = read_license_plate(license_plate_crop)

                if license_text:
                    label = f"{license_text} ({text_score:.2f})"
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

                    car_id, car_x1, car_y1, car_x2, car_y2 = get_car((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls), vehicle_track_ids)
                    if car_id != -1:
                        frame_results[car_id] = {
                            "car": {"bbox": [car_x1, car_y1, car_x2, car_y2]},
                            "license_plate": {
                                "bbox": xyxy,
                                "bbox_score": conf,
                                "text": license_text,
                                "text_score": text_score
                            }
                        }

                else:
                    label = f"{tensorrt_model.names[int(cls)]} ({conf:.2f})"
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

    results_dict[frame_nmr] = frame_results
    frame_nmr += 1

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
output_csv_path = "license_plate_results.csv"
write_csv(results_dict, output_csv_path)

cap.release()
cv2.destroyAllWindows()
