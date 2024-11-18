from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort  

CONFDENCE_THRESHOLD = 0.8

model = YOLO('yolov8n_ncnn_model')
tracker = DeepSort(max_age = 50)

cam = cv2.VideoCapture(0)

def getColour(cls_index):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_index % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] *
    (cls_index // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

while True:
    ret, frame = cam.read()
    if not ret:
        continue
    frame_resize = cv2.resize(frame, (640, 480))

    results =model.track(frame_resize, stream = True)

    detections = []

    for result in results:
        classes_names = result.names 
        for box in result.boxes:
            if box.conf[0] > CONFDENCE_THRESHOLD:
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])
                class_name = classes_names[cls]
                colour = getColour(cls)
                # cv2.rectangle(frame_resize, (x1, y1), (x2, y2), colour, 2)
                # cv2.putText(frame_resize, f'{classes_names[int(box.cls[0])]}{box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
                detections.append([[x1, y1, x2 - x1, y2 - y1], box.conf[0], box.cls[0]])

    tracks = tracker.update_tracks(detections, frame = frame_resize)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        cv2.rectangle(frame_resize, (xmin, ymin), (xmax, ymax), getColour(2), 2)
        cv2.rectangle(frame_resize, (xmin, ymin - 20), (xmin + 20, ymin), getColour(2), -1)
        cv2.putText(frame_resize, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, getColour(3), 2) 

    cv2.imshow('frame', frame_resize)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()