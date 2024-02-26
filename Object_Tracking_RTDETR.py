import torch
import numpy as np
import cv2
from ultralytics import RTDETR
from time import time
import supervision as sv

class Detection_Transformer:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device:", self.device)
        self.model = RTDETR("rtdetr-l.pt")

        # Create a dictionary mapping COCO class IDs to class names
        coco_class_ids = list(range(91))  # COCO has 91 classes
        # coco_class_names = self.model.model.names
        
        coco_class_names = [    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush']
        
        self.CLASS_NAMES_DICT = dict(zip(coco_class_ids, coco_class_names))

        # Reduce the thickness of bounding boxes
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=2, text_thickness=1, text_scale=0.5,text_padding =1)

    def plot_bboxes(self, results, frame):
        
        # Extract detections
        
        boxes = results[0].boxes.cpu().numpy()
        class_id = boxes.cls
        conf = boxes.conf
        xyxy = boxes.xyxy
        
    
        class_id = class_id.astype(np.int32)
    
        
        # Setup detections for visualization
        detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=conf,
                    class_id=class_id,
                    )
    
        # # Format custom labels
        # self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" for xyxy, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id)]

        
        # # Annotate and display frame
        # frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        
        # Filter detections based on specific class IDs
        filtered_xyxy = []
        filtered_confidence = []
        filtered_class_id = []

        for xyxy, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
            if class_id in [1,2,3,5,7,11]:
                filtered_xyxy.append(xyxy)
                filtered_confidence.append(confidence)
                filtered_class_id.append(class_id)

        # # Format custom labels for filtered detections
        # self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" for xyxy, confidence, class_id in zip(filtered_xyxy, filtered_confidence, filtered_class_id)]
        # Ensure filtered_xyxy is not empty before creating the Detections object
        
        if filtered_xyxy:
            # Format custom labels for filtered detections
            self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" for xyxy, confidence, class_id in zip(filtered_xyxy, filtered_confidence, filtered_class_id)]

            # Annotate and display frame with filtered detections
            frame = self.box_annotator.annotate(scene=frame, detections=sv.Detections(xyxy=np.array(filtered_xyxy), confidence=np.array(filtered_confidence), class_id=np.array(filtered_class_id)), labels=self.labels)
        
        # # Annotate and display frame with filtered detections
        # frame = self.box_annotator.annotate(scene=frame, detections=sv.Detections(xyxy=filtered_xyxy, confidence=filtered_confidence, class_id=filtered_class_id), labels=self.labels)
        
        return frame

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened()
        
       # Get the original frame dimensions
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        
         # Create VideoWriter object to save video
        output_filename = "output_video.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30
        frame_size = (1280, 720)
        out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

        # Define the desired output size (adjust as needed)
        output_width = 1280
        output_height = 720

        while True:
            ret, frame = cap.read()

            if not ret:
                # Break the loop if the video is finished
                break

            # Resize the frame to the desired output size
            frame = cv2.resize(frame, (output_width, output_height))

            results = self.model.predict(frame)
            frame = self.plot_bboxes(results, frame)
            
            out.write(frame)
            cv2.imshow('RTDETR Detection', frame)
            

            if cv2.waitKey(5) & 0xFF == 27:
                # Break the loop if the 'Esc' key is pressed
                break

        cap.release()
        cv2.destroyAllWindows()

    def __call__(self, video_path):
        self.process_video(video_path)

# Example usage:
video_path = r'istockphoto-531954524-640_adpp_is.mp4'
detector = Detection_Transformer(capture_index=0)
detector(video_path)
