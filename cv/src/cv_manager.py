"""Manages the CV model."""

from typing import Any
import cv2
import numpy as np
from ultralytics import YOLO

class CVManager:

    def __init__(self):
        # This is where you can initialize your model and any static configurations.
        # TODO
        self.model = YOLO("best.pt")  # trained model
        self.conf = 0.05


    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A list of `dict`s containing your CV model's predictions. See
            cv/README.md for the expected format.
        """

        # Your inference code goes here.
        # TODO
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return []

        results = self.model.predict(
            img,
            conf=self.conf,
            imgsz=960,
            iou=0.6,
            verbose=False,
            augment=True,
            max_det=300
        )

        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())

                detections.append({
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1),
                    ],
                    "category_id": cls_id,
                })

        return detections