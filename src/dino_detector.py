# src/dino_detector.py
import torch
from PIL import Image
import cv2
import numpy as np
import supervision as sv
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from pathlib import Path
import pickle

class DinoDetector:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", box_thresh: float = 0.35, text_thresh: float = 0.25):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.box_thresh = box_thresh
        self.text_thresh = text_thresh

    def _map_label_to_id(self, label: str) -> int:
        lbl = label.lower()
        if "person" in lbl:
            return 0
        elif "pallet truck" in lbl or "pallet jack" in lbl or "manual pallet jack" in lbl:
            return 2
        elif "forklift" in lbl:
            return 1
        return -1

    def detect(self, frame: np.ndarray, prompt: str) -> sv.Detections:
        """Runs Grounding DINO on a single BGR OpenCV frame."""
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil_img, text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=self.box_thresh,
                text_threshold=self.text_thresh,
                target_sizes=[pil_img.size[::-1]]
            )[0]
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=self.box_thresh,
                text_threshold=self.text_thresh,
                target_sizes=[pil_img.size[::-1]]
            )[0]

        xyxy = results["boxes"].cpu().numpy()
        conf = results["scores"].cpu().numpy()
        raw_labels = results["text_labels"] if "text_labels" in results else results["labels"]

        class_ids = np.array([self._map_label_to_id(l) for l in raw_labels] , dtype = np.int64)
        valid_mask = [cid != -1 for cid in class_ids]
 
        if not any(valid_mask):
            return sv.Detections.empty()

        valid_idx = np.where(valid_mask)[0]
        return sv.Detections(
            xyxy=np.ascontiguousarray(xyxy[valid_idx] , dtype = np.float32),
            confidence=np.ascontiguousarray(conf[valid_idx] , dtype = np.float32),
            class_id=np.ascontiguousarray(class_ids[valid_idx] , dtype = np.int64)
        )

    def precompute_video_cache(self, video_paths: list, prompt: str, save_path: str, stride: int = 1) -> dict:
        """Processes videos, stores frame-by-frame detections, and saves them to a pickle file."""
        dino_cache = {}
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)

        for v_path in video_paths:
            v_name = Path(v_path).stem
            dino_cache[v_name] = {}
            print(f"🔍 Precomputing Grounding DINO for {v_name}...")
            
            last_det = sv.Detections.empty()
            for idx, frame in enumerate(sv.get_video_frames_generator(source_path=v_path)):
                if idx % stride == 0:
                    last_det = self.detect(frame, prompt=prompt)
                dino_cache[v_name][idx] = last_det

            print(f"   Done {v_name} ({idx + 1} frames)")

        with open(save_file, "wb") as f:
            pickle.dump(dino_cache, f)

        print(f"✅ Successfully saved pickle cache at: {save_file}")
        return dino_cache

    @staticmethod
    def load_cache(cache_path: str) -> dict:
        """Loads precomputed Grounding DINO detections from disk."""
        c_path = Path(cache_path)
        if not c_path.exists():
            raise FileNotFoundError(f"Cache file not found at: {c_path}")
        with open(c_path, "rb") as f:
            return pickle.load(f)