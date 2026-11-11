# main.py
import pickle
import cv2
import supervision as sv
from config import (
    MODELS_DIR, VIDEOS_DIR, OUTPUTS_DIR,
    CLASS_MAPPING, DINO_PROMPT, YOLO_CONF_THRESH,
    DINO_BOX_THRESH, DINO_TEXT_THRESH, NMS_IOU_THRESH
)
from yolo_detector import YoloDetector
from dino_detector import DinoDetector
from fusion import BoxFusion

def main():
    video_path = str(VIDEOS_DIR / "warehouse2.mp4")
    yolo_weight_path = str(MODELS_DIR / "yolo11s_warehouse_best.pt") # Or your chosen model
    out_video_path = str(OUTPUTS_DIR / "warehouse2_fused.mp4")

    # 1. Initialize models
    print("📦 Loading YOLO...")
    yolo = YoloDetector(yolo_weight_path, conf=YOLO_CONF_THRESH)
    
    print("🦖 Loading Grounding DINO...")
    dino = DinoDetector(box_thresh=DINO_BOX_THRESH, text_thresh=DINO_TEXT_THRESH)
    
    fusion = BoxFusion(iou_threshold=NMS_IOU_THRESH)

    # 2. Setup video reader & writer
    video_info = sv.VideoInfo.from_video_path(video_path)
    video_info.fps = int(round(video_info.fps)) or 25

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    print(f"🎥 Processing {video_path}...")
    with sv.VideoSink(target_path=out_video_path, video_info=video_info, codec="mp4v") as sink:
        for idx, frame in enumerate(sv.get_video_frames_generator(source_path=video_path)):
            # Detect with YOLO (runs fast on every frame)
            yolo_det = yolo.detect(frame)

            # Detect with DINO (every 2nd frame to optimize speed)
            if idx % 1 == 0:
                dino_det = dino.detect(frame, prompt=DINO_PROMPT)
                last_dino_det = dino_det
            else:
                dino_det = last_dino_det

            # Fuse both model outputs
            fused_det = fusion.merge_and_suppress(yolo_det, dino_det)

            # Annotate
            labels = [
                f"{CLASS_MAPPING[int(cid)]} {conf:.2f}"
                for cid, conf in zip(fused_det.class_id, fused_det.confidence)
            ]
            annotated = box_annotator.annotate(scene=frame.copy(), detections=fused_det)
            annotated = label_annotator.annotate(scene=annotated, detections=fused_det, labels=labels)
            
            sink.write_frame(annotated)

    print(f"✅ Finished! Video saved to: {out_video_path}")

if __name__ == "__main__":
    main()