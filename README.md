# 🛡️ VisionSentinel
### AI-Powered Warehouse Safety Monitoring and Video Intelligence

VisionSentinel is a computer vision-based warehouse safety monitoring project designed to analyze CCTV footage, detect warehouse objects, track their movements, and identify potentially unsafe interactions between people and industrial vehicles.

The project combines deep learning-based object detection, multi-object tracking, spatial analysis, and temporal reasoning to build a foundation for intelligent warehouse safety monitoring.

**Current focus:** YOLO-based object detection, detector experimentation, ByteTrack integration, object trajectory analysis, and person–vehicle proximity monitoring.

---

## 🎯 Project Objectives

Warehouses involve continuous interaction between workers, forklifts, and pallet trucks. Monitoring these interactions manually through CCTV footage can be challenging.

VisionSentinel aims to develop an intelligent video analytics system that can:

- Detect people, forklifts, and pallet trucks in warehouse environments.
- Track multiple objects across consecutive video frames.
- Maintain persistent tracking identities.
- Analyze object trajectories and movement patterns.
- Calculate pixel-based distances between people and industrial vehicles.
- Identify approaching and separating movement patterns.
- Generate configurable proximity alerts and structured event logs.
- Provide a foundation for future Vision Language Model (VLM)-based scene understanding.

## 🧠 Current System Architecture

```text
                 Warehouse CCTV
                       |
                       v
                Video Processing
                       |
                       v
              Object Detection Layer
                       |
          +------------+------------+
          |            |            |
          v            v            v
        YOLO11s   Grounding DINO   RF-DETR
          |            |            |
          +------------+------------+
                       |
                       v
                Box Fusion (NMS)
                       |
                       v
               Zone-Based Filtering
                       |
                       v
                ByteTrack Tracking
                       |
             +---------+---------+
             |                   |
             v                   v
       Trajectory Analysis   Proximity Analysis
             |                   |
             +---------+---------+
                       |
                       v
                Safety Monitoring
                       |
             +---------+---------+
             |                   |
             v                   v
        Annotated Video      Event Logging
```

The architecture is modular, allowing different detection models and tracking components to be evaluated independently.

## 🚀 Work Completed So Far

### 1. Warehouse Object Detection

Implemented a YOLO11s-based object detection pipeline using a custom warehouse dataset.

The current detector recognizes three classes:

| Class ID | Class |
|---|---|
| 0 | Person |
| 1 | Forklift |
| 2 | Pallet Truck |

**YOLO11s validation results**

The model was trained for 150 epochs at an image size of 800 pixels using a Tesla T4 GPU.

| Metric | Result |
|---|---:|
| Precision | 90.9% |
| Recall | 86.3% |
| mAP@50 | 90.6% |
| mAP@50–95 | 62.6% |

**Per-class validation performance**

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| Forklift | 98.4% | 95.4% | 99.0% | 69.8% |
| Pallet Truck | 81.3% | 78.3% | 82.1% | 50.4% |
| Person | 93.1% | 85.2% | 90.7% | 67.5% |

These are validation results from the reported experiment, not guarantees of real-world CCTV performance.

### 2. Multiple Detection Model Experiments

The project includes modular detector implementations for:

- **YOLO11s:** Primary trained warehouse object detector.
- **Grounding DINO:** Text-prompt-based open-vocabulary object detection.
- **RF-DETR:** Transformer-based object detection experimentation.
- **Box Fusion:** Combines detections from different models using class-aware Non-Maximum Suppression (NMS).

This provides a foundation for comparing detector performance and investigating detection gaps.

### 3. Zone-Based Detection Filtering

Implemented a camera-specific zone management and detection filtering system.

Current components include:

- Camera profiles and registration.
- Camera reference images.
- Configurable warehouse zones.
- Zone policy management.
- Detection filtering based on configured zones.
- Suppressed detection candidates and audit records.

This allows detection outputs to be interpreted in the context of warehouse camera layouts and operational areas.

### 4. ByteTrack Multi-Object Tracking

Integrated a custom ByteTrack adapter based on the Ultralytics ByteTrack implementation.

The tracking pipeline is designed to:

- Associate detections across consecutive frames.
- Maintain object tracking IDs.
- Track people, forklifts, and pallet trucks.
- Apply class-aware tracking association.
- Maintain object trajectories.
- Handle temporary object disappearance using tracker buffering.

Tracking is implemented separately from the detection pipeline, allowing detection outputs from different models to be tracked.

### 5. Object Trajectory Analysis

Implemented trajectory management for tracked objects.

The system maintains movement history and uses bounding-box bottom-centre coordinates to represent object positions.

Current capabilities include:

- Per-object trajectory history.
- Configurable history length.
- Trajectory visualization.
- Track cleanup after configurable periods.
- Frame-level tracking information.

### 6. Person–Vehicle Proximity Monitoring

A warehouse safety monitoring module has been developed to analyze interactions between people and industrial vehicles.

The current implementation considers:

- Person–forklift interactions.
- Person–pallet-truck interactions.
- Euclidean centre-to-centre pixel distance.
- Minimum bounding-box edge distance.
- Relative approach speed.
- Approaching, separating, and stable movement states.
- Configurable proximity levels.

Initial experimental thresholds:

| Proximity level | Pixel distance |
|---|---|
| SAFE | Greater than 150 px |
| CAUTION | 80–150 px |
| DANGER | Below 80 px |

These are configurable image-space thresholds, not physical distances or validated industrial safety limits.

### 7. Temporal Safety Event Monitoring

Implemented a configurable safety monitoring layer with:

- Temporal distance smoothing.
- Consecutive observation requirements.
- Approach-state analysis.
- Alert cooldowns.
- Proximity state transitions.
- High-risk approach indicators.

The objective is to avoid generating alerts based on isolated detections or single-frame fluctuations.

### 8. Event Logging and Reporting

The tracking pipeline supports structured event logging.

Event information includes:

- Timestamp and frame number.
- Person tracking ID.
- Vehicle tracking ID and class.
- Centre distance.
- Bounding-box distance.
- Relative approach speed.
- Approach state.
- Proximity level.
- Alert type.

The system generates CSV event logs and JSON tracking summaries.

### 9. Streamlit Application

A multi-page Streamlit application has been developed as the project interface.

Current application sections include:

| Module | Purpose |
|---|---|
| Camera Registry | Manage camera profiles and reference images |
| Video Analysis | Run detection on uploaded warehouse footage |
| Detection Lab | Experiment with detection models |
| Tracking Lab | Apply ByteTrack to saved detection results |
| Safety Audit | Safety monitoring interface |
| Alerts & Reports | Reporting interface |
| VLM Analysis | Placeholder for future VLM development |

### 10. Testing

The repository includes dedicated test modules covering:

- ByteTrack adapter.
- Tracking safety logic.
- Tracking pipeline data.
- Detection pipeline models.
- Detection fusion.
- RF-DETR detector.
- Zone policies.
- Video utilities.
- Detection data export.

Tests are included in the repository; complete test-suite execution and integration verification remain part of the ongoing validation process.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| Object Detection | YOLO11, Grounding DINO, RF-DETR |
| Multi-Object Tracking | ByteTrack |
| Computer Vision | OpenCV |
| Detection Processing | Supervision |
| Numerical Computing | NumPy, SciPy |
| Data Processing | Pandas |
| Configuration | YAML, JSON |
| Application | Streamlit |
| Visualization | Matplotlib, OpenCV |
| Video Processing | OpenCV, FFmpeg |
| Testing | Pytest |

## 📁 Project Structure

```text
VisionSentinel/
│
├── app.py
│
├── configs/
│   ├── bytetrack.yaml
│   ├── tracking_safety.yaml
│   └── cameras/
│
├── core/
│   ├── camera/
│   │   └── profile_manager.py
│   │
│   ├── tracking/
│   │   ├── byte_tracker.py
│   │   ├── trajectory.py
│   │   ├── proximity.py
│   │   ├── safety_monitor.py
│   │   ├── safety_events.py
│   │   └── event_logger.py
│   │
│   └── zones/
│       ├── detection_filter.py
│       ├── zone_engine.py
│       └── zone_policy.py
│
├── src/
│   ├── config.py
│   ├── yolo_detector.py
│   ├── dino_detector.py
│   ├── rfdetr_detector.py
│   ├── fusion.py
│   ├── detection_pipeline.py
│   ├── tracking_pipeline.py
│   └── video_utils.py
│
├── pages/
│   ├── camera_registry.py
│   ├── video_analysis.py
│   ├── detection_lab.py
│   ├── tracking_lab.py
│   ├── safety_audit.py
│   ├── alerts_reports.py
│   └── vlm_analysis.py
│
├── notebooks/
│
├── scripts/
│
├── tests/
│
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd VisionSentinel
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Note: PyTorch and its hardware-specific dependencies may require a separate installation depending on your operating system and GPU.

### 4. Add the trained model

Place the trained YOLO11s checkpoint in the expected models directory:

```text
models/
└── yolo11s_warehouse_best.pt
```

Model weights and datasets are excluded from the Git repository and must be supplied separately.

### 5. Run the application

```bash
streamlit run app.py
```

## 🔬 Future Development Roadmap

VisionSentinel is being developed incrementally. The following are planned extensions and research directions, not claims of completed functionality.

### Phase 1: Complete and Validate Tracking

- Debug and verify ByteTrack integration.
- Evaluate persistent ID consistency.
- Investigate ID switches and temporary tracking loss.
- Validate trajectory calculations.
- Test person–vehicle proximity monitoring.
- Evaluate approach and separation detection.
- Measure end-to-end FPS and processing latency.
- Improve tracking reliability on different warehouse videos.

### Phase 2: Vision Language Model Integration

Introduce Vision Language Models for language-driven warehouse scene understanding.

Planned experimentation:

- Qwen Vision Language Models.
- Image and video frame understanding.
- Natural-language scene descriptions.
- Prompt-based safety observations.
- Multi-frame reasoning.
- Structured JSON event extraction.
- Scene summarization and anomaly investigation.

The objective is to complement conventional object detection with higher-level visual reasoning.

### Phase 3: Intelligent Safety Event Analysis

Explore combining detection, tracking and VLM outputs to investigate:

- Person–vehicle interactions.
- Potential near-miss situations.
- Unsafe movement patterns.
- Restricted-zone activity.
- Warehouse obstruction events.
- Contextual safety observations.

### Phase 4: Notification and Agentic Workflows

Explore event-driven workflows for:

- Safety alert generation.
- Telegram or WhatsApp notifications.
- Event snapshots and video clips.
- Incident summaries.
- Automated safety reports.

### Phase 5: Deployment and Optimization

- Docker-based deployment.
- Edge AI experimentation.
- NVIDIA Jetson deployment exploration.
- Model optimization and quantization.
- Processing latency improvements.
- Scalable multi-camera architecture.

---

## ⚠️ Current Limitations

- Pixel-based proximity measurements are camera-dependent and cannot directly represent physical distances.
- Tracking IDs can switch during occlusion or difficult scenes.
- Detection performance depends on camera angle, lighting, object scale and environmental conditions.
- Validation metrics do not guarantee equivalent performance on unseen CCTV footage.
- VLM-based reasoning and contextual safety analysis are planned future developments.
- The system is a research and development prototype, not a certified industrial safety system.

## 🎯 Project Vision

The long-term goal is to develop an intelligent warehouse safety platform that combines:

**Computer Vision + Multi-Object Tracking + Vision Language Models + Temporal Reasoning + Agentic Workflows**

VisionSentinel aims to progress from conventional object detection towards contextual warehouse video understanding, enabling more informative safety monitoring and incident investigation.

---

**Project Status:** Active Development

**Current Development Focus:** ByteTrack integration, tracking validation and person–vehicle proximity analysis.

**Next Major Milestone:** Vision Language Model integration for warehouse scene understanding.

