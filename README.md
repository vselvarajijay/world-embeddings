# world-embeddings

Persistent entity memory for embodied agents — a deterministic, embedding-centric world memory engine for autonomous robotics.

## Dependencies

- **ROS 2** (Humble or Iron)
- **OpenCV**
- **Eigen3**
- **ONNX Runtime** (install via system package or [releases](https://github.com/microsoft/onnxruntime/releases))
- **nlohmann/json** (fetched by CMake)
- **cv_bridge** (ROS 2)

## Models

The node expects ONNX models on disk; the repo does not ship them.

- **YOLO (detection):** Export YOLOv8 (e.g. `yolov8n.pt`) to ONNX (Ultralytics: `model.export(format="onnx")`). Use the detection model (output shape `[1, 84, 8400]` for 80 classes).
- **CLIP (image encoder):** Export the **image encoder** subgraph only to ONNX (e.g. [CLIP-to-onnx-converter](https://github.com/jalberse/CLIP-to-onnx-converter)). Input: `[1, 3, 224, 224]` (NCHW), normalized with ImageNet stats; output: 512-dim embedding for ViT-B/32.

Place the ONNX files in a `models/` directory or set paths via parameters.

## Build

From a ROS 2 workspace with this package in `src/`:

```bash
source /opt/ros/humble/setup.bash  # or iron
colcon build --packages-select world_embeddings
source install/setup.bash
```

## Tests

```bash
colcon test --packages-select world_embeddings
colcon test-result --all
```

Encoder and detector tests that require ONNX models are skipped unless env vars are set:

- `WORLD_EMBEDDINGS_CLIP_ONNX_PATH` — path to CLIP image encoder ONNX
- `WORLD_EMBEDDINGS_YOLO_ONNX_PATH` — path to YOLOv8 detection ONNX

## Run

```bash
ros2 launch world_embeddings world_embeddings.launch.py
```

Set parameters (e.g. in `config/params.yaml` or via launch) for:

- `image_topic` — camera image topic
- `snapshot_path` — where to write JSON snapshots
- `snapshot_interval_sec` — snapshot period
- `yolo_onnx_path`, `clip_onnx_path` — paths to ONNX models
- Association and EMA parameters (see `config/params.yaml`)

If `yolo_onnx_path` or `clip_onnx_path` is empty, the node runs but does not process frames (no detection/encoding).

## Output

- **JSON snapshot:** Entity list with `id`, `embedding`, `pose`, `last_seen_tick`, `confidence`, `semantic_label`, and `history`. Written periodically and on shutdown to `snapshot_path`.
- **Pose (MVP):** Derived from 2D detection box center (`x`, `y` in image coords, `z=0`). Replace with real 3D when a camera model is available.

## Project structure

```
world-embeddings/
├── CMakeLists.txt
├── package.xml
├── include/world_embeddings/   # Public API
│   ├── types.hpp
│   ├── association.hpp
│   ├── entity_store.hpp
│   ├── snapshot.hpp
│   ├── encoder.hpp
│   └── detector.hpp
├── src/                        # Library + node
├── test/                       # Unit and integration tests
├── config/params.yaml
└── launch/world_embeddings.launch.py
```

## Visualization

Use the exported JSON (e.g. load embeddings in Python and run t-SNE/UMAP, or plot poses) for visualization.
