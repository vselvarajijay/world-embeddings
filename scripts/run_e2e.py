#!/usr/bin/env python3
"""
E2E test: publish synthetic images to /camera/image_raw, wait for snapshot,
inspect memory (entity list) and embeddings in the JSON output.
Without ONNX models the node runs but produces 0 entities.
"""
import json
import os
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header


def bgr8_image(width: int, height: int, value: int = 128) -> bytes:
    """Synthetic BGR8 image as bytes (row-major)."""
    return bytes([value] * (width * height * 3))


class ImagePublisher(Node):
    def __init__(self, topic: str):
        super().__init__("e2e_image_publisher")
        self.pub = self.create_publisher(Image, topic, 10)

    def publish_image(self, width: int = 640, height: int = 480):
        msg = Image()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_optical_frame"
        msg.height = height
        msg.width = width
        msg.encoding = "bgr8"
        msg.step = width * 3
        msg.is_bigendian = 0
        msg.data = bgr8_image(width, height)
        self.pub.publish(msg)


def wait_for_snapshot(path: str, timeout_sec: float = 15.0, poll_interval: float = 0.5) -> bool:
    """Return True when path exists and has been written (non-empty)."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if os.path.isfile(path):
            try:
                if os.path.getsize(path) >= 2:  # at least "{}"
                    return True
            except OSError:
                pass
        time.sleep(poll_interval)
    return False


def assert_snapshot_structure(data: dict, expect_entities: bool = False) -> None:
    """Assert snapshot has entities array and each entity has required keys and valid embeddings."""
    if "entities" not in data:
        raise AssertionError("Snapshot missing 'entities' key")
    entities = data["entities"]
    if not isinstance(entities, list):
        raise AssertionError("'entities' must be a list")
    if expect_entities and len(entities) == 0:
        raise AssertionError("Expected at least one entity (models enabled?)")
    required = {"id", "embedding", "pose", "last_seen_tick", "confidence", "history"}
    for i, e in enumerate(entities):
        if not isinstance(e, dict):
            raise AssertionError(f"entities[{i}] must be a dict")
        missing = required - set(e.keys())
        if missing:
            raise AssertionError(f"entities[{i}] missing keys: {missing}")
        emb = e["embedding"]
        if not isinstance(emb, list):
            raise AssertionError(f"entities[{i}].embedding must be a list")
        for j, x in enumerate(emb):
            if not isinstance(x, (int, float)):
                raise AssertionError(f"entities[{i}].embedding[{j}] must be numeric")
        pose = e["pose"]
        if not isinstance(pose, dict) or not {"x", "y", "z"}.issubset(set(pose.keys())):
            raise AssertionError(f"entities[{i}].pose must have x, y, z")
        if not isinstance(e["history"], list):
            raise AssertionError(f"entities[{i}].history must be a list")


def main() -> int:
    snapshot_path = os.environ.get("SNAPSHOT_PATH", "/data/snapshot.json")
    image_topic = os.environ.get("IMAGE_TOPIC", "/camera/image_raw")
    expect_entities = os.environ.get("WORLD_EMBEDDINGS_EXPECT_ENTITIES", "").lower() in ("1", "true", "yes")

    rclpy.init()
    try:
        node = ImagePublisher(image_topic)
        # Give subscription time to connect
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.2)
        # Publish a few images
        num_images = 5
        rate_hz = 2
        for k in range(num_images):
            node.publish_image()
            for _ in range(int(10 / rate_hz)):
                rclpy.spin_once(node, timeout_sec=0.1)
        node.get_logger().info("Published %d images, waiting for snapshot at %s" % (num_images, snapshot_path))
        # Wait for snapshot (node writes every 2s in e2e config)
        if not wait_for_snapshot(snapshot_path):
            print("E2E failed: snapshot file did not appear in time", file=sys.stderr)
            return 1
        with open(snapshot_path) as f:
            data = json.load(f)
        assert_snapshot_structure(data, expect_entities=expect_entities)
        print("E2E passed: snapshot structure and embeddings OK (entities=%d)" % len(data["entities"]))
        return 0
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
