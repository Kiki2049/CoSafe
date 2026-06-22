"""
LMDrive Remote Inference Client
Python 3.7 compatible HTTP client for the LMDrive inference server.
"""

import base64
import io
import json
import time

import numpy as np
import requests
from PIL import Image


class LMDriveClient(object):
    """Client for the LMDrive remote inference server."""

    def __init__(self, server_url="http://localhost:8100", session_id="default", timeout=30, max_retries=5):
        self.server_url = server_url.rstrip("/")
        self.session_id = session_id
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.trust_env = False


    @staticmethod
    def encode_image(rgb_array):
        """Encode a numpy RGB array (H, W, 3) uint8 to base64 JPEG string."""
        pil_img = Image.fromarray(rgb_array)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def encode_lidar(lidar_array):
        """Encode a numpy float32 array (40000, 4) to base64 string."""
        raw = lidar_array.astype(np.float32).tobytes()
        return base64.b64encode(raw).decode("ascii")


    def _post(self, endpoint, payload):
        """POST with exponential backoff retry on connection failure."""
        url = self.server_url + endpoint
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                wait = min(2 ** attempt, 30)
                print("[LMDriveClient] %s attempt %d/%d failed: %s. Retrying in %ds..."
                      % (endpoint, attempt + 1, self.max_retries, str(e)[:120], wait))
                time.sleep(wait)
            except requests.HTTPError as e:
                print("[LMDriveClient] HTTP error: %s" % str(e))
                return None
        print("[LMDriveClient] All %d retries exhausted for %s: %s"
              % (self.max_retries, endpoint, str(last_err)[:200]))
        return None

    def _get(self, endpoint):
        """GET with exponential backoff retry."""
        url = self.server_url + endpoint
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                wait = min(2 ** attempt, 30)
                print("[LMDriveClient] GET %s attempt %d/%d failed: %s. Retrying in %ds..."
                      % (endpoint, attempt + 1, self.max_retries, str(e)[:120], wait))
                time.sleep(wait)
            except requests.HTTPError as e:
                print("[LMDriveClient] HTTP error: %s" % str(e))
                return None
        print("[LMDriveClient] All %d retries exhausted for GET %s: %s"
              % (self.max_retries, endpoint, str(last_err)[:200]))
        return None


    def infer(self, ego_id, rgb_front, rgb_left, rgb_right, rgb_rear,
              lidar, num_points, target_point, velocity,
              text_input="Follow the lane", reset_buffer=False,
              use_notice=False, notice_text="", notice_frame_id=-1):
        """
        Run remote inference.

        Args:
            ego_id (int): Ego vehicle index.
            rgb_front (np.ndarray): Front camera image (H, W, 3) uint8.
            rgb_left (np.ndarray): Left camera image (H, W, 3) uint8.
            rgb_right (np.ndarray): Right camera image (H, W, 3) uint8.
            rgb_rear (np.ndarray): Rear camera image (H, W, 3) uint8.
            lidar (np.ndarray): Processed lidar (40000, 4) float32.
            num_points (int): Number of valid lidar points.
            target_point (np.ndarray or list): Local target point [x, y].
            velocity (float): Current speed in m/s.
            text_input (str): Driving instruction.
            reset_buffer (bool): Whether to reset temporal buffer before inference.
            use_notice (bool): Whether to include notice.
            notice_text (str): Notice text.
            notice_frame_id (int): Notice frame ID.

        Returns:
            tuple: (waypoints_np, end_prob, buffer_length) or (None, None, None) on failure.
                   waypoints_np is np.ndarray of shape (5, 2).
        """
        payload = {
            "session_id": self.session_id,
            "ego_id": int(ego_id),
            "rgb_front": self.encode_image(rgb_front),
            "rgb_left": self.encode_image(rgb_left),
            "rgb_right": self.encode_image(rgb_right),
            "rgb_rear": self.encode_image(rgb_rear),
            "lidar": self.encode_lidar(lidar),
            "num_points": int(num_points),
            "target_point": [float(target_point[0]), float(target_point[1])],
            "velocity": float(velocity),
            "text_input": str(text_input),
            "reset_buffer": bool(reset_buffer),
            "use_notice": bool(use_notice),
            "notice_text": str(notice_text),
            "notice_frame_id": int(notice_frame_id),
        }

        result = self._post("/infer", payload)
        if result is None:
            return (None, None, None)

        try:
            waypoints_np = np.array(result["waypoints"], dtype=np.float32)
            end_prob = float(result["end_prob"])
            buffer_length = int(result["buffer_length"])
            return (waypoints_np, end_prob, buffer_length)
        except (KeyError, TypeError, ValueError) as e:
            print("[LMDriveClient] Failed to parse response: %s" % str(e))
            return (None, None, None)

    def reset(self, ego_id):
        """Reset the temporal buffer for an ego vehicle on the server."""
        payload = {"session_id": self.session_id, "ego_id": int(ego_id)}
        result = self._post("/reset", payload)
        return result is not None

    def health_check(self):
        """Check if the server is healthy."""
        result = self._get("/health")
        if result is not None:
            return result.get("status") == "ok"
        return False
