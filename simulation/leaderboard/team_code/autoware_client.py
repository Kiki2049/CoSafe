"""Autoware bridge HTTP client (Python 3.7 / colmdriver env).

One client instance per ego vehicle.  Each client talks to a dedicated
``autoware_bridge.py`` process running inside an Autoware Docker container
(ROS_DOMAIN_ID-isolated).  Mirrors the retry/session pattern of
``lmdrive_client.py`` for consistency.
"""

from __future__ import annotations

import time

import requests


class AutowareClient(object):
    """HTTP client for a single Autoware bridge."""

    def __init__(self, server_url="http://localhost:8100", session_id="ego0",
                 step_timeout=1.0, reset_timeout=5.0, max_retries=5):
        self.server_url = server_url.rstrip("/")
        self.session_id = session_id
        self.step_timeout = step_timeout
        self.reset_timeout = reset_timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.trust_env = False

    def _post(self, endpoint, payload, timeout, retries):
        url = self.server_url + endpoint
        last_err = None
        for attempt in range(retries):
            try:
                resp = self._session.post(url, json=payload, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                wait = min(2 ** attempt, 10)
                print("[AutowareClient %s] %s attempt %d/%d failed: %s. retry in %ds"
                      % (self.session_id, endpoint, attempt + 1, retries,
                         str(e)[:120], wait))
                time.sleep(wait)
            except requests.HTTPError as e:
                print("[AutowareClient %s] HTTP error on %s: %s"
                      % (self.session_id, endpoint, str(e)))
                return None
        print("[AutowareClient %s] all %d retries exhausted for %s: %s"
              % (self.session_id, retries, endpoint, str(last_err)[:200]))
        return None

    def _get(self, endpoint, timeout):
        url = self.server_url + endpoint
        try:
            resp = self._session.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print("[AutowareClient %s] GET %s failed: %s"
                  % (self.session_id, endpoint, str(e)[:120]))
            return None

    def health(self):
        return self._get("/health", timeout=self.step_timeout) is not None

    def reset(self, global_plan_world):
        """Send the global plan as a list of [x, y, z, roll, pitch, yaw] (degrees).

        Returns True on success, False otherwise.
        """
        payload = {
            "session_id": self.session_id,
            "global_plan_world": [list(map(float, wp)) for wp in global_plan_world],
        }
        result = self._post("/reset", payload,
                            timeout=self.reset_timeout, retries=self.max_retries)
        return bool(result and result.get("ok"))

    def step(self, payload):
        """Send one-tick payload, return dict with steer/throttle/brake or None."""
        payload = dict(payload)
        payload["session_id"] = self.session_id
        result = self._post("/step", payload,
                            timeout=self.step_timeout, retries=1)
        return result

    def close(self):
        try:
            self._session.close()
        except Exception:
            pass
