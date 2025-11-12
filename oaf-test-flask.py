from __future__ import annotations
import sys, time, atexit, signal, threading, argparse
from typing import Optional, Union, Tuple, List
import cv2
import numpy as np
from collections import deque
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from flask import Flask, request, jsonify
import threading, time

class ApiServer:
    def __init__(self, app_ref):
        self.app_ref = app_ref  # reference to your App instance
        self.app = Flask(__name__)

        @self.app.get("/health")
        def health():
            return jsonify(ok=True, backend=self.app_ref.backend, running=self.app_ref.is_running())

        @self.app.post("/detect")
        def detect():
            # Ask the UI thread to re-detect fans on next tick
            self.app_ref.request_redetect()
            return jsonify(ok=True)

        @self.app.get("/status")
        def status():
            # Return current fan states (thread-safe snapshot)
            snap = self.app_ref.snapshot_status()
            return jsonify(snap)

        @self.app.post("/sleep")
        def sleep():
            self.app_ref.pause_grabber()
            return jsonify(ok=True)

        @self.app.post("/wake")
        def wake():
            ok = self.app_ref.resume_grabber()
            return jsonify(ok=bool(ok))

        @self.app.post("/params")
        def params():
            data = request.get_json(silent=True) or {}
            changed = self.app_ref.update_params(data)
            return jsonify(ok=True, changed=changed)

    def start(self, host="127.0.0.1", port=5001):
        t = threading.Thread(target=lambda: self.app.run(host=host, port=port, debug=False, use_reloader=False), daemon=True)
        t.start()
        return t

# ----------------------------
# Defaults / tuning
# ----------------------------
DEFAULT_SRC: Union[int, str] = "rtsp://192.168.1.13:554/live/0"
FRAME_INTERVAL_MS = 16 

# Hough & flow tuning
HOUGH_MIN_DIST = 80
HOUGH_PARAM1   = 100
HOUGH_PARAM2   = 70
HOUGH_MIN_R    = 80 
HOUGH_MAX_R    = 130

FLOW_MAG_THRESH = 3.0          # was 1.0; raise to be less sensitive
MOTION_PIXELS_START = 20000     # need this many moving pixels to declare "running"
MOTION_PIXELS_STOP  = 10000     # drop below this to declare "stopped" (hysteresis)
RING_INNER_FRACTION = 0.15     # analyze an annulus between 55%R and 95%R
RING_OUTER_FRACTION = 1.0
DIR_HISTORY = 10               # frames to keep for direction smoothing
DIR_FLIP_CONFIRM = 4           # need >= this many consecutive opposite votes to flip


# ----------------------------
# Graceful teardown for RTSP sessions
# ----------------------------
_caps: List[cv2.VideoCapture] = []
def track_cap(c: cv2.VideoCapture) -> cv2.VideoCapture:
    _caps.append(c); return c
def _release_all():
    for c in _caps:
        try: c.release()
        except Exception: pass
atexit.register(_release_all)
signal.signal(signal.SIGINT, lambda s,f: (_release_all(), sys.exit(130)))

# ----------------------------
# Helpers
# ----------------------------
def _append_q(url: str, kv: str) -> str:
    return url + ("&" if "?" in url else "?") + kv

def _ffmpeg_url(url: str) -> str:
    # Keep options minimal to avoid long stalls; TCP + short timeouts only
    for kv in ["rtsp_transport=tcp", "stimeout=4000000", "rw_timeout=6000000"]:
        url = _append_q(url, kv)
    return url

def _open_with_backend(src: Union[int, str], backend: str,
                       ff_open_ms: int, ff_read_ms: int) -> Tuple[Optional[cv2.VideoCapture], str]:
    if isinstance(src, int):
        cap = track_cap(cv2.VideoCapture(src))
        return (cap if cap.isOpened() else None), "USB"
    if backend == "any":
        cap = track_cap(cv2.VideoCapture(src, cv2.CAP_ANY))
        return (cap if cap.isOpened() else None), "CAP_ANY"
    if backend == "msmf":
        cap = track_cap(cv2.VideoCapture(src, cv2.CAP_MSMF))
        return (cap if cap.isOpened() else None), "CAP_MSMF"
    if backend == "dshow":
        cap = track_cap(cv2.VideoCapture(src, cv2.CAP_DSHOW))
        return (cap if cap.isOpened() else None), "CAP_DSHOW"
    if backend == "ffmpeg":
        u = _ffmpeg_url(str(src))
        cap = track_cap(cv2.VideoCapture(u, cv2.CAP_FFMPEG))
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        try: cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, float(ff_open_ms))
        except Exception: pass
        try: cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(ff_read_ms))
        except Exception: pass
        return (cap if cap.isOpened() else None), "CAP_FFMPEG(TCP)"
    return None, backend

def open_source(src: Union[int, str], backend: str, allow_fallback: bool,
                ff_open_ms: int, ff_read_ms: int) -> Tuple[cv2.VideoCapture, str]:
    order = [backend] if not allow_fallback else (
        [backend] + [b for b in ["any", "msmf", "dshow", "ffmpeg"] if b != backend]
    )
    errors = []
    for b in order:
        print(f"[connect] trying backend={b} ...")
        cap, desc = _open_with_backend(src, b, ff_open_ms, ff_read_ms)
        if cap is not None and cap.isOpened():
            print(f"[connect] SUCCESS with {desc}")
            return cap, desc
        errors.append(desc)
    raise RuntimeError("Failed to open source; attempted: " + ", ".join(errors))

# ----------------------------
# Threaded frame grabber (latest frame only)
# ----------------------------
class Grabber:
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.ok = False
        self.stopped = False
        self.t = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.t.start(); return self

    def stop(self):
        self.stopped = True
        try: self.t.join(timeout=0.5)
        except Exception: pass

    def _loop(self):
        while not self.stopped:
            ok, f = self.cap.read()
            if ok and f is not None:
                with self.lock:
                    self.ok = True
                    self.frame = f
            else:
                time.sleep(0.01)

    def get(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            return (self.ok and self.frame is not None), (None if self.frame is None else self.frame.copy())

# ----------------------------
# Fan logic
# ----------------------------
class Fan:
    def __init__(self, x: int, y: int, r: int, init_gray: np.ndarray):
        self.x, self.y, self.r = x, y, r
        self.prev_gray = init_gray.copy()

        h, w = init_gray.shape
        ys, xs = np.indices((h, w))
        cx, cy = w // 2, h // 2
        rx = xs - cx
        ry = ys - cy
        rsq = rx * rx + ry * ry
        r2  = (r * r)
        # precompute an annulus mask (ignore hub and far outside)
        inner = (RING_INNER_FRACTION * r) ** 2
        outer = (RING_OUTER_FRACTION * r) ** 2
        self.annulus = (rsq >= inner) & (rsq <= outer)

        # store geometry for direction calc
        self.rx, self.ry = rx, ry

        # state
        self.running = False
        self.clockwise = True

        # temporal smoothing
        self.dir_hist = deque(maxlen=DIR_HISTORY)
        self.last_dir = 1  # +1 = CW, -1 = CCW

    def process(self, gray_roi: np.ndarray):
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray_roi, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        u, v = flow[..., 0], flow[..., 1]

        # compensate global jitter: subtract median flow
        u = u - np.median(u)
        v = v - np.median(v)

        mag = np.hypot(u, v)

        # motion mask: magnitude threshold + annulus restriction
        mask = (mag > FLOW_MAG_THRESH) & self.annulus
        motion_pixels = int(mask.sum())

        # hysteresis for running / stopped
        if self.running:
            self.running = motion_pixels > MOTION_PIXELS_STOP
        else:
            self.running = motion_pixels > MOTION_PIXELS_START

        # direction (only if running)
        if self.running and motion_pixels > 0:
            # cross-product sign around center → rotation sign
            cross = self.rx[mask] * v[mask] - self.ry[mask] * u[mask]
            mean_cross = float(cross.mean()) if cross.size else 0.0
            inst_dir = 1 if (mean_cross < 0.0) else -1   # +1 = CW, -1 = CCW
            self.dir_hist.append(inst_dir)

            # require a few consecutive opposite votes to flip
            if len(self.dir_hist) >= DIR_FLIP_CONFIRM:
                # check last K
                recent = list(self.dir_hist)[-DIR_FLIP_CONFIRM:]
                if all(d == -self.last_dir for d in recent):
                    self.last_dir = -self.last_dir

            # also set from majority over the whole history as a backstop
            if self.dir_hist:
                maj = 1 if sum(self.dir_hist) >= 0 else -1
                # nudge last_dir toward the majority without flipping too often
                if maj != self.last_dir and self.dir_hist.count(maj) > DIR_FLIP_CONFIRM:
                    self.last_dir = maj
        else:
            # not running; keep last_dir as-is
            pass

        self.clockwise = (self.last_dir == 1)
        self.prev_gray = gray_roi.copy()
        return self.running, self.clockwise
# ----------------------------
# GUI
# ----------------------------
class App(QWidget):
    def __init__(self, src: Union[int, str], backend: str, allow_fallback: bool,
                 ff_open_ms: int, ff_read_ms: int):
        super().__init__()
        self.setWindowTitle("IP Viewer + Multi-Fan Detector")
        self.video = QLabel(alignment=Qt.AlignCenter)
        self.video.setFixedSize(900, 700)
        layout = QVBoxLayout(self)
        layout.addWidget(self.video)

        self.cap, self.backend = open_source(src, backend, allow_fallback, ff_open_ms, ff_read_ms)

        # Show window immediately with a connecting splash
        self.video.setText(f"Connecting via {self.backend}...")
        QApplication.processEvents()

        ok, first = self.cap.read()
        if not ok or first is None:
            raise RuntimeError("Failed to read first frame")

        self.h, self.w = first.shape[:2]
        self.gray0 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

        # Start grabber
        self.grabber = Grabber(self.cap).start()

        # Fan detection
        self.fans: List[Fan] = []
        self.show_edges = False

        # UI timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(FRAME_INTERVAL_MS)
        self._lock = threading.Lock()
        self._redetect_flag = False
        self._paused = False
        self.last_status = {"detected": False, "fans": [], "t": 0.0}

        # start REST server
        self.api = ApiServer(self)
        self.api.start(port=5001)   # LabVIEW points to http://127.0.0.1:5001

    # LabVIEW triggers this: re-run Hough next frame
    def request_redetect(self):
        with self._lock:
            self._redetect_flag = True

    def is_running(self):
        return not self._paused

    def pause_grabber(self):
        self._paused = True
        # optional: self.grabber.stop(); self.cap.release()
        #           or just skip processing in _tick

    def resume_grabber(self):
        self._paused = False
        # optional: reopen if you actually released resources
        return True

    def update_params(self, data: dict):
        changed = {}
        # Example: allow LabVIEW to tweak thresholds on the fly
        global FLOW_MAG_THRESH, MOTION_PIXELS_HI, MOTION_PIXELS_LO
        if "flow_mag" in data:
            FLOW_MAG_THRESH = float(data["flow_mag"]); changed["flow_mag"] = FLOW_MAG_THRESH
        if "motion_hi" in data:
            MOTION_PIXELS_HI = int(data["motion_hi"]); changed["motion_hi"] = MOTION_PIXELS_HI
        if "motion_lo" in data:
            MOTION_PIXELS_LO = int(data["motion_lo"]); changed["motion_lo"] = MOTION_PIXELS_LO
        return changed

    def snapshot_status(self):
        with self._lock:
            fans_out = []
            for f in self.fans:
                fans_out.append({
                    "x": int(f.x), "y": int(f.y), "r": int(f.r),
                    "running": bool(f.running),
                    "direction": "CW" if f.clockwise else "CCW",
                    "confidence": round(abs(getattr(f, "dir_score", 0.0)), 3)
                })
            snap = {"detected": len(self.fans) > 0, "fans": fans_out, "t": time.time()}
            self.last_status = snap
            return snap

    def _detect_fans(self, gray: np.ndarray) -> List[Fan]:
        blur   = cv2.GaussianBlur(gray, (9,9), 2)
        edges  = cv2.Canny(blur, 50, 150)

        if self.show_edges:
            cv2.imshow("edges_for_hough", cv2.resize(edges, (400, 400)))
            cv2.waitKey(1)

        circles = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT, dp=1.0,
            minDist=HOUGH_MIN_DIST, param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
            minRadius=HOUGH_MIN_R, maxRadius=HOUGH_MAX_R
        )

        fans: List[Fan] = []
        if circles is not None:
            for x, y, r in np.round(circles[0]).astype(int):
                y0, y1 = max(0, y-r), min(self.h, y+r)
                x0, x1 = max(0, x-r), min(self.w, x+r)
                roi = gray[y0:y1, x0:x1]
                if roi.size == 0: continue
                fans.append(Fan(x, y, r, roi))
        return fans

    def _tick(self):
        # Pause loop if requested
        if getattr(self, "_paused", False):
            return

        ok, frame = self.grabber.get()
        if not ok or frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Re-detect fans if asked (or if none yet)
        with self._lock:
            if getattr(self, "_redetect_flag", False) or not self.fans:
                self.fans = self._detect_fans(gray)
                self._redetect_flag = False

        # Process each fan and draw overlays
        fans_out = []
        for fan in self.fans:
            x, y, r = fan.x, fan.y, fan.r
            y0, y1 = max(0, y - r), min(self.h, y + r)
            x0, x1 = max(0, x - r), min(self.w, x + r)
            roi = gray[y0:y1, x0:x1]
            if roi.size == 0:
                continue

            running, cw = fan.process(roi)

            # Color & label: STOP=yellow, CW=green, CCW=red
            if running:
                if cw:
                    color = (0, 0, 255)     # CW -> green (BGR)
                    label = "CCW"
                else:
                    color = (0, 255, 0)     # CCW -> red (BGR)
                    label = "CW"
            else:
                color = (0, 255, 255)       # not moving -> yellow (BGR)
                label = "STOP"

            # Thicker outline (6 px)
            cv2.circle(frame, (x, y), r, color, 20)
            cv2.putText(
                frame, label, (x - r // 3, y + r // 3),
                cv2.FONT_HERSHEY_SIMPLEX, max(r / 120.0, 0.6),
                color, 2, cv2.LINE_AA
            )

            # Collect status for API consumers (optional)
            fans_out.append({
                "x": int(x), "y": int(y), "r": int(r),
                "running": bool(running),
                "direction": "CW" if cw else "CCW",
                "confidence": round(abs(getattr(fan, "dir_score", 0.0)), 3)
            })

        # Optional: keep last_status fresh for /status without extra locking
        try:
            with self._lock:
                self.last_status = {
                    "detected": len(self.fans) > 0,
                    "fans": fans_out,
                    "t": __import__("time").time()
                }
        except Exception:
            pass

        # Push to GUI
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h2, w2, ch = rgb.shape
        qimg = QImage(rgb.data, w2, h2, ch * w2, QImage.Format_RGB888)
        self.video.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )


    # Hotkeys
    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        elif e.key() == Qt.Key_F:
            self.fans = []  # re-detect
        elif e.key() == Qt.Key_D:
            self.show_edges = not self.show_edges
            if not self.show_edges:
                try: cv2.destroyWindow("edges_for_hough")
                except Exception: pass

    def closeEvent(self, e):
        try: self.grabber.stop()
        except Exception: pass
        try: self.cap.release()
        except Exception: pass
        super().closeEvent(e)

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="IP viewer + multi-fan detector (backend-controlled)")
    p.add_argument("source", nargs="?", default=None, help="RTSP URL or USB index (int).")
    p.add_argument("--backend", default="any", choices=["any","msmf","dshow","ffmpeg"],
                   help="Capture backend to use first (default: any).")
    p.add_argument("--fallback", action="store_true",
                   help="If set, try other backends if the chosen one fails.")
    p.add_argument("--ffmpeg-open", type=int, default=4000, help="FFmpeg open timeout (ms).")
    p.add_argument("--ffmpeg-read", type=int, default=6000, help="FFmpeg read timeout (ms).")
    return p.parse_args()

def resolve_src(arg: Optional[str]) -> Union[int, str]:
    if arg is None: return DEFAULT_SRC
    try: return int(arg)
    except ValueError: return arg

def main():
    a = parse_args()
    src = resolve_src(a.source)
    app = QApplication(sys.argv)
    win = App(src, backend=a.backend, allow_fallback=a.fallback,
              ff_open_ms=a.ffmpeg_open, ff_read_ms=a.ffmpeg_read)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
