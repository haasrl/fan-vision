#!/usr/bin/env python3
"""
oaf-test-ip.py
Minimal, responsive RTSP/IP viewer + multi-fan detection — with explicit backend control.

Goal
----
Avoid the 30s FFmpeg stall by default. We try ONLY the simple path first (like sick.py), and
we do NOT fall back to FFmpeg unless you ask for it, or pass --fallback.

Backends
--------
--backend any    (default; same as sick.py: cv2.VideoCapture(url, CAP_ANY))
--backend msmf   (Windows Media Foundation)
--backend dshow  (DirectShow)
--backend ffmpeg (FFmpeg; add TCP/timeouts; fastest if the camera NEEDS it)

Flags
-----
--fallback           allow trying other backends if the chosen one fails (off by default)
--ffmpeg-open 3000   ms (default 4000)  Open timeout for FFmpeg
--ffmpeg-read 5000   ms (default 6000)  Read timeout for FFmpeg

Hotkeys: F=re-detect fans, D=toggle edges, Q/Esc=quit
"""

from __future__ import annotations
import sys, time, atexit, signal, threading, argparse
from typing import Optional, Union, Tuple, List
import cv2
import numpy as np

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout

# ----------------------------
# Defaults / tuning
# ----------------------------
DEFAULT_SRC: Union[int, str] = "rtsp://192.168.136.100:554/live/0"
FRAME_INTERVAL_MS = 30

# Hough & flow tuning
HOUGH_MIN_DIST = 300
HOUGH_PARAM1   = 100
HOUGH_PARAM2   = 70
HOUGH_MIN_R    = 160 
HOUGH_MAX_R    = 220

FLOW_MAG_THRESH = 1.8
MOTION_PIXELS   = 10000

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
        self.rx = xs - w//2
        self.ry = ys - h//2
        self.running = False
        self.clockwise = True

    def process(self, gray_roi: np.ndarray) -> Tuple[bool, bool]:
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray_roi, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        u, v = flow[...,0], flow[...,1]
        mag = np.hypot(u, v)
        mask = mag > FLOW_MAG_THRESH
        motion_pixels = int(mask.sum())

        if motion_pixels > MOTION_PIXELS:
            cross = self.rx[mask]*v[mask] - self.ry[mask]*u[mask]
            mean_cross = cross.mean()
            self.clockwise = (mean_cross < 0)
            self.running = True
        else:
            self.running = False

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
        ok, frame = self.grabber.get()
        if not ok or frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not self.fans:
            self.fans = self._detect_fans(gray)

        for fan in self.fans:
            x, y, r = fan.x, fan.y, fan.r
            y0, y1 = max(0, y-r), min(self.h, y+r)
            x0, x1 = max(0, x-r), min(self.w, x+r)
            roi = gray[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            running, cw = fan.process(roi)
            color = (0,255,0) if running else (0,0,255)
            cv2.circle(frame, (x,y), r, color, 2)
            cv2.putText(frame, "CW" if cw else "CCW",
                        (x - r//3, y + r//3), cv2.FONT_HERSHEY_SIMPLEX,
                        max(r/120.0, 0.6), color, 2, cv2.LINE_AA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h2, w2, ch = rgb.shape
        img = QImage(rgb.data, w2, h2, ch*w2, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.video.setPixmap(pix.scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

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
