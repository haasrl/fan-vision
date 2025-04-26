import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout

# ————— CONFIG —————
CAM_INDEX       = 1       # your webcam index
FLOW_MAG_THRESH = 1.0     # px/frame minimum flow magnitude
MOTION_PIXELS   = 5000    # min # flow-pixels for “running”
HOUGH_DP        = 1.2     # HoughCircles dp parameter
HOUGH_MIN_DIST  = 100     # min dist between detected centers
HOUGH_PARAM1    = 50      # Canny high-threshold
HOUGH_PARAM2    = 30      # accumulator threshold
HOUGH_MIN_R     = 30      # min circle radius
HOUGH_MAX_R     = 200     # max circle radius
OVERLAP_FACTOR  = 1.2     # if centers closer than (r1+r2)*OVERLAP_FACTOR, drop smaller
FRAME_INTERVAL  = 30      # ms between frames (~33ms → 30FPS)
# ————————————————————

class Fan:
    def __init__(self, x, y, r, init_gray):
        self.x, self.y, self.r = x, y, r
        self.prev_gray = init_gray.copy()
        h, w = init_gray.shape
        ys, xs = np.indices((h, w))
        self.rx = xs - w//2
        self.ry = ys - h//2
        self.running = False
        self.clockwise = True

    def process(self, gray_roi):
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

class MultiFanGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Fan Spin Detector")
        self._init_ui()
        self._open_cam()
        self.fans = []
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update)
        self.timer.start(FRAME_INTERVAL)

    def _init_ui(self):
        self.video = QLabel(alignment=Qt.AlignCenter)
        self.video.setFixedSize(800, 600)
        layout = QVBoxLayout(self)
        layout.addWidget(self.video)

    def _open_cam(self):
        self.cap = cv2.VideoCapture(CAM_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {CAM_INDEX}")
        ret, frm = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame")
        self.h, self.w = frm.shape[:2]
        self.base_gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

    def _detect_fans(self, gray):
        blur = cv2.GaussianBlur(gray, (9,9), 2)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
            param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
            minRadius=HOUGH_MIN_R, maxRadius=HOUGH_MAX_R
        )
        dets = []
        if circles is not None:
            for x,y,r in np.round(circles[0]).astype(int):
                dets.append((x,y,r))
        # Non-max suppression: keep largest circles, drop overlapping smaller ones
        dets.sort(key=lambda c: c[2], reverse=True)  # sort by radius desc
        keep = []
        for x,y,r in dets:
            too_close = False
            for (kx,ky,kr) in keep:
                d = np.hypot(x-kx, y-ky)
                if d < (r+kr)*OVERLAP_FACTOR:
                    too_close = True
                    break
            if not too_close:
                keep.append((x,y,r))
        # Instantiate Fan objects
        fans = []
        for x,y,r in keep:
            y0,y1 = max(0, y-r), min(self.h, y+r)
            x0,x1 = max(0, x-r), min(self.w, x+r)
            roi = gray[y0:y1, x0:x1]
            fans.append(Fan(x, y, r, roi))
        return fans

    def _update(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not self.fans:
            self.fans = self._detect_fans(gray)

        for fan in self.fans:
            x,y,r = fan.x, fan.y, fan.r
            y0,y1 = max(0, y-r), min(self.h, y+r)
            x0,x1 = max(0, x-r), min(self.w, x+r)
            roi = gray[y0:y1, x0:x1]

            running, cw = fan.process(roi)
            color = (0,255,0) if running else (0,0,255)
            cv2.circle(frame, (x,y), r, color, 2)
            arrow = "→" if cw else "←"
            cv2.putText(frame, arrow, (x - r//3, y + r//3),
                        cv2.FONT_HERSHEY_SIMPLEX, r/80, color, 2, cv2.LINE_AA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h2, w2, ch = rgb.shape
        img = QImage(rgb.data, w2, h2, ch*w2, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.video.setPixmap(pix.scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def closeEvent(self, e):
        self.cap.release()
        super().closeEvent(e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MultiFanGUI()
    win.show()
    sys.exit(app.exec_())

