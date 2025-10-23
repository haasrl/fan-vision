import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout

CAM_INDEX       = 0
FLOW_MAG_THRESH = 1.0
MOTION_PIXELS   = 5000
FRAME_INTERVAL  = 30

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
        blur   = cv2.GaussianBlur(gray, (9,9), 2)
        edges  = cv2.Canny(blur, 50, 150) 

        cv2.imshow("edges_for_hough", cv2.resize(edges,(400,400)))
        cv2.waitKey(1)

        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1.0,                    # accumulator resolution
            minDist=300,               # minimum center‐to‐center distance
            param1=100,                # Canny high‐threshold
            param2=20,                 # accumulator threshold (lower = more false circles)
            minRadius=80,              # smallest radius to look for
            maxRadius=100              # largest radius to look for
        )

        fans = []
        if circles is not None:
            for x, y, r in np.round(circles[0]).astype(int):
                # crop ROI safely
                y0, y1 = max(0,   y-r), min(self.h, y+r)
                x0, x1 = max(0,   x-r), min(self.w, x+r)
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
            arrow = "CW" if cw else "CCW"
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

