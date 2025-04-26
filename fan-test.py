import sys
import time
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout

# ————— CONFIG —————
FLOW_MAG_THRESH   = 1.0     # px/frame minimum flow magnitude to count
MOTION_PIXELS     = 5000    # min number of flow pixels to call "running"
CROSS_VOTE_THRESH = 0.0     # min mean_cross to cast a vote
TEST_DURATION_SEC = 10.0    # sampling window length (s)
FRAME_INTERVAL_MS = 33      # ~30 FPS
# ——————————————————

class FanMonitorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fan Spin Detector")
        self._init_ui()
        self._open_camera()
        self._init_state()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(FRAME_INTERVAL_MS)

    def _init_ui(self):
        app = QApplication.instance()
        app.setStyle("Fusion")
        dark = QPalette()
        dark.setColor(QPalette.Window, QColor(45,45,45))
        dark.setColor(QPalette.WindowText, Qt.white)
        dark.setColor(QPalette.Base, QColor(30,30,30))
        dark.setColor(QPalette.Text, Qt.white)
        app.setPalette(dark)

        self.video_label = QLabel()
        self.video_label.setFixedSize(640,480)
        self.video_label.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel("Stopped")
        self.dir_label    = QLabel("–")
        for lbl in (self.status_label, self.dir_label):
            lbl.setAlignment(Qt.AlignCenter)
            f = QFont(); f.setPointSize(14)
            lbl.setFont(f)
            lbl.setFixedSize(140,40)
        self.status_label.setStyleSheet("background-color:red; color:white; border-radius:8px;")

        self.btn_test = QPushButton("Start 10s Test")
        self.btn_test.clicked.connect(self._start_test)

        hl = QHBoxLayout()
        hl.addWidget(self.status_label)
        hl.addWidget(self.dir_label)
        v = QVBoxLayout(self)
        v.addWidget(self.video_label)
        v.addLayout(hl)
        v.addWidget(self.btn_test, alignment=Qt.AlignCenter)

    def _open_camera(self):
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame")
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Precompute for cross-product
        h, w = self.prev_gray.shape
        ys, xs = np.indices((h, w))
        self.rx = xs - w//2
        self.ry = ys - h//2

    def _init_state(self):
        self.clockwise = True
        self.in_test     = False
        self.test_start  = 0.0
        self.cw_votes    = 0
        self.ccw_votes   = 0

    def _start_test(self):
        self.in_test    = True
        self.test_start = time.time()
        self.cw_votes   = 0
        self.ccw_votes  = 0
        self.btn_test.setEnabled(False)
        self.status_label.setText("Testing…")
        self.status_label.setStyleSheet("background-color:orange; color:white; border-radius:8px;")
        self.dir_label.setText("…")

    def _end_test(self):
        total = self.cw_votes + self.ccw_votes
        if total == 0:
            self.status_label.setText("No Motion")
            self.status_label.setStyleSheet("background-color:gray; color:white; border-radius:8px;")
            self.dir_label.setText("–")
        else:
            self.clockwise = (self.cw_votes > self.ccw_votes)
            self.status_label.setText("Done")
            self.status_label.setStyleSheet("background-color:green; color:white; border-radius:8px;")
            self.dir_label.setText("→" if self.clockwise else "←")
        self.btn_test.setEnabled(True)
        self.in_test = False

    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) Dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        u, v = flow[...,0], flow[...,1]

        # 2) Motion mask & pixel count
        mag = np.sqrt(u*u + v*v)
        mask = mag > FLOW_MAG_THRESH
        motion_pixels = int(mask.sum())

        if motion_pixels > MOTION_PIXELS:
            # 3) Cross-product for angular direction
            cross = self.rx[mask]*v[mask] - self.ry[mask]*u[mask]
            mean_cross = cross.mean()
            frame_cw   = (mean_cross < 0)

            if self.in_test:
                # voting
                if mean_cross < -CROSS_VOTE_THRESH:
                    self.cw_votes += 1
                elif mean_cross > CROSS_VOTE_THRESH:
                    self.ccw_votes += 1
            else:
                # live feedback
                self.clockwise = frame_cw
                self.status_label.setText("Running")
                self.status_label.setStyleSheet(
                    "background-color:green; color:white; border-radius:8px;"
                )
                self.dir_label.setText("→" if self.clockwise else "←")

        else:
            # below motion threshold => stopped
            if not self.in_test:
                self.status_label.setText("Stopped")
                self.status_label.setStyleSheet(
                    "background-color:red; color:white; border-radius:8px;"
                )
                self.dir_label.setText("–")

        # 4) Show video
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h2, w2, ch = rgb.shape
        qt_img = QImage(rgb.data, w2, h2, ch*w2, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(pix.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        self.prev_gray = gray

        # 5) End test if time’s up
        if self.in_test and (time.time() - self.test_start) >= TEST_DURATION_SEC:
            self._end_test()

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    gui = FanMonitorGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

