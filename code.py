# ============================================================
# Edge Detection GUI — PyQt5
# Operators: Roberts, Prewitt, Sobel, Laplacian, LoG, Canny
# ============================================================
# Install: pip install PyQt5 opencv-python numpy scipy

import sys
import cv2
import numpy as np
from scipy.ndimage import label as scipy_label
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QSlider, QFileDialog,
    QGroupBox, QRadioButton, QButtonGroup, QSizePolicy,
    QStatusBar, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

# ============================================================
# KERNELS
# ============================================================
ROBERTS_GX = np.array([[ 1, 0],[ 0,-1]], dtype=np.float64)
ROBERTS_GY = np.array([[ 0, 1],[-1, 0]], dtype=np.float64)
PREWITT_GX = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float64)
PREWITT_GY = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=np.float64)
SOBEL_GX   = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float64)
SOBEL_GY   = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float64)
LAPLACIAN4 = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float64)
LAPLACIAN8 = np.array([[1,1,1],[1,-8,1],[1,1,1]], dtype=np.float64)

def apply_kernel(img, kernel):
    return cv2.filter2D(img, cv2.CV_64F, np.flip(kernel))

def normalize(arr):
    arr = np.abs(arr)
    if arr.max() == 0: return np.zeros_like(arr, dtype=np.uint8)
    return (arr / arr.max() * 255).astype(np.uint8)

def nms_vectorized(M, alpha):
    angle = alpha % 180
    out   = M.copy()
    mask0   = (angle < 22.5)  | (angle >= 157.5)
    cond0   = (M >= np.roll(M, 1, axis=1)) & (M >= np.roll(M,-1, axis=1))
    mask90  = (angle >= 67.5) & (angle < 112.5)
    cond90  = (M >= np.roll(M, 1, axis=0)) & (M >= np.roll(M,-1, axis=0))
    mask45  = (angle >= 22.5) & (angle < 67.5)
    cond45  = (M >= np.roll(np.roll(M, 1,axis=0),-1,axis=1)) & \
              (M >= np.roll(np.roll(M,-1,axis=0), 1,axis=1))
    mask135 = (angle >= 112.5) & (angle < 157.5)
    cond135 = (M >= np.roll(np.roll(M, 1,axis=0), 1,axis=1)) & \
              (M >= np.roll(np.roll(M,-1,axis=0),-1,axis=1))
    keep = (mask0&cond0)|(mask90&cond90)|(mask45&cond45)|(mask135&cond135)
    out[~keep] = 0
    return out

# ============================================================
# OPERATOR FUNCTIONS
# ============================================================
def op_roberts(img, **kwargs):
    Gx = apply_kernel(img, ROBERTS_GX)
    Gy = apply_kernel(img, ROBERTS_GY)
    return normalize(np.sqrt(Gx**2 + Gy**2))

def op_prewitt(img, sigma=1.0, **kwargs):
    b = cv2.GaussianBlur(img.astype(np.uint8),(0,0),sigma).astype(np.float64) if sigma>0 else img
    Gx = apply_kernel(b, PREWITT_GX)
    Gy = apply_kernel(b, PREWITT_GY)
    return normalize(np.sqrt(Gx**2 + Gy**2))

def op_sobel(img, sigma=1.0, **kwargs):
    b = cv2.GaussianBlur(img.astype(np.uint8),(0,0),sigma).astype(np.float64) if sigma>0 else img
    Gx = apply_kernel(b, SOBEL_GX)
    Gy = apply_kernel(b, SOBEL_GY)
    return normalize(np.sqrt(Gx**2 + Gy**2))

def op_laplacian(img, conn=4, **kwargs):
    k = LAPLACIAN4 if conn == 4 else LAPLACIAN8
    return normalize(apply_kernel(img, k))

def op_log(img, sigma=1.4, **kwargs):
    b = cv2.GaussianBlur(img.astype(np.uint8),(0,0),sigma).astype(np.float64)
    return normalize(apply_kernel(b, LAPLACIAN4))

def op_canny(img, sigma=1.4, t_low=30, t_high=90, **kwargs):
    ksize = int(6*sigma+1)|1
    b  = cv2.GaussianBlur(img.astype(np.uint8),(ksize,ksize),sigma)
    Gx = apply_kernel(b.astype(np.float64), SOBEL_GX)
    Gy = apply_kernel(b.astype(np.float64), SOBEL_GY)
    M  = np.sqrt(Gx**2+Gy**2)
    al = np.degrees(np.arctan2(Gy, Gx))
    Mn = (M/M.max()*255) if M.max()>0 else M
    nms = nms_vectorized(Mn, al)
    tH  = nms.max() * (t_high/255.0)
    tL  = nms.max() * (t_low /255.0)
    strong = nms >= tH
    cand   = nms >= tL
    lbl, _ = scipy_label(cand)
    sl = set(lbl[strong]); sl.discard(0)
    res = strong.copy()
    for s in sl: res |= (lbl==s)
    return (res*255).astype(np.uint8)

# ============================================================
# MAIN WINDOW
# ============================================================
class EdgeDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.img_gray = None
        self.img_path = None
        self.result   = None
        self.timer    = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.apply_operator)
        self.setWindowTitle("Edge Detection — Interactive GUI")
        self.setMinimumSize(1100, 650)
        self._build_ui()
        self._apply_dark_theme()

    # --------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12,12,12,12)

        # ===== LEFT PANEL =====
        left = QVBoxLayout()
        left.setSpacing(10)

        btn_load = QPushButton("📂  Load Image")
        btn_load.setFixedHeight(38)
        btn_load.clicked.connect(self.load_image)
        left.addWidget(btn_load)

        op_group = QGroupBox("Operator")
        op_layout = QVBoxLayout()
        self.op_buttons = QButtonGroup()
        operators = [
            ("Roberts Cross", "roberts"),
            ("Prewitt",       "prewitt"),
            ("Sobel",         "sobel"),
            ("Laplacian",     "laplacian"),
            ("LoG",           "log"),
            ("Canny",         "canny"),
        ]
        for i, (label, key) in enumerate(operators):
            rb = QRadioButton(label)
            rb.setProperty("op_key", key)
            rb.setFont(QFont("Courier New", 10))
            if key == "canny": rb.setChecked(True)
            rb.toggled.connect(self._on_operator_changed)
            self.op_buttons.addButton(rb, i)
            op_layout.addWidget(rb)
        op_group.setLayout(op_layout)
        left.addWidget(op_group)

        self.param_group = QGroupBox("Parameters")
        self.param_layout = QVBoxLayout()
        self.param_group.setLayout(self.param_layout)
        left.addWidget(self.param_group)

        btn_row = QHBoxLayout()
        btn_apply = QPushButton("▶  Apply")
        btn_apply.setFixedHeight(34)
        btn_apply.clicked.connect(self.apply_operator)
        btn_reset = QPushButton("↺  Reset")
        btn_reset.setFixedHeight(34)
        btn_reset.clicked.connect(self.reset_params)
        btn_row.addWidget(btn_apply)
        btn_row.addWidget(btn_reset)
        left.addLayout(btn_row)

        btn_save = QPushButton("💾  Save Result PNG")
        btn_save.setFixedHeight(34)
        btn_save.clicked.connect(self.save_result)
        left.addWidget(btn_save)

        self.stats_label = QLabel("Edge pixels: —")
        self.stats_label.setFont(QFont("Courier New", 9))
        self.stats_label.setAlignment(Qt.AlignCenter)
        left.addWidget(self.stats_label)

        left.addStretch()
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(260)
        main_layout.addWidget(left_widget)

        div = QFrame()
        div.setFrameShape(QFrame.VLine)
        div.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(div)

        # ===== RIGHT PANEL =====
        right = QVBoxLayout()
        right.setSpacing(8)

        img_row = QHBoxLayout()
        img_row.setSpacing(10)

        left_img = QVBoxLayout()
        lbl_orig_title = QLabel("Original (Grayscale)")
        lbl_orig_title.setAlignment(Qt.AlignCenter)
        lbl_orig_title.setFont(QFont("Courier New", 10, QFont.Bold))
        self.lbl_orig = QLabel()
        self.lbl_orig.setAlignment(Qt.AlignCenter)
        self.lbl_orig.setMinimumSize(380, 380)
        self.lbl_orig.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_orig.setStyleSheet("border: 1px solid #2e2e3a; background: #0a0a0c;")
        left_img.addWidget(lbl_orig_title)
        left_img.addWidget(self.lbl_orig)

        right_img = QVBoxLayout()
        self.lbl_result_title = QLabel("Result — Canny")
        self.lbl_result_title.setAlignment(Qt.AlignCenter)
        self.lbl_result_title.setFont(QFont("Courier New", 10, QFont.Bold))
        self.lbl_result_title.setStyleSheet("color: #6aaddc;")
        self.lbl_result = QLabel()
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.setMinimumSize(380, 380)
        self.lbl_result.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_result.setStyleSheet("border: 1px solid #2e2e3a; background: #0a0a0c;")
        right_img.addWidget(self.lbl_result_title)
        right_img.addWidget(self.lbl_result)

        img_row.addLayout(left_img)
        img_row.addLayout(right_img)
        right.addLayout(img_row)

        right_widget = QWidget()
        right_widget.setLayout(right)
        main_layout.addWidget(right_widget, stretch=1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Load an image to get started...")

        self._build_params("canny")

    # --------------------------------------------------------
    def _make_slider(self, label, min_val, max_val, default, scale=1.0, decimals=1):
        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(0,4,0,0)
        layout.setSpacing(2)

        row = QHBoxLayout()
        lbl_name = QLabel(label)
        lbl_name.setFont(QFont("Courier New", 9))
        lbl_val  = QLabel(f"{default * scale:.{decimals}f}")
        lbl_val.setFont(QFont("Courier New", 9, QFont.Bold))
        lbl_val.setStyleSheet("color: #e8b845;")
        lbl_val.setAlignment(Qt.AlignRight)
        row.addWidget(lbl_name)
        row.addWidget(lbl_val)
        layout.addLayout(row)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(max(1, (max_val - min_val) // 5))

        def on_change(val):
            real = val * scale
            lbl_val.setText(f"{real:.{decimals}f}")
            self.timer.start(300)

        slider.valueChanged.connect(on_change)
        layout.addWidget(slider)

        container.slider    = slider
        container.get_value = lambda: slider.value() * scale
        container.reset_val = default
        return container

    # --------------------------------------------------------
    def _build_params(self, op_key):
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.sliders = {}

        if op_key == "roberts":
            lbl = QLabel("No parameters\n(fixed 2×2 kernel)")
            lbl.setFont(QFont("Courier New", 9))
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #555;")
            self.param_layout.addWidget(lbl)

        elif op_key in ("prewitt", "sobel"):
            s = self._make_slider("Pre-blur σ", 0, 30, 10, scale=0.1, decimals=1)
            self.param_layout.addWidget(s)
            self.sliders["sigma"] = s

        elif op_key == "laplacian":
            s = self._make_slider("Connectivity", 0, 1, 0, scale=1, decimals=0)
            s.get_value = lambda: 4 if s.slider.value() == 0 else 8
            hint = QLabel("0 = 4-connected  |  1 = 8-connected")
            hint.setFont(QFont("Courier New", 8))
            hint.setStyleSheet("color: #666;")
            self.param_layout.addWidget(s)
            self.param_layout.addWidget(hint)
            self.sliders["conn"] = s

        elif op_key == "log":
            s = self._make_slider("Gaussian σ", 5, 50, 14, scale=0.1, decimals=1)
            self.param_layout.addWidget(s)
            self.sliders["sigma"] = s

        elif op_key == "canny":
            s_sigma = self._make_slider("Gaussian σ",  5, 50,  14, scale=0.1, decimals=1)
            s_tlow  = self._make_slider("Threshold Low",  1, 254, 30, scale=1, decimals=0)
            s_thigh = self._make_slider("Threshold High", 1, 254, 90, scale=1, decimals=0)
            hint = QLabel("Recommended ratio:  T_high = 2× or 3× T_low")
            hint.setFont(QFont("Courier New", 8))
            hint.setStyleSheet("color: #e8b845; font-style: italic;")
            self.param_layout.addWidget(s_sigma)
            self.param_layout.addWidget(s_tlow)
            self.param_layout.addWidget(s_thigh)
            self.param_layout.addWidget(hint)
            self.sliders["sigma"]  = s_sigma
            self.sliders["t_low"]  = s_tlow
            self.sliders["t_high"] = s_thigh

        self.param_layout.addStretch()

    # --------------------------------------------------------
    def _on_operator_changed(self, checked):
        if not checked: return
        rb     = self.sender()
        op_key = rb.property("op_key")
        self._build_params(op_key)
        colors = {
            "roberts":"#f0924a","prewitt":"#e8b845","sobel":"#5fb3ba",
            "laplacian":"#d47bb5","log":"#78bf50","canny":"#6aaddc"
        }
        names = {
            "roberts":"Roberts Cross","prewitt":"Prewitt",
            "sobel":"Sobel","laplacian":"Laplacian",
            "log":"LoG","canny":"Canny"
        }
        c = colors.get(op_key, "#e2e2e8")
        self.lbl_result_title.setText(f"Result — {names[op_key]}")
        self.lbl_result_title.setStyleSheet(f"color: {c};")
        if self.img_gray is not None:
            self.timer.start(100)

    # --------------------------------------------------------
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", "Images (*.jpg *.jpeg *.png *.bmp *.tif)"
        )
        if not path: return
        self.img_path = path
        img_bgr = cv2.imread(path)
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w    = gray.shape
        if max(h, w) > 600:
            scale = 600 / max(h, w)
            gray  = cv2.resize(gray, (int(w*scale), int(h*scale)))
        self.img_gray = gray.astype(np.float64)
        self._show_image(self.lbl_orig, gray)
        self.status.showMessage(f"Loaded: {path}  ({gray.shape[1]}×{gray.shape[0]} px)")
        self.apply_operator()

    # --------------------------------------------------------
    def apply_operator(self):
        if self.img_gray is None:
            self.status.showMessage("⚠️  Please load an image first!")
            return
        for btn in self.op_buttons.buttons():
            if btn.isChecked():
                op_key = btn.property("op_key")
                break
        params = {k: s.get_value() for k, s in self.sliders.items()}
        op_map = {
            "roberts":   op_roberts,
            "prewitt":   op_prewitt,
            "sobel":     op_sobel,
            "laplacian": op_laplacian,
            "log":       op_log,
            "canny":     op_canny,
        }
        try:
            self.result = op_map[op_key](self.img_gray, **params)
        except Exception as e:
            self.status.showMessage(f"Error: {e}")
            return
        self._show_image(self.lbl_result, self.result)
        total  = self.result.shape[0] * self.result.shape[1]
        n_edge = int((self.result > 50).sum())
        pct    = n_edge / total * 100
        self.stats_label.setText(f"Edge pixels: {n_edge:,}  ({pct:.1f}%)")
        self.status.showMessage(
            f"[{op_key.upper()}]  params={params}  →  {n_edge:,} px ({pct:.1f}%)"
        )

    # --------------------------------------------------------
    def reset_params(self):
        for s in self.sliders.values():
            s.slider.setValue(s.reset_val)
        self.timer.start(100)

    # --------------------------------------------------------
    def save_result(self):
        if self.result is None:
            self.status.showMessage("⚠️  No result to save!")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Result", "edge_result.png", "PNG (*.png)"
        )
        if path:
            cv2.imwrite(path, self.result)
            self.status.showMessage(f"✅  Saved: {path}")

    # --------------------------------------------------------
    def _show_image(self, label, img_array):
        h, w = img_array.shape[:2]
        qimg = QImage(
            img_array.astype(np.uint8).data, w, h, w,
            QImage.Format_Grayscale8
        )
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(
            label.width()-4, label.height()-4,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pix)

    # --------------------------------------------------------
    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0f0f11;
                color: #e2e2e8;
            }
            QGroupBox {
                border: 1px solid #2e2e3a;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                font-family: 'Courier New';
                font-size: 10px;
                font-weight: bold;
                color: #8888a0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                color: #8888a0;
            }
            QPushButton {
                background-color: #1e1e2a;
                border: 1px solid #3e3e5a;
                border-radius: 5px;
                padding: 6px 12px;
                font-family: 'Courier New';
                font-size: 10px;
                color: #e2e2e8;
            }
            QPushButton:hover {
                background-color: #2e2e4a;
                border-color: #5fb3ba;
                color: #5fb3ba;
            }
            QPushButton:pressed {
                background-color: #5fb3ba;
                color: #0f0f11;
            }
            QRadioButton {
                spacing: 8px;
                font-family: 'Courier New';
                font-size: 10px;
                color: #c0c0d0;
            }
            QRadioButton:checked {
                color: #e8b845;
                font-weight: bold;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #2e2e3a;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #5fb3ba;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #5fb3ba;
                border-radius: 2px;
            }
            QStatusBar {
                background: #0a0a0c;
                color: #666688;
                font-family: 'Courier New';
                font-size: 9px;
            }
            QLabel { font-family: 'Courier New'; }
        """)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.img_gray is not None:
            self._show_image(self.lbl_orig, self.img_gray.astype(np.uint8))
        if self.result is not None:
            self._show_image(self.lbl_result, self.result)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = EdgeDetectionGUI()
    win.show()
    sys.exit(app.exec_())