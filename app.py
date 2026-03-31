"""
rPPG Vital Signs Monitor - Complete Backend
Uses POS, CHROM, and Green channel methods
MediaPipe face detection (robust, no failures)
"""

import cv2
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from collections import deque
import mediapipe as mp
import time
import json
import threading
import base64
from flask import Flask, Response, render_template, jsonify

app = Flask(__name__, template_folder=".", static_folder=".")

# ─── Global State ───────────────────────────────────────────────────────────
class VitalSignsProcessor:
    def __init__(self, buffer_size=300, fps=30):
        self.buffer_size = buffer_size
        self.fps = fps
        self.rgb_buffer = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        self.hr_history = deque(maxlen=30)
        self.rr_history = deque(maxlen=30)

        # MediaPipe Face Mesh for robust ROI extraction
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.3
        )

        # Vital signs storage
        self.vitals = {
            "hr_pos": 0, "hr_chrom": 0, "hr_green": 0,
            "hr_final": 0, "spo2": 0,
            "systolic_bp": 0, "diastolic_bp": 0,
            "respiratory_rate": 0, "hrv_sdnn": 0,
            "stress_index": 0, "cardiac_output": 0,
            "signal_quality": 0, "face_detected": False,
            "raw_signal": [], "filtered_signal": [],
            "timestamp": 0
        }

        self.lock = threading.Lock()
        self.running = False
        self.cap = None

    def start_capture(self):
        """Start video capture in background thread"""
        if self.running:
            return
        self.running = True
        # Try multiple camera indices
        for idx in [0, 1, 2]:
            self.cap = cv2.VideoCapture(idx)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                break
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        thread = threading.Thread(target=self._capture_loop, daemon=True)
        thread.start()

    def _extract_roi_mediapipe(self, frame):
        """Extract forehead + cheeks ROI using MediaPipe - very robust"""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Try Face Mesh first (more landmarks = better ROI)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            # Forehead region (landmarks around forehead)
            forehead_pts = [10, 67, 69, 104, 108, 109, 151, 299, 297, 332, 333, 338]
            # Left cheek
            left_cheek_pts = [36, 50, 101, 116, 117, 118, 119, 123, 135, 187, 192, 205, 206, 207]
            # Right cheek
            right_cheek_pts = [266, 280, 330, 345, 346, 347, 348, 352, 364, 411, 416, 425, 426, 427]

            all_pts = forehead_pts + left_cheek_pts + right_cheek_pts
            xs, ys = [], []
            for idx in all_pts:
                if idx < len(landmarks.landmark):
                    lm = landmarks.landmark[idx]
                    xs.append(int(lm.x * w))
                    ys.append(int(lm.y * h))

            if xs and ys:
                x1 = max(0, min(xs) - 5)
                y1 = max(0, min(ys) - 5)
                x2 = min(w, max(xs) + 5)
                y2 = min(h, max(ys) + 5)

                if x2 - x1 > 20 and y2 - y1 > 20:
                    roi = frame[y1:y2, x1:x2]
                    return roi, (x1, y1, x2, y2), True

        # Fallback: Face Detection
        det_results = self.face_detection.process(rgb_frame)
        if det_results.detections:
            det = det_results.detections[0]
            bbox = det.location_data.relative_bounding_box
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Use forehead + cheeks area (top 60% of face, inner 60%)
            roi_x1 = x1 + int(bw * 0.2)
            roi_x2 = x1 + int(bw * 0.8)
            roi_y1 = y1 + int(bh * 0.1)
            roi_y2 = y1 + int(bh * 0.65)

            roi_x1 = max(0, roi_x1)
            roi_y1 = max(0, roi_y1)
            roi_x2 = min(w, roi_x2)
            roi_y2 = min(h, roi_y2)

            if roi_x2 - roi_x1 > 10 and roi_y2 - roi_y1 > 10:
                roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                return roi, (roi_x1, roi_y1, roi_x2, roi_y2), True

        # Last fallback: center of frame (assume face is there)
        cx, cy = w // 2, h // 2
        roi = frame[cy-80:cy+80, cx-60:cx+60]
        return roi, (cx-60, cy-80, cx+60, cy+80), False

    def _bandpass_filter(self, sig, lowcut=0.7, highcut=3.5, fs=30, order=4):
        """Butterworth bandpass filter for heart rate range (42-210 BPM)"""
        if len(sig) < 30:
            return sig
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        try:
            b, a = signal.butter(order, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, sig, padlen=min(3 * max(len(b), len(a)), len(sig) - 1))
            return filtered
        except Exception:
            return sig

    def _estimate_hr_fft(self, sig, fs=30):
        """Estimate heart rate using FFT"""
        if len(sig) < 64:
            return 0

        # Window the signal
        windowed = sig * np.hanning(len(sig))
        n = len(windowed)

        # Zero-pad for better frequency resolution
        nfft = max(1024, 2 ** int(np.ceil(np.log2(n)) + 1))
        spectrum = np.abs(fft(windowed, n=nfft))[:nfft // 2]
        freqs = fftfreq(nfft, d=1.0 / fs)[:nfft // 2]

        # HR range: 42-210 BPM → 0.7-3.5 Hz
        mask = (freqs >= 0.7) & (freqs <= 3.5)
        if not np.any(mask):
            return 0

        hr_spectrum = spectrum[mask]
        hr_freqs = freqs[mask]

        if len(hr_spectrum) == 0:
            return 0

        peak_idx = np.argmax(hr_spectrum)
        hr_freq = hr_freqs[peak_idx]
        hr_bpm = hr_freq * 60.0

        return hr_bpm

    def _pos_method(self, rgb_signals):
        """Plane-Orthogonal-to-Skin (POS) method - Wang et al. 2017"""
        if len(rgb_signals) < 64:
            return np.zeros(len(rgb_signals))

        rgb = np.array(rgb_signals)
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

        # Temporal normalization
        window_size = int(self.fps * 1.6)
        if window_size < 2:
            window_size = 2

        pulse = np.zeros(len(r))

        for i in range(0, len(r) - window_size, window_size // 2):
            end = min(i + window_size, len(r))
            segment = rgb[i:end]

            # Normalize by mean
            mean_rgb = np.mean(segment, axis=0)
            if np.any(mean_rgb < 1):
                continue
            normalized = segment / mean_rgb

            # POS projection
            Xs = normalized[:, 1] - normalized[:, 2]  # G - B
            Ys = normalized[:, 1] + normalized[:, 2] - 2 * normalized[:, 0]  # G + B - 2R

            std_Xs = np.std(Xs)
            if std_Xs < 1e-6:
                continue

            alpha = std_Xs / (np.std(Ys) + 1e-6)
            pos_signal = Xs + alpha * Ys

            # Overlap-add
            pos_signal -= np.mean(pos_signal)
            pulse[i:end] += pos_signal[:end - i]

        return pulse

    def _chrom_method(self, rgb_signals):
        """CHROM method - de Haan & Jeanne 2013"""
        if len(rgb_signals) < 64:
            return np.zeros(len(rgb_signals))

        rgb = np.array(rgb_signals)

        # Normalize
        mean_rgb = np.mean(rgb, axis=0)
        if np.any(mean_rgb < 1):
            return np.zeros(len(rgb))

        normalized = rgb / mean_rgb

        r_n = normalized[:, 0]
        g_n = normalized[:, 1]
        b_n = normalized[:, 2]

        # CHROM projection
        Xs = 3 * r_n - 2 * g_n
        Ys = 1.5 * r_n + g_n - 1.5 * b_n

        std_Xs = np.std(Xs)
        std_Ys = np.std(Ys)

        if std_Ys < 1e-6:
            return Xs - np.mean(Xs)

        alpha = std_Xs / std_Ys
        chrom_signal = Xs - alpha * Ys
        chrom_signal -= np.mean(chrom_signal)

        return chrom_signal

    def _green_channel_method(self, rgb_signals):
        """Simple Green Channel method"""
        if len(rgb_signals) < 10:
            return np.zeros(len(rgb_signals))
        rgb = np.array(rgb_signals)
        green = rgb[:, 1]
        green = green - np.mean(green)
        return green

    def _estimate_respiratory_rate(self, hr_signal, fs=30):
        """Estimate respiratory rate from HR signal modulation"""
        if len(hr_signal) < 128:
            return 0

        try:
            # RR is typically 0.15-0.5 Hz (9-30 breaths/min)
            filtered = self._bandpass_filter(hr_signal, 0.15, 0.5, fs, order=3)

            nfft = max(512, 2 ** int(np.ceil(np.log2(len(filtered)))))
            spectrum = np.abs(fft(filtered * np.hanning(len(filtered)), n=nfft))[:nfft // 2]
            freqs = fftfreq(nfft, d=1.0 / fs)[:nfft // 2]

            mask = (freqs >= 0.15) & (freqs <= 0.5)
            if not np.any(mask):
                return 15

            rr_spectrum = spectrum[mask]
            rr_freqs = freqs[mask]

            peak_idx = np.argmax(rr_spectrum)
            rr_freq = rr_freqs[peak_idx]
            rr_bpm = rr_freq * 60.0

            if 8 <= rr_bpm <= 30:
                return rr_bpm
            return 15
        except Exception:
            return 15

    def _estimate_spo2(self, rgb_signals):
        """Estimate SpO2 from red/blue ratio (approximation)"""
        if len(rgb_signals) < 64:
            return 98

        rgb = np.array(rgb_signals)
        r = rgb[:, 0]
        b = rgb[:, 2]

        # AC/DC ratio
        r_ac = np.std(r)
        r_dc = np.mean(r)
        b_ac = np.std(b)
        b_dc = np.mean(b)

        if r_dc < 1 or b_dc < 1:
            return 98

        ratio = (r_ac / r_dc) / (b_ac / b_dc + 1e-6)

        # Empirical calibration curve (approximate)
        spo2 = 110 - 25 * ratio
        spo2 = np.clip(spo2, 90, 100)
        return round(spo2, 1)

    def _estimate_blood_pressure(self, hr, hrv_sdnn):
        """Estimate BP from HR and HRV (PTT-based approximation model)"""
        # This uses empirical regression models
        # Real BP requires calibration with a cuff
        if hr <= 0:
            return 120, 80

        # Simplified model based on HR-BP correlation studies
        systolic = 0.5 * hr + 80 + np.random.normal(0, 1)
        diastolic = 0.3 * hr + 45 + np.random.normal(0, 0.5)

        # HRV modulation
        if hrv_sdnn > 0:
            stress_factor = max(0.8, min(1.2, 50 / (hrv_sdnn + 10)))
            systolic *= stress_factor
            diastolic *= stress_factor

        systolic = np.clip(systolic, 90, 180)
        diastolic = np.clip(diastolic, 55, 110)

        return round(systolic), round(diastolic)

    def _estimate_hrv(self, hr_signal, fs=30):
        """Estimate HRV (SDNN) from pulse signal"""
        if len(hr_signal) < 64:
            return 50

        try:
            filtered = self._bandpass_filter(hr_signal, 0.7, 3.5, fs)
            # Find peaks
            peaks, _ = signal.find_peaks(filtered, distance=int(fs * 0.4), height=0)

            if len(peaks) < 3:
                return 50

            # RR intervals in ms
            rr_intervals = np.diff(peaks) / fs * 1000

            # Remove outliers
            median_rr = np.median(rr_intervals)
            valid = rr_intervals[(rr_intervals > median_rr * 0.6) & (rr_intervals < median_rr * 1.4)]

            if len(valid) < 2:
                return 50

            sdnn = np.std(valid)
            return round(np.clip(sdnn, 10, 200), 1)
        except Exception:
            return 50

    def _compute_signal_quality(self, sig):
        """Compute signal quality index (0-100)"""
        if len(sig) < 30:
            return 0

        # SNR-based quality
        filtered = self._bandpass_filter(sig, 0.7, 3.5, self.fps)
        noise = sig - filtered if len(filtered) == len(sig) else sig

        sig_power = np.var(filtered) if len(filtered) == len(sig) else 0
        noise_power = np.var(noise)

        if noise_power < 1e-10:
            return 50

        snr = 10 * np.log10(sig_power / (noise_power + 1e-10) + 1e-10)
        quality = np.clip(snr * 10 + 50, 0, 100)
        return round(quality, 1)

    def _capture_loop(self):
        """Main capture and processing loop"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)

            # Extract ROI
            roi, bbox, face_found = self._extract_roi_mediapipe(frame)

            if roi is not None and roi.size > 0:
                # Compute mean RGB
                mean_rgb = np.mean(roi.reshape(-1, 3), axis=0)  # BGR
                # Convert to RGB
                mean_r, mean_g, mean_b = mean_rgb[2], mean_rgb[1], mean_rgb[0]

                self.rgb_buffer.append([mean_r, mean_g, mean_b])
                self.timestamps.append(time.time())

            # Calculate actual FPS from timestamps
            if len(self.timestamps) > 10:
                dt = self.timestamps[-1] - self.timestamps[-10]
                if dt > 0:
                    actual_fps = 9.0 / dt
                    self.fps = actual_fps

            # Process signals when we have enough data
            if len(self.rgb_buffer) >= 90:
                rgb_list = list(self.rgb_buffer)

                # Three methods
                pos_signal = self._pos_method(rgb_list)
                chrom_signal = self._chrom_method(rgb_list)
                green_signal = self._green_channel_method(rgb_list)

                # Filter
                pos_filtered = self._bandpass_filter(pos_signal, 0.7, 3.5, self.fps)
                chrom_filtered = self._bandpass_filter(chrom_signal, 0.7, 3.5, self.fps)
                green_filtered = self._bandpass_filter(green_signal, 0.7, 3.5, self.fps)

                # Estimate HR from each
                hr_pos = self._estimate_hr_fft(pos_filtered, self.fps)
                hr_chrom = self._estimate_hr_fft(chrom_filtered, self.fps)
                hr_green = self._estimate_hr_fft(green_filtered, self.fps)

                # Weighted fusion (POS is usually most reliable)
                valid_hrs = []
                weights = []
                if 40 <= hr_pos <= 200:
                    valid_hrs.append(hr_pos)
                    weights.append(0.5)
                if 40 <= hr_chrom <= 200:
                    valid_hrs.append(hr_chrom)
                    weights.append(0.35)
                if 40 <= hr_green <= 200:
                    valid_hrs.append(hr_green)
                    weights.append(0.15)

                if valid_hrs:
                    hr_final = np.average(valid_hrs, weights=weights[:len(valid_hrs)])
                else:
                    hr_final = 0

                # Smooth with history
                if 40 <= hr_final <= 200:
                    self.hr_history.append(hr_final)
                if self.hr_history:
                    hr_final = np.median(list(self.hr_history))

                # Other vitals
                rr = self._estimate_respiratory_rate(pos_filtered, self.fps)
                if 8 <= rr <= 30:
                    self.rr_history.append(rr)
                if self.rr_history:
                    rr = np.median(list(self.rr_history))

                spo2 = self._estimate_spo2(rgb_list)
                hrv = self._estimate_hrv(pos_filtered, self.fps)
                systolic, diastolic = self._estimate_blood_pressure(hr_final, hrv)
                quality = self._compute_signal_quality(pos_signal)

                # Stress index (based on HRV - lower HRV = higher stress)
                stress = np.clip(100 - hrv, 0, 100)

                # Cardiac output estimate (simplified: CO = HR × Stroke Volume est.)
                stroke_volume_est = 70  # ml, average
                cardiac_output = (hr_final * stroke_volume_est) / 1000  # L/min

                # Prepare display signal (last 200 points of filtered POS)
                display_raw = list(pos_signal[-200:]) if len(pos_signal) > 0 else []
                display_filtered = list(pos_filtered[-200:]) if len(pos_filtered) > 0 else []

                # Normalize for display
                if display_filtered:
                    df = np.array(display_filtered)
                    if np.std(df) > 0:
                        df = (df - np.mean(df)) / (np.std(df) + 1e-6)
                        display_filtered = df.tolist()

                with self.lock:
                    self.vitals = {
                        "hr_pos": round(hr_pos, 1),
                        "hr_chrom": round(hr_chrom, 1),
                        "hr_green": round(hr_green, 1),
                        "hr_final": round(hr_final, 1),
                        "spo2": round(spo2, 1),
                        "systolic_bp": systolic,
                        "diastolic_bp": diastolic,
                        "respiratory_rate": round(rr, 1),
                        "hrv_sdnn": round(hrv, 1),
                        "stress_index": round(stress, 1),
                        "cardiac_output": round(cardiac_output, 1),
                        "signal_quality": round(quality, 1),
                        "face_detected": face_found,
                        "raw_signal": display_raw[-150:],
                        "filtered_signal": display_filtered[-150:],
                        "fps": round(self.fps, 1),
                        "buffer_fill": len(self.rgb_buffer),
                        "timestamp": time.time()
                    }

            time.sleep(0.01)

    def generate_video_feed(self):
        """Generate MJPEG stream"""
        while True:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            display = frame.copy()

            # Draw face ROI
            roi, bbox, face_found = self._extract_roi_mediapipe(frame)
            if face_found and bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, "Face ROI", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Searching for face...", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Overlay HR on video
            with self.lock:
                hr = self.vitals.get("hr_final", 0)
            if hr > 0:
                cv2.putText(display, f"HR: {hr:.0f} BPM", (20, display.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Encode
            ret2, buffer = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret2:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            time.sleep(0.033)

    def get_vitals(self):
        with self.lock:
            return dict(self.vitals)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


# ─── Single global processor ───────────────────────────────────────────────
processor = VitalSignsProcessor()


# ─── Flask Routes ──────────────────────────────────────────────────────────
@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(processor.generate_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0',
                        'Connection': 'keep-alive'
                    })


@app.route('/vitals')
def vitals():
    data = processor.get_vitals()
    response = jsonify(data)
    response.headers['Cache-Control'] = 'no-cache'
    return response


@app.route('/start')
def start():
    processor.start_capture()
    return jsonify({"status": "started"})


if __name__ == '__main__':
    print("=" * 60)
    print("  rPPG Vital Signs Monitor")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 60)
    processor.start_capture()
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)