#!/usr/bin/env python3
"""Live microphone scope: pick an input device and stream time- and frequency-domain plots."""

import sys
if '--help' in sys.argv or '-h' in sys.argv:
    print("""usage: micscope.py [options]

Live audio scope with input-device selection, waveform and spectrum views.

options:
  -h, --help        show this help message and exit
  --list            list input devices and exit
  --device ID       preselect input device by index
  --rate HZ         sample rate (default: device default)
  --fft N           FFT size, power of two (default: 4096)
  --window MS       time-domain window length in ms (default: 50)
  --outdir DIR      folder for recordings (default: ./recordings)
  --freqs F1,F2     tone frequencies (Hz): played while recording and read out
                    on the spectrum (default: 2377,2900)
  --tone-levels L1,L2  level of each played tone in dBFS, one per --freqs entry
                    (default: -45,-55, i.e. f1 is 10 dB above f2)
""")
    sys.exit(0)

import argparse
import threading
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# seaborn 'deep' palette
DEEP = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
        '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']
FFT_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
REC_RATE = 48000
DEFAULT_FREQS = [2377.0, 2900.0]
DEFAULT_TONE_LEVELS = [-45.0, -55.0]  # dBFS per tone; f1 sits 10 dB above f2
PEAK_SEARCH_BINS = 2  # read the max within +/- this many FFT bins of each target
COMMON_RATES = [8000, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 192000]


def output_devices() -> list:
    """Return (index, label) tuples for every device with output channels."""
    hostapis = sd.query_hostapis()
    return [(i, f"[{i}] {d['name']} ({hostapis[d['hostapi']]['name']}, {d['max_output_channels']} ch)")
            for i, d in enumerate(sd.query_devices()) if d['max_output_channels'] > 0]


def input_devices() -> list:
    """Return (index, label, info) tuples for every device with input channels."""
    hostapis = sd.query_hostapis()
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0:
            api = hostapis[d['hostapi']]['name']
            out.append((i, f"[{i}] {d['name']} ({api}, {d['max_input_channels']} ch)", d))
    return out


class RingBuffer:
    """Thread-safe mono ring buffer fed from the audio callback."""

    def __init__(self, size: int):
        self.data = np.zeros(size, dtype=np.float32)
        self.pos = 0
        self.lock = threading.Lock()

    def write(self, x: np.ndarray) -> None:
        n = len(x)
        size = len(self.data)
        if n >= size:
            x = x[-size:]
            n = size
        with self.lock:
            end = self.pos + n
            if end <= size:
                self.data[self.pos:end] = x
            else:
                k = size - self.pos
                self.data[self.pos:] = x[:k]
                self.data[:n - k] = x[k:]
            self.pos = end % size

    def latest(self, n: int) -> np.ndarray:
        """Return the most recent n samples in chronological order."""
        with self.lock:
            return np.roll(self.data, -self.pos)[-n:].copy()


class TonePlayer:
    """Plays a phase-continuous sum of sine tones on an output device."""

    FADE_S = 0.01  # fade-in to avoid a click at the start

    def __init__(self, freqs: list, levels_dbfs: list):
        self.freqs = np.asarray(freqs, dtype=np.float64)
        self.amps = 10 ** (np.asarray(levels_dbfs, dtype=np.float64) / 20)
        self.stream: Optional[sd.OutputStream] = None
        self.n = 0

    def start(self, device: int) -> None:
        info = sd.query_devices(device)
        try:
            sd.check_output_settings(device=device, samplerate=REC_RATE)
            rate = REC_RATE
        except Exception:
            rate = int(info['default_samplerate'])
        channels = min(info['max_output_channels'], 2)
        fade_len = int(rate * self.FADE_S)
        self.n = 0

        def callback(outdata, frames, time_info, status):
            idx = np.arange(self.n, self.n + frames)
            t = idx / rate
            x = (self.amps[:, None] * np.sin(2 * np.pi * self.freqs[:, None] * t)).sum(axis=0)
            x *= np.minimum(idx / fade_len, 1.0)
            outdata[:] = x[:, None].astype(np.float32)
            self.n += frames

        self.stream = sd.OutputStream(device=device, samplerate=rate, channels=channels,
                                      dtype='float32', latency=0.1, callback=callback)
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


class MicScope(QtWidgets.QMainWindow):
    """Main window: device controls on top, waveform and spectrum plots below."""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.setWindowTitle('Mic Scope')
        self.resize(1100, 750)
        self.stream: Optional[sd.InputStream] = None
        self.buffer: Optional[RingBuffer] = None
        self.rate = 48000
        self.status_flags = ''
        self.outdir = Path(args.outdir)
        self.target_freqs = args.freqs
        self.tones = TonePlayer(args.freqs, args.tone_levels)
        self.rec_chunks: Optional[list] = None
        self.rec_lock = threading.Lock()

        self._build_controls(args)
        self._build_plots()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(33)

        # Platform Close key: Cmd+W on macOS
        close_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence.Close, self)
        close_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
        close_shortcut.activated.connect(self.close)

        record_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self)
        record_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
        record_shortcut.activated.connect(self._toggle_record)

        self._start()

    # ---------- UI ----------
    def _build_controls(self, args: argparse.Namespace) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        bar = QtWidgets.QHBoxLayout()
        layout.addLayout(bar)
        self.main_layout = layout

        self.device_box = QtWidgets.QComboBox()
        self.device_box.setMinimumWidth(320)
        self.out_box = QtWidgets.QComboBox()
        self.out_box.setMinimumWidth(220)
        self.channel_box = QtWidgets.QComboBox()
        self.rate_box = QtWidgets.QComboBox()
        self.fft_box = QtWidgets.QComboBox()
        for n in FFT_SIZES:
            self.fft_box.addItem(str(n), n)
        self.fft_box.setCurrentIndex(FFT_SIZES.index(args.fft) if args.fft in FFT_SIZES else 3)

        self.window_spin = QtWidgets.QSpinBox()
        self.window_spin.setRange(5, 2000)
        self.window_spin.setSuffix(' ms')
        self.window_spin.setValue(args.window)

        self.logx_check = QtWidgets.QCheckBox('Log freq')
        self.logx_check.setChecked(True)
        self.hold_check = QtWidgets.QCheckBox('Peak hold')
        self.refresh_btn = QtWidgets.QPushButton('Rescan')
        self.start_btn = QtWidgets.QPushButton('Start')
        self.rec_btn = QtWidgets.QPushButton('Record')

        for label, w in [('Device', self.device_box), ('Channel', self.channel_box), ('Out', self.out_box),
                         ('Rate', self.rate_box), ('FFT', self.fft_box),
                         ('Window', self.window_spin)]:
            bar.addWidget(QtWidgets.QLabel(label))
            bar.addWidget(w)
        bar.addWidget(self.logx_check)
        bar.addWidget(self.hold_check)
        bar.addStretch(1)
        bar.addWidget(self.refresh_btn)
        bar.addWidget(self.start_btn)
        bar.addWidget(self.rec_btn)

        # Keep keyboard focus off the controls so Space always reaches the record shortcut
        for w in (self.device_box, self.out_box, self.channel_box, self.rate_box, self.fft_box,
                  self.logx_check, self.hold_check, self.refresh_btn, self.start_btn,
                  self.rec_btn):
            w.setFocusPolicy(QtCore.Qt.NoFocus)
        self.window_spin.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.window_spin.editingFinished.connect(self.window_spin.clearFocus)

        self.status = QtWidgets.QLabel()
        self.statusBar().addWidget(self.status)

        self.device_box.currentIndexChanged.connect(self._device_changed)
        self.refresh_btn.clicked.connect(self._populate_devices)
        self.start_btn.clicked.connect(self._toggle)
        self.rec_btn.clicked.connect(self._toggle_record)
        self.logx_check.toggled.connect(lambda on: self.freq_plot.setLogMode(x=on))
        self.hold_check.toggled.connect(self._reset_hold)
        self.fft_box.currentIndexChanged.connect(self._reset_hold)
        for w in (self.channel_box, self.rate_box, self.fft_box):
            w.currentIndexChanged.connect(self._restart_if_running)

        self._populate_devices(preselect=args.device)
        if args.rate:
            idx = self.rate_box.findData(args.rate)
            if idx < 0:
                self.rate_box.addItem(f'{args.rate} Hz', args.rate)
                idx = self.rate_box.count() - 1
            self.rate_box.setCurrentIndex(idx)

    def _build_plots(self) -> None:
        # Antialiasing and thick pens make Qt painting slow enough to hold the GIL
        # past the audio deadline, which silently drops input blocks.
        pg.setConfigOptions(antialias=False)
        plots = pg.GraphicsLayoutWidget()
        self.main_layout.addWidget(plots, 1)

        self.time_plot = plots.addPlot(row=0, col=0, title='Time domain')
        self.time_plot.setLabel('bottom', 'Time', units='s')
        self.time_plot.setLabel('left', 'Amplitude')
        self.time_plot.setYRange(-1, 1)
        self.time_plot.showGrid(x=True, y=True, alpha=0.3)
        self.time_curve = self.time_plot.plot(pen=pg.mkPen(DEEP[0], width=1))

        self.freq_plot = plots.addPlot(row=1, col=0, title='Frequency domain')
        self.freq_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.freq_plot.setLabel('left', 'Magnitude', units='dBFS')
        self.freq_plot.setYRange(-120, 0)
        self.freq_plot.showGrid(x=True, y=True, alpha=0.3)
        self.freq_plot.setLogMode(x=True)
        self.freq_curve = self.freq_plot.plot(pen=pg.mkPen(DEEP[1], width=1))
        self.hold_curve = self.freq_plot.plot(
            pen=pg.mkPen(DEEP[2], width=1, style=QtCore.Qt.DashLine))
        for plot in (self.time_plot, self.freq_plot):
            plot.setClipToView(True)
            plot.setDownsampling(auto=True, mode='peak')
        self.targets = []
        label_font = QtGui.QFont()
        label_font.setPointSize(14)
        label_font.setBold(True)
        for i, ft in enumerate(self.target_freqs):
            color = DEEP[4 + i % 6]
            line = pg.InfiniteLine(angle=90, movable=False,
                                   pen=pg.mkPen(color, width=1, style=QtCore.Qt.DotLine))
            marker = self.freq_plot.plot([], [], pen=None, symbol='o', symbolSize=9,
                                         symbolBrush=color, symbolPen='w')
            # Alternate labels left/right of their line so neighbouring targets don't collide
            anchor = (1.05, 0) if i % 2 == 0 else (-0.05, 0)
            text = pg.TextItem(color=color, anchor=anchor, fill=pg.mkBrush(0, 0, 0, 170))
            text.setFont(label_font)
            self.freq_plot.addItem(line, ignoreBounds=True)
            self.freq_plot.addItem(text, ignoreBounds=True)
            self.targets.append((ft, line, marker, text))
        self.hold: Optional[np.ndarray] = None
        # Lock the dB axis; set last because adding items/log mode can re-enable autorange
        self.freq_plot.enableAutoRange(y=False)
        self.freq_plot.setYRange(-120, 0, padding=0)

    # ---------- device handling ----------
    def _populate_devices(self, preselect: Optional[int] = None) -> None:
        if preselect is None or isinstance(preselect, bool):
            preselect = self.device_box.currentData()
        was_running = self.stream is not None
        self._stop()
        sd._terminate()
        sd._initialize()
        self.device_box.blockSignals(True)
        self.device_box.clear()
        devs = input_devices()
        for idx, label, _ in devs:
            self.device_box.addItem(label, idx)
        try:
            default_in = sd.default.device[0]
        except Exception:
            default_in = None
        target = preselect if preselect is not None else default_in
        pos = self.device_box.findData(target)
        self.device_box.setCurrentIndex(max(pos, 0))
        self.device_box.blockSignals(False)

        prev_out = self.out_box.currentData()
        self.out_box.clear()
        for idx, label in output_devices():
            self.out_box.addItem(label, idx)
        try:
            default_out = sd.default.device[1]
        except Exception:
            default_out = None
        pos = self.out_box.findData(prev_out if prev_out is not None else default_out)
        self.out_box.setCurrentIndex(max(pos, 0))
        if not devs:
            self.status.setText('No input devices found')
            return
        self._device_changed()
        if was_running:
            self._start()

    def _device_changed(self) -> None:
        dev = self.device_box.currentData()
        if dev is None:
            return
        info = sd.query_devices(dev)
        self.channel_box.blockSignals(True)
        self.channel_box.clear()
        for c in range(info['max_input_channels']):
            self.channel_box.addItem(f'Ch {c + 1}', c)
        if info['max_input_channels'] > 1:
            self.channel_box.addItem('Mix (avg)', -1)
        self.channel_box.blockSignals(False)

        self.rate_box.blockSignals(True)
        self.rate_box.clear()
        default_rate = int(info['default_samplerate'])
        rates = []
        for r in sorted(set(COMMON_RATES + [default_rate])):
            try:
                sd.check_input_settings(device=dev, samplerate=r, channels=1)
                rates.append(r)
            except Exception:
                pass
        for r in rates or [default_rate]:
            self.rate_box.addItem(f'{r} Hz', r)
        preferred = REC_RATE if self.rate_box.findData(REC_RATE) >= 0 else default_rate
        self.rate_box.setCurrentIndex(max(self.rate_box.findData(preferred), 0))
        self.rate_box.blockSignals(False)
        self._restart_if_running()

    # ---------- streaming ----------
    def _toggle(self) -> None:
        if self.stream is None:
            self._start()
        else:
            self._stop()

    def _restart_if_running(self, *_) -> None:
        if self.stream is not None:
            self._stop()
            self._start()

    def _start(self) -> None:
        dev = self.device_box.currentData()
        if dev is None:
            return
        info = sd.query_devices(dev)
        channel = self.channel_box.currentData()
        n_ch = info['max_input_channels'] if channel == -1 else channel + 1
        self.rate = self.rate_box.currentData() or int(info['default_samplerate'])
        self.buffer = RingBuffer(max(int(self.rate * 2.5), 32768))
        self._reset_hold()

        def callback(indata, frames, time_info, status):
            if status:
                self.status_flags = str(status)
            x = indata.mean(axis=1) if channel == -1 else indata[:, channel]
            self.buffer.write(x)
            with self.rec_lock:
                if self.rec_chunks is not None:
                    self.rec_chunks.append(x.copy())

        try:
            self.stream = sd.InputStream(device=dev, channels=n_ch, samplerate=self.rate,
                                         dtype='float32', latency=0.1,
                                         callback=callback)
            self.stream.start()
        except Exception as e:
            self.stream = None
            self.status.setText(f'Failed to open device: {e}')
            return
        self.start_btn.setText('Stop')
        self.status.setText(f'Streaming {info["name"]} @ {self.rate} Hz')

    def _stop(self) -> None:
        self._stop_record()
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.start_btn.setText('Start')

    # ---------- recording ----------
    def _toggle_record(self) -> None:
        if self.rec_chunks is None:
            self._start_record()
        else:
            self._stop_record()

    def _start_record(self) -> None:
        if self.stream is None:
            self._start()
            if self.stream is None:
                return
        out_dev = self.out_box.currentData()
        if out_dev is not None:
            try:
                self.tones.start(out_dev)
            except Exception as e:
                self.statusBar().showMessage(f'Tone output failed: {e}', 8000)
        with self.rec_lock:
            self.rec_chunks = []
        self.rec_btn.setText('Stop Rec')
        self.rec_btn.setStyleSheet(f'background-color: {DEEP[3]}; color: white;')

    def _stop_record(self) -> None:
        """Stop recording and write the captured audio to a 16-bit mono WAV at REC_RATE."""
        with self.rec_lock:
            chunks, self.rec_chunks = self.rec_chunks, None
        self.tones.stop()
        self.rec_btn.setText('Record')
        self.rec_btn.setStyleSheet('')
        if not chunks:
            return
        audio = np.concatenate(chunks)
        if self.rate != REC_RATE:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(REC_RATE, self.rate)
            audio = resample_poly(audio, REC_RATE // g, self.rate // g)
        pcm = (np.clip(audio, -1, 1) * 32767).astype('<i2')
        self.outdir.mkdir(parents=True, exist_ok=True)
        path = self.outdir / f"rec_{datetime.now():%Y%m%d_%H%M%S}.wav"
        with wave.open(str(path), 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(REC_RATE)
            f.writeframes(pcm.tobytes())
        self.statusBar().showMessage(f'Saved {len(audio) / REC_RATE:.1f} s to {path}', 8000)

    def _reset_hold(self, *_) -> None:
        self.hold = None
        self.hold_curve.clear()

    # ---------- drawing ----------
    def _refresh(self) -> None:
        if self.stream is None or self.buffer is None:
            return
        rate = self.rate
        n_time = max(int(rate * self.window_spin.value() / 1000), 2)
        n_time = min(n_time, len(self.buffer.data))
        y = self.buffer.latest(n_time)
        t = np.arange(n_time) / rate - n_time / rate
        self.time_curve.setData(t, y)

        n_fft = self.fft_box.currentData()
        seg = self.buffer.latest(n_fft)
        win = np.hanning(n_fft).astype(np.float32)
        spec = np.fft.rfft(seg * win)
        # Scale so a full-scale sine reads 0 dBFS
        mag = np.abs(spec) * 2 / win.sum()
        db = 20 * np.log10(np.maximum(mag, 1e-12))
        freqs = np.fft.rfftfreq(n_fft, 1 / rate)
        # Drop DC so log-x axis works
        f, d = freqs[1:], db[1:]
        self.freq_curve.setData(f, d)

        if self.hold_check.isChecked():
            self.hold = d if self.hold is None or len(self.hold) != len(d) else np.maximum(self.hold, d)
            self.hold_curve.setData(f, self.hold)

        k = int(np.argmax(d))

        bin_hz = rate / n_fft
        y_top = self.freq_plot.getViewBox().viewRange()[1][1]
        logx = self.logx_check.isChecked()
        for i, (ft, line, marker, text) in enumerate(self.targets):
            line.setPos(np.log10(ft) if logx else ft)
            if ft >= rate / 2:
                marker.setData([], [])
                text.setText(f'f{i + 1} {ft:g} Hz: above Nyquist')
                text.setPos(np.log10(ft) if logx else ft, y_top)
                continue
            band = np.abs(f - ft) <= PEAK_SEARCH_BINS * bin_hz
            j = np.flatnonzero(band)[np.argmax(d[band])]
            level = d[j]
            if 0 < j < len(d) - 1:
                # Parabolic fit across neighbouring bins corrects Hann scalloping loss
                a, b, c = d[j - 1], d[j], d[j + 1]
                denom = a - 2 * b + c
                if denom < 0:
                    level = b - (a - c) ** 2 / (8 * denom)
            fx = f[j]
            marker.setData([fx], [level])
            text.setText(f'f{i + 1} {ft:g} Hz\n{level:.1f} dB')
            text.setPos(np.log10(ft) if logx else ft, y_top)

        rms = np.sqrt(np.mean(y ** 2))
        rms_db = 20 * np.log10(max(rms, 1e-12))
        msg = (f'Streaming @ {rate} Hz   RMS {rms_db:.1f} dBFS   peak {np.max(np.abs(y)):.3f}'
               f'   max {f[k]:.1f} Hz {d[k]:.1f} dB')
        if self.rec_chunks is not None:
            with self.rec_lock:
                n_rec = sum(len(c) for c in self.rec_chunks)
            msg += f'   REC {n_rec / rate:.1f} s'
        if self.status_flags:
            msg += f'   [{self.status_flags}]'
            self.status_flags = ''
        self.status.setText(msg)

    def closeEvent(self, event) -> None:
        self._stop()
        super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--rate', type=int, default=None)
    parser.add_argument('--fft', type=int, default=4096)
    parser.add_argument('--window', type=int, default=50)
    parser.add_argument('--outdir', default='recordings')
    parser.add_argument('--freqs', default=','.join(f'{x:g}' for x in DEFAULT_FREQS))
    parser.add_argument('--tone-levels', default=','.join(f'{x:g}' for x in DEFAULT_TONE_LEVELS))
    args = parser.parse_args()

    try:
        args.freqs = [float(x) for x in args.freqs.split(',') if x.strip()]
    except ValueError:
        parser.error('--freqs must be comma-separated numbers, e.g. 2377,2900')
    if not args.freqs or any(x <= 0 for x in args.freqs):
        parser.error('--freqs must contain positive frequencies')

    try:
        args.tone_levels = [float(x) for x in args.tone_levels.split(',') if x.strip()]
    except ValueError:
        parser.error('--tone-levels must be comma-separated numbers, e.g. -12,-22')
    if len(args.tone_levels) != len(args.freqs):
        parser.error(f'--tone-levels needs {len(args.freqs)} values, one per --freqs entry')
    if any(x > 0 for x in args.tone_levels):
        parser.error('--tone-levels must be <= 0 dBFS')
    if sum(10 ** (x / 20) for x in args.tone_levels) > 1:
        print('warning: combined tone levels may clip', file=sys.stderr)

    if args.list:
        for _, label, _ in input_devices():
            print(label)
        return
    if args.fft not in FFT_SIZES:
        parser.error(f'--fft must be one of {FFT_SIZES}')
    if args.device is not None and args.device not in [i for i, _, _ in input_devices()]:
        parser.error(f'--device {args.device} is not an input device (see --list)')

    app = QtWidgets.QApplication(sys.argv)
    win = MicScope(args)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
