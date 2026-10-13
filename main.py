import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk,)


BIT_DEPTHS = {
    'PCM_U8': 8,
    'PCM_16': 16,
    'PCM_24': 24,
    'PCM_32': 32,
    'FLOAT': 32,
    'DOUBLE': 64,
}

# Samples have shape (sample frames, channels).
samples = None
rate = None
info = None
source_path = None
load_time = 0


def time_units(seconds):
    if 0 < seconds < 1e-6:
        return 1e9, 'ns'
    if 0 < seconds < 1e-3:
        return 1e6, 'µs'
    if 0 < seconds < 1:
        return 1e3, 'ms'
    return 1, 's'


def format_time(seconds):
    scale, unit = time_units(seconds)
    return f'{seconds * scale:.6g} {unit}'


def calculate_zcr(values, sample_rate, frame_ms=20):
    """Return complete-frame start indices and zero-crossing rates."""
    frame_size = round(sample_rate * frame_ms / 1000)
    hop_size = max(1, round(frame_size / 2))

    if frame_size < 2:
        raise ValueError('Each frame must contain at least two samples.')

    if len(values) < frame_size:
        raise ValueError(f'Select at least {frame_ms} ms of audio.')

    starts = np.arange(
        0,
        len(values) - frame_size + 1,
        hop_size,
    )
    zcr = np.empty(len(starts))

    for i, start in enumerate(starts):
        frame = values[start:start + frame_size]

        # In the assignment, zero belongs to the positive sign.
        positive = frame >= 0
        crossings = np.count_nonzero(
            positive[1:] != positive[:-1]
        )

        # Equivalent to the assignment's sign-difference formula.
        zcr[i] = crossings / frame_size

    return starts, zcr


def open_file():
    global samples, rate, info, source_path, load_time

    sound_folder = Path(__file__).parent / 'Garsai#1'
    initial_folder = (
        sound_folder if sound_folder.is_dir()
        else Path(__file__).parent
    )

    path = filedialog.askopenfilename(
        parent=window,
        title='Open WAV recording',
        filetypes=[('WAV recordings', '*.wav')],
        initialdir=str(initial_folder),
    )

    if not path:
        return

    try:
        started = perf_counter()
        new_info = sf.info(path)

        if (
            new_info.format not in ('WAV', 'WAVEX', 'RF64')
            or new_info.subtype not in BIT_DEPTHS
        ):
            raise ValueError(
                'Choose an integer PCM or floating-point WAV recording.'
            )

        new_samples, new_rate = sf.read(
            path,
            dtype='float64',
            always_2d=True,
        )

        if not len(new_samples) or not np.isfinite(new_samples).all():
            raise ValueError(
                'The recording is empty or contains non-finite samples.'
            )

        samples = new_samples
        rate = new_rate
        info = new_info
        source_path = Path(path)
        load_time = perf_counter() - started

        start_field.set('0')
        end_field.set(str(len(samples) / rate))

        draw_waveforms()

    except (OSError, ValueError, RuntimeError, MemoryError) as error:
        messagebox.showerror(
            'Could not open audio',
            str(error),
            parent=window,
        )


def get_selection():
    """Convert the selected start/end times to sample indices."""
    if samples is None:
        raise ValueError('Open a WAV file first.')

    try:
        start = float(start_field.get())
        end = float(end_field.get())
    except ValueError:
        raise ValueError('Enter numeric start and end times.') from None

    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError('Enter finite numeric times.')

    total_duration = len(samples) / rate

    if start < 0 or end > total_duration + 1e-9:
        raise ValueError(
            f'Select an interval inside '
            f'0–{total_duration:.6f} seconds.'
        )

    if end <= start:
        raise ValueError('End time must be greater than start time.')

    first = round(start * rate)
    last = min(len(samples), round(end * rate))

    if first >= last:
        raise ValueError(
            'The interval must contain at least one sample frame.'
        )

    return first, last


def draw_waveforms(highlight=False, segment_only=False):
    """Draw the full recording or selection, with one plot per channel."""
    try:
        if samples is None:
            raise ValueError('Open a WAV file first.')

        first, last = 0, len(samples)

        if highlight or segment_only:
            selected_first, selected_last = get_selection()

            if segment_only:
                first, last = selected_first, selected_last

        started = perf_counter()
        duration = (last - first) / rate
        scale, unit = time_units(duration)

        if rate >= 1_000_000:
            rate_label = f'{rate / 1_000_000:g} MHz'
        elif rate >= 1000:
            rate_label = f'{rate / 1000:g} kHz'
        else:
            rate_label = f'{rate:g} Hz'

        bits = BIT_DEPTHS[info.subtype]
        is_pcm = info.subtype.startswith('PCM')

        # SoundFile normalizes PCM samples.
        amplitude_scale = 2 ** (bits - 1) if is_pcm else 1

        figure.clear()
        axes = figure.subplots(
            info.channels,
            1,
            sharex=True,
            squeeze=False,
        )[:, 0]

        figure.suptitle(
            f'{source_path.name}\n'
            f'{info.channels} channels | '
            f'{rate_label} | {bits}-bit {info.subtype}',
            fontsize=10,
        )

        figure.supylabel(
            'Amplitude (PCM counts)'
            if is_pcm
            else 'Amplitude (float)'
        )

        # Long views show a min/max envelope to preserve peaks.
        block_size = max(1, math.ceil((last - first) / 2500))
        indices = np.arange(0, last - first, block_size)
        times = (first + indices) / rate * scale

        for channel, axis in enumerate(axes):
            values = samples[first:last, channel]
            color = f'C{channel % 10}'

            if block_size == 1:
                axis.plot(
                    times,
                    values * amplitude_scale,
                    color=color,
                    linewidth=0.8,
                )
            else:
                low = (
                    np.minimum.reduceat(values, indices)
                    * amplitude_scale
                )
                high = (
                    np.maximum.reduceat(values, indices)
                    * amplitude_scale
                )

                axis.fill_between(
                    times, low, high,
                    color=color,
                    alpha=0.5,
                )
                axis.vlines(
                    times, low, high,
                    color=color,
                    linewidth=0.5,
                )

            axis.set_ylabel(f'Ch {channel + 1}')
            axis.ticklabel_format(
                axis='both',
                style='plain',
                useOffset=False,
            )
            axis.set_xlim(
                first / rate * scale,
                last / rate * scale,
            )
            axis.grid(alpha=0.25)

            if highlight:
                axis.axvspan(
                    selected_first / rate * scale,
                    selected_last / rate * scale,
                    color='orange',
                    alpha=0.3,
                )

        axes[-1].set_xlabel(f'Time ({unit})')

        # Clear navigation history when replacing the plots.
        toolbar.update()
        canvas.draw()

        status.set(
            f'Duration: {format_time(len(samples) / rate)} | '
            f'Sample array: {samples.nbytes / 1024**2:.1f} MiB | '
            f'Load: {format_time(load_time)} | '
            f'Plot: {format_time(perf_counter() - started)}'
        )

    except (ValueError, MemoryError) as error:
        messagebox.showerror(
            'Could not display audio',
            str(error),
            parent=window,
        )


def analyze_zcr():
    try:
        first, last = get_selection()
        frame_ms = 20

        # Analyze the original samples separately for each channel.
        results = [
            calculate_zcr(
                samples[first:last, channel],
                rate,
                frame_ms,
            )
            for channel in range(info.channels)
        ]

        draw_waveforms(highlight=True)

        result_window = tk.Toplevel(window)
        result_window.title('Zero-crossing analysis')
        result_window.geometry('1000x600')

        result_figure = Figure(
            figsize=(10, 6),
            layout='constrained',
        )
        axis = result_figure.subplots()

        result_figure.suptitle(
            f'{source_path.name}\n'
            f'Frames: {frame_ms} ms | Overlap: approximately 50%\n'
            f'Interval: {first / rate:.3f}–{last / rate:.3f} s',
            fontsize=10,
        )

        frame_size = round(rate * frame_ms / 1000)

        for channel, (starts, zcr) in enumerate(results):
            # Plot values at the centers of their analysis frames.
            times = (
                first + starts + (frame_size - 1) / 2
            ) / rate

            axis.plot(
                times,
                zcr,
                color=f'C{channel % 10}',
                linewidth=1,
                marker='o',
                markersize=3,
                label=f'Ch {channel + 1}',
            )

        axis.set_xlabel('Time (s)')
        axis.set_ylabel('Zero-crossing rate')
        axis.set_ylim(bottom=0)
        axis.set_xlim(first / rate, last / rate)
        axis.ticklabel_format(
            axis='x',
            style='plain',
            useOffset=False,
        )
        axis.grid(alpha=0.25)
        axis.legend()

        result_canvas = FigureCanvasTkAgg(
            result_figure,
            master=result_window,
        )

        result_toolbar = NavigationToolbar2Tk(
            result_canvas,
            result_window,
            pack_toolbar=False,
        )
        result_toolbar.update()
        result_toolbar.pack(side='bottom', fill='x')

        result_canvas.get_tk_widget().pack(
            side='top',
            fill='both',
            expand=True,
        )
        result_canvas.draw()

        # Keep references while this window is open.
        result_window.result_canvas = result_canvas
        result_window.result_toolbar = result_toolbar

    except (ValueError, MemoryError) as error:
        messagebox.showerror(
            'Could not analyze audio',
            str(error),
            parent=window,
        )


def save_selection():
    """Save selected samples with their original rate and subtype."""
    try:
        first, last = get_selection()

        path = filedialog.asksaveasfilename(
            parent=window,
            defaultextension='.wav',
            filetypes=[('WAV recording', '*.wav')],
            initialfile=source_path.stem + '_segment.wav',
        )

        if not path:
            return

        if Path(path).resolve() == source_path.resolve():
            raise ValueError(
                'Choose a new filename to preserve the original recording.'
            )

        sf.write(
            path,
            samples[first:last],
            rate,
            subtype=info.subtype,
            format=info.format,
        )

        messagebox.showinfo(
            'Saved',
            f'Saved {format_time((last - first) / rate)} to:\n{path}',
            parent=window,
        )

    except (OSError, ValueError, RuntimeError) as error:
        messagebox.showerror(
            'Could not save segment',
            str(error),
            parent=window,
        )


if __name__ == '__main__':
    window = tk.Tk()
    window.title('Audio Lab 2 — Zero-crossing analysis')
    window.geometry('1100x850')

    # File loading and interval controls.
    controls = ttk.Frame(window, padding=10)
    controls.pack(fill='x')

    ttk.Button(
        controls,
        text='Open WAV',
        command=open_file,
    ).pack(side='left', padx=5)

    start_field = tk.StringVar(value='0')
    end_field = tk.StringVar(value='1')

    for label, variable in [
        ('Start (s)', start_field),
        ('End (s)', end_field),
    ]:
        ttk.Label(
            controls,
            text=label,
        ).pack(side='left', padx=5)

        ttk.Entry(
            controls,
            textvariable=variable,
            width=12,
        ).pack(side='left')

    # Separate row keeps the buttons from getting crowded.
    actions = ttk.Frame(window, padding=(10, 0, 10, 8))
    actions.pack(fill='x')

    for label, command in [
        ('Highlight', lambda: draw_waveforms(highlight=True)),
        ('View segment', lambda: draw_waveforms(segment_only=True)),
        ('Full recording', draw_waveforms),
        ('Save segment', save_selection),
        ('Analyze ZCR', analyze_zcr),
    ]:
        ttk.Button(
            actions,
            text=label,
            command=command,
        ).pack(side='left', padx=4)

    status = tk.StringVar(
        value='Open a WAV file to begin. Selection fields use seconds.'
    )
    ttk.Label(
        window,
        textvariable=status,
        padding=8,
        wraplength=1050,
    ).pack(side='bottom', fill='x')

    figure = Figure(figsize=(10, 7), layout='constrained')
    canvas = FigureCanvasTkAgg(figure, master=window)

    toolbar = NavigationToolbar2Tk(
        canvas,
        window,
        pack_toolbar=False,
    )
    toolbar.update()
    toolbar.pack(side='bottom', fill='x')

    canvas.get_tk_widget().pack(
        side='top',
        fill='both',
        expand=True,
    )

    window.mainloop()