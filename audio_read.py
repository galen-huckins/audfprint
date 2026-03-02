# coding=utf-8
"""audio_read reads in a whole audio file with resampling."""

# Equivalent to:
# import librosa
# def audio_read(filename, sr=11025, channels=1):
#    """Read a soundfile, return (d, sr)."""
#    d, sr = librosa.load(filename, sr=sr, mono=(channels == 1))
#    return d, sr

# The code below is adapted from:
# https://github.com/bmcfee/librosa/blob/master/librosa/core/audio.py
# This is its original copyright notice:

# Copyright (c) 2014, Brian McFee, Matt McVicar, Dawen Liang, Colin Raffel, Douglas Repetto, Dan Ellis.
#
# Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
from __future__ import division

import os
import re
import subprocess
import threading
import time

import numpy as np
# For wavread fallback.
import scipy.io.wavfile as wav


try:
    import queue
except ImportError:
    # noinspection PyUnresolvedReferences
    import Queue as queue


# If ffmpeg is unavailable, you can set HAVE_FFMPEG to False which will cause
# soundfile reads to go via scipy.io.wavfile.  However, this means that only
# *.wav files are supported *and* they must already be resampled to the
# system sampling rate (e.g. 11025 Hz).

HAVE_FFMPEG = True

def wavread(filename):
  """Read in audio data from a wav file.  Return d, sr."""
  # Read in wav file.
  samplerate, wave_data = wav.read(filename)
  # Normalize short ints to floats in range [-1..1).
  data = np.asfarray(wave_data) / 32768.0
  return data, samplerate


def audio_read(filename, sr=None, channels=None, mid_cancel=0.0):
    """Read a soundfile, return (d, sr).
    
    Args:
        filename: Path to audio file
        sr: Target sample rate (None = native)
        channels: Number of channels (1 = mono, 2 = stereo, None = native)
        mid_cancel: Amount of center/mid channel to cancel (0.0 = normal mono,
                   1.0 = pure side channel L-R). Values between 0-1 blend.
                   This helps detect music under speech in podcasts.
    """
    if HAVE_FFMPEG:
        return audio_read_ffmpeg(filename, sr, channels, mid_cancel=mid_cancel)
    else:
        data, samplerate = wavread(filename)
        if channels == 1 and len(data.shape) == 2 and data.shape[-1] != 1:
            # Convert stereo to mono.
            data = np.mean(data, axis=-1)
        if sr and sr != samplerate:
            raise ValueError("Wav file has samplerate %f but %f requested." % (
                samplerate, sr))
        return data, samplerate


def audio_read_ffmpeg(filename, sr=None, channels=None, mid_cancel=0.0):
    """Read a soundfile, return (d, sr).
    
    Args:
        mid_cancel: 0.0 = normal mono (L+R), 1.0 = pure side (L-R)
                   Values between blend: output = (1-mid_cancel)*(L+R) + mid_cancel*(L-R)
    """
    # Hacked version of librosa.load and audioread/ff.
    offset = 0.0
    duration = None
    dtype = np.float32
    
    # If mid_cancel is set, we need stereo to do the processing
    request_channels = 2 if (mid_cancel > 0 and channels == 1) else channels
    
    with FFmpegAudioFile(os.path.realpath(filename),
                         sample_rate=sr, channels=request_channels) as input_file:
        sr = input_file.sample_rate
        actual_channels = input_file.channels
        s_start = int(np.floor(sr * offset) * actual_channels)
        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + int(np.ceil(sr * duration) * actual_channels)
        
        # Pre-allocate a contiguous buffer based on duration estimate.
        # This avoids appending hundreds of small numpy arrays and then
        # concatenating, which fragments CPython's memory allocator and
        # causes memory that is never returned to the OS.
        est_samples = 0
        if hasattr(input_file, 'duration') and input_file.duration > 0:
            est_samples = int(input_file.duration * sr * actual_channels) + sr  # +1s padding
        
        if est_samples > 0:
            y_buf = np.empty(est_samples, dtype=dtype)
            y_pos = 0  # write cursor into y_buf
        else:
            y_buf = None
            y_list = []  # fallback to list-append if duration unknown
        
        num_read = 0
        for frame in input_file:
            frame = buf_to_float(frame, dtype=dtype)
            num_read_prev = num_read
            num_read += len(frame)
            if num_read < s_start:
                # offset is after the current frame, keep reading.
                continue
            if s_end < num_read_prev:
                # we're off the end.  stop reading
                break
            if s_end < num_read:
                # the end is in this frame.  crop.
                frame = frame[:s_end - num_read_prev]
            if num_read_prev <= s_start < num_read:
                # beginning is in this frame
                frame = frame[(s_start - num_read_prev):]
            
            # Write into pre-allocated buffer (or fallback to list)
            if y_buf is not None:
                end_pos = y_pos + len(frame)
                if end_pos > len(y_buf):
                    # Duration estimate was short — grow buffer by 50%
                    new_size = max(end_pos, int(len(y_buf) * 1.5))
                    new_buf = np.empty(new_size, dtype=dtype)
                    new_buf[:y_pos] = y_buf[:y_pos]
                    y_buf = new_buf
                y_buf[y_pos:end_pos] = frame
                y_pos = end_pos
            else:
                y_list.append(frame)

        # Build final array from buffer or list
        if y_buf is not None:
            if y_pos == 0:
                y = np.zeros(0, dtype=dtype)
            else:
                # Slice to actual size — this is a view, not a copy
                y = y_buf[:y_pos].copy()
                del y_buf  # Release the over-sized buffer immediately
            if actual_channels > 1 and y.size > 0:
                y = y.reshape((-1, 2)).T
        else:
            if not len(y_list):
                y = np.zeros(0, dtype=dtype)
            else:
                y = np.concatenate(y_list)
                del y_list
                if actual_channels > 1:
                    y = y.reshape((-1, 2)).T

    # Final cleanup for dtype and contiguity
    y = np.ascontiguousarray(y, dtype=dtype)
    
    # Apply mid-channel cancellation if requested (for speech removal)
    # 
    # CROSSFADE APPROACH (user proposal):
    # 1. Compute normalized mono: (L + R) / 2
    # 2. Compute full side channel: L - R (NOT normalized - this is the 100% output)
    # 3. Crossfade: output = (1 - mid_cancel) * mono + mid_cancel * side
    #
    # This gives:
    # - mid_cancel=0: (L+R)/2 (normal mono)
    # - mid_cancel=0.5: 0.5*(L+R)/2 + 0.5*(L-R) = (L+R)/4 + (L-R)/2 = (3L-R)/4
    # - mid_cancel=1.0: L-R (full side, speech cancelled)
    #
    # The side channel is NOT normalized, so it has more weight at intermediate
    # values, progressively bringing in more of the speech-cancelled signal.
    if mid_cancel > 0 and len(y.shape) == 2 and y.shape[0] == 2:
        L, R = y[0], y[1]
        mono = (L + R) / 2.0  # Normalized mono (original audio)
        side = L - R          # Full side channel (100% mid-cancel output)
        # Crossfade between normalized mono and full side
        y = (1.0 - mid_cancel) * mono + mid_cancel * side
    elif channels == 1 and len(y.shape) == 2:
        # Standard mono conversion if no mid_cancel but we got stereo
        y = np.mean(y, axis=0)

    return (y, sr)


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.
    .. seealso:: :func:`librosa.util.buf_to_float`
    :parameters:
        - x : np.ndarray [dtype=int]
            The integer-valued data buffer
        - n_bytes : int [1, 2, 4]
            The number of bytes per sample in ``x``
        - dtype : numeric type
            The target output type (default: 32-bit float)
    :return:
        - x_float : np.ndarray [dtype=float]
            The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1. / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)


# The code below is adapted from:
# https://github.com/sampsyo/audioread/blob/master/audioread/ffdec.py
# Below is its original copyright notice:

# This file is part of audioread.
# Copyright 2014, Adrian Sampson.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.


class QueueReaderThread(threading.Thread):
    """A thread that consumes data from a filehandle and sends the data
    over a Queue.
    """

    def __init__(self, fh, blocksize=1024, discard=False):
        super(QueueReaderThread, self).__init__()
        self.fh = fh
        self.blocksize = blocksize
        self.daemon = True
        self.discard = discard
        self.queue = None if discard else queue.Queue()

    def run(self):
        while True:
            data = self.fh.read(self.blocksize)
            if not self.discard:
                self.queue.put(data)
            if not data:
                # Stream closed (EOF).
                break


class FFmpegAudioFile(object):
    """An audio file decoded by the ffmpeg command-line utility."""

    def __init__(self, filename, channels=None, sample_rate=None, block_size=4096):
        if not os.path.isfile(filename):
            raise ValueError(filename + " not found.")
        popen_args = ['ffmpeg', '-i', filename, '-f', 's16le']
        self.channels = channels
        self.sample_rate = sample_rate
        if channels:
            popen_args.extend(['-ac', str(channels)])
        if sample_rate:
            popen_args.extend(['-ar', str(sample_rate)])
        popen_args.append('-')
        self.proc = subprocess.Popen(
                popen_args,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Start another thread to consume the standard output of the
        # process, which contains raw audio data.
        self.stdout_reader = QueueReaderThread(self.proc.stdout, block_size)
        self.stdout_reader.start()

        # Read relevant information from stderr.
        try:
            self._get_info()
        except ValueError:
            raise ValueError("Error reading header info from " + filename)

        # Start a separate thread to read the rest of the data from
        # stderr. This (a) avoids filling up the OS buffer and (b)
        # collects the error output for diagnosis.
        self.stderr_reader = QueueReaderThread(self.proc.stderr)
        self.stderr_reader.start()

    def read_data(self, timeout=10.0):
        """Read blocks of raw PCM data from the file."""
        # Read from stdout in a separate thread and consume data from
        # the queue.
        start_time = time.time()
        while True:
            # Wait for data to be available or a timeout.
            data = None
            try:
                data = self.stdout_reader.queue.get(timeout=timeout)
                if data:
                    yield data
                else:
                    # End of file.
                    break
            except queue.Empty:
                # Queue read timed out.
                end_time = time.time()
                if not data:
                    if end_time - start_time >= timeout:
                        # Nothing interesting has happened for a while --
                        # FFmpeg is probably hanging.
                        raise ValueError('ffmpeg output: {}'.format(
                                ''.join(self.stderr_reader.queue.queue)
                        ))
                    else:
                        start_time = end_time
                        # Keep waiting.
                        continue

    def _get_info(self):
        """Reads the tool's output from its stderr stream, extracts the
        relevant information, and parses it.
        """
        out_parts = []
        while True:
            line = self.proc.stderr.readline()
            if not line:
                # EOF and data not found.
                raise ValueError("stream info not found")

            # In Python 3, result of reading from stderr is bytes.
            if isinstance(line, bytes):
                line = line.decode('utf8', 'ignore')

            line = line.strip().lower()

            if 'no such file' in line:
                raise IOError('file not found')
            elif 'invalid data found' in line:
                raise ValueError()
            elif 'duration:' in line:
                out_parts.append(line)
            elif 'audio:' in line:
                out_parts.append(line)
                self._parse_info(''.join(out_parts))
                break

    def _parse_info(self, s):
        """Given relevant data from the ffmpeg output, set audio
        parameter fields on this object.
        """
        # Sample rate.
        match = re.search(r'(\d+) hz', s)
        if match:
            self.sample_rate_orig = int(match.group(1))
        else:
            self.sample_rate_orig = 0
        if self.sample_rate is None:
            self.sample_rate = self.sample_rate_orig

        # Channel count.
        match = re.search(r'hz, ([^,]+),', s)
        if match:
            mode = match.group(1)
            if mode == 'stereo':
                self.channels_orig = 2
            else:
                match = re.match(r'(\d+) ', mode)
                if match:
                    self.channels_orig = int(match.group(1))
                else:
                    self.channels_orig = 1
        else:
            self.channels_orig = 0
        if self.channels is None:
            self.channels = self.channels_orig

        # Duration.
        match = re.search(
                r'duration: (\d+):(\d+):(\d+).(\d)', s
        )
        if match:
            durparts = list(map(int, match.groups()))
            duration = (
                    durparts[0] * 60 * 60 +
                    durparts[1] * 60 +
                    durparts[2] +
                    float(durparts[3]) / 10
            )
            self.duration = duration
        else:
            # No duration found.
            self.duration = 0

    def close(self):
        """Close the ffmpeg process used to perform the decoding."""
        # Kill the process if it is still running.
        if hasattr(self, 'proc') and self.proc.returncode is None:
            self.proc.kill()
            self.proc.wait()

    def __del__(self):
        self.close()

    # Iteration.
    def __iter__(self):
        return self.read_data()

    # Context manager.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
