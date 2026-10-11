# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Input latency under run(): the time from a key reaching the terminal to the end of the first frame that shows it
(python -m benchmarks.latency --scene arena --size 206x49). Unix only.

A genre scene (see genres.py) runs under run() in a pseudo-terminal; this process plays the terminal: it types a
letter at a random moment, and the program writes that letter eight times in the corner of the next frame it draws.
The latency is the time from typing it to reading the end of that frame's output (its synchronized-update marker),
which is when a terminal would show it. What the terminal then takes to draw the frame is left out: it is the same
whatever the program does.

--rate limits how fast the terminal reads (MB/s) to stand in for one slower than the program, the case where
drawing the next frame while the last is written (run()'s background output) could hold a frame back. --sync
writes each frame from the drawing thread instead, as run() did before that, to compare.
"""
import argparse
import json
import os
import random
import select
import statistics
import sys
import tempfile
import time

SYNC_END = b"\x1b[?2026l"
MARK = 8  # letters in the marker


def child(scene, fps, sync, trace, stats_path):
    """The program: a genre scene under run(), showing the last letter typed in the top left corner."""
    from unicode3d import terminal
    from unicode3d.scene import Renderer
    from .genres import DT, GENRES

    terminal.BACKGROUND_OUTPUT = not sync
    writes = []  # with --trace: (seconds from queuing a frame to its write starting, seconds the write took)
    if trace:
        from unicode3d.console import PosixConsole
        queue, write, queued = terminal._Writer.write, PosixConsole.write, [0.0]

        def timed_queue(self, data):
            queue(self, data)
            queued[0] = time.perf_counter()

        def timed_write(self, text):
            t = time.perf_counter()
            write(self, text)
            writes.append((t - queued[0] if not sync else 0.0, time.perf_counter() - t))
        terminal._Writer.write, PosixConsole.write = timed_queue, timed_write
    game = GENRES[scene]()
    state = {"renderer": None, "letter": " ", "times": []}

    def frame(screen, dt, keys):
        if state["renderer"] is None:
            rows, cols = screen.size()
            state["renderer"] = Renderer(cols, rows, screen.cell_pixels, **game.renderer_options)
        for key in keys:
            if key == ord("0"):
                return False
            if ord("A") <= key <= ord("Z"):
                state["letter"] = chr(key)
        t0 = time.perf_counter()
        objects, camera, lights = game.update(DT)
        screen.draw_frame(state["renderer"].render(objects, camera, lights))
        game.hud(screen, state["renderer"])
        screen.text(0, 0, state["letter"] * MARK)
        screen.refresh()
        state["times"].append(time.perf_counter() - t0)

    terminal.run(frame, fps=fps)
    with open(stats_path, "w") as f:
        json.dump({"times": state["times"], "writes": writes}, f)


def measure(scene, rows, cols, fps, sync, rate, trials, trace=False, warmup=60):
    """Run the scene in a pseudo-terminal and type `trials` letters; returns (latencies, frame gaps, frame times,
    bytes a frame, writes: see child()), times in seconds."""
    import fcntl
    import pty
    import struct
    import termios

    stats = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    args = ["-m", "benchmarks.latency", "--child", scene, "--fps", str(fps), "--stats", stats] + (["--sync"] if sync else []) + (["--trace"] if trace else [])
    pid, fd = pty.fork()
    if pid == 0:
        try:
            os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            os.execve(sys.executable, [sys.executable, *args], dict(os.environ, TERM="xterm-256color"))
        finally:
            os._exit(127)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    tail, frames, ends, sizes, size = b"", 0, [], [], 0
    letters = [chr(ord("A") + i % 26) for i in range(trials)]
    latencies, waiting, sent, next_key = [], None, 0.0, None
    budget, last = 0.0, time.perf_counter()
    rng = random.Random(1)
    try:
        while len(latencies) < trials:
            now = time.perf_counter()
            if rate:  # read no faster than the terminal would
                budget = min(budget + (now - last) * rate * 1e6, 1 << 16)
                last = now
            if waiting is None and frames >= warmup and now >= (next_key or 0):
                waiting = letters[len(latencies)]
                os.write(fd, waiting.encode())
                sent = time.perf_counter()
            timeout = 0.0005 if rate and budget < 1 else 0.05
            if waiting is None and next_key is not None:  # (wake when the key is due: typing must not wait for output)
                timeout = min(timeout, max(0.0, next_key - now))
            if not select.select([fd], [], [], timeout)[0] or (rate and budget < 1):
                continue
            try:
                data = os.read(fd, int(budget) if rate else 1 << 20)
            except OSError:  # the program has exited
                data = b""
            if not data:
                raise RuntimeError(f"the program stopped:\n{(tail[-3000:]).decode('utf-8', 'replace')}")
            budget -= len(data) if rate else 0
            size += len(data)
            buf = tail + data
            start = 0
            while True:  # each frame end in this read
                i = buf.find(SYNC_END, start)
                if i < 0:
                    break
                at = time.perf_counter()
                frames += 1
                ends.append(at)
                sizes.append(size)
                size = 0
                if waiting is not None and (waiting * MARK).encode() in buf[:i]:
                    latencies.append(at - sent)
                    waiting = None
                    next_key = at + rng.uniform(0.0, 0.1)  # (a random moment in the frames to come)
                start = i + len(SYNC_END)
            tail = buf[start:]  # (the frame on its way, which may hold the marker)
        os.write(fd, b"0")
        while select.select([fd], [], [], 5.0)[0]:
            try:
                if not os.read(fd, 1 << 20):
                    break
            except OSError:
                break
        os.waitpid(pid, 0)
    finally:
        os.close(fd)
    with open(stats) as f:
        got = json.load(f)
    os.unlink(stats)
    times, writes = got["times"], got["writes"]
    gaps = [b - a for a, b in zip(ends[warmup:], ends[warmup + 1:])]
    return latencies, gaps, times[warmup:], sizes[warmup:], writes[warmup:]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="arena", help="genre scene (corridor, arena, board)")
    parser.add_argument("--size", default="206x49", help="terminal size, COLSxROWS")
    parser.add_argument("--fps", type=float, default=30, help="run()'s target frame rate (0: as fast as it can)")
    parser.add_argument("--rate", type=float, default=0, help="MB/s the terminal reads (0: as fast as it can)")
    parser.add_argument("--sync", action="store_true", help="write each frame from the drawing thread")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--trace", action="store_true", help="also time each write, and its wait to start")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--stats", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return child(args.child, args.fps, args.sync, args.trace, args.stats)
    cols, rows = map(int, args.size.split("x"))
    lat, gaps, times, sizes, writes = measure(args.scene, rows, cols, args.fps, args.sync, args.rate, args.trials,
                                              args.trace)
    ms = lambda xs, q: 1000 * statistics.quantiles(xs, n=20)[q] if len(xs) > 1 else float("nan")
    print(f"{args.scene} {cols}x{rows} fps {args.fps:g} rate {args.rate or 'unlimited'} "
          f"{'sync' if args.sync else 'background'}: latency mean {1000 * statistics.mean(lat):.1f} ms, "
          f"median {ms(lat, 9):.1f}, p95 {ms(lat, 18):.1f}, max {1000 * max(lat):.1f}; "
          f"frame gap {1000 * statistics.mean(gaps):.1f} ms, drawing {1000 * statistics.mean(times):.1f} ms, "
          f"{statistics.mean(sizes) / 1000:.0f} kB a frame")
    if writes:
        print(f"writes: {1000 * statistics.mean(w for _, w in writes):.1f} ms mean, after waiting "
              f"{1000 * statistics.mean(q for q, _ in writes):.1f} ms mean to start")


if __name__ == "__main__":
    main()
