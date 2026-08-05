"""PTY-backed regression tests for the curses TUI renderer.

The cell tests use a lightweight fake window. This module additionally starts
curses in a real child process attached to a pseudoterminal, which catches
terminal-size, bottom-row, and initialization problems that a fake window
cannot reproduce.

The CLI lives in ``benchmark/cli.py`` (the root ``ai-benchmark.py`` is a thin
launcher); the child process loads it via ``spec_from_file_location``.
"""
import os
import pathlib
import pty
import select
import struct
import sys
import termios
import time
import unittest

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_AI_BENCHMARK_PATH = _THIS_DIR.parent / "benchmark" / "cli.py"


class TestTuiPty(unittest.TestCase):
    """Exercise the renderer through a real curses terminal session."""

    @unittest.skipUnless(hasattr(pty, "fork"), "PTY support is unavailable")
    def test_curses_redraws_shrinking_footer_and_unicode_without_corruption(self):
        module_path = str(_AI_BENCHMARK_PATH)
        child_script = f"""
import curses
import importlib.util
import json
import os
import sys

spec = importlib.util.spec_from_file_location("ai_benchmark", {module_path!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

screen = None
try:
    screen = curses.initscr()
    curses.noecho()
    curses.cbreak()
    screen.nodelay(True)
    max_y, max_x = screen.getmaxyx()
    if max_y < 4 or max_x < 24:
        raise RuntimeError(f"terminal too small: {{max_y}}x{{max_x}}")

    # Render a wide frame first, then wait for the parent to resize the
    # actual PTY to a portrait-shaped terminal. This reproduces the mobile
    # case rather than only exercising a fixed 80-column fake.
    module._wr(screen, max_x, max_y, max_y - 1, 0, "038s")
    screen.refresh()
    os.write(2, b"PTY_READY")
    os.read(0, 1)
    curses.resizeterm(20, 24)
    max_y, max_x = screen.getmaxyx()
    module._wr(screen, max_x, max_y, max_y - 1, 0, "38s")
    module._wr(screen, max_x, max_y, 0, 0,
               "A👨‍👩‍👧‍👦B" + chr(27) + "[2J" + chr(27) + "[H done")
    screen.refresh()

    footer = screen.instr(max_y - 1, 0, max_x).decode("utf-8", "replace")
    header = screen.instr(0, 0, max_x).decode("utf-8", "replace")
    if not footer.startswith("38s") or "038s" in footer:
        raise AssertionError(f"bad footer row: {{footer!r}}")
    if chr(27) in header or chr(27) in footer:
        raise AssertionError("terminal control character reached curses")
    if "A" not in header or "👨" not in header:
        raise AssertionError(f"Unicode row was not rendered: {{header!r}}")
finally:
    if screen is not None:
        try:
            curses.echo()
            curses.nocbreak()
        finally:
            curses.endwin()

os.write(2, ("PTY_FINAL=" + json.dumps({{"footer": footer, "header": header}})).encode())
"""

        pid, master_fd = pty.fork()
        if pid == 0:
            os.environ["TERM"] = "xterm-256color"
            os.execv(sys.executable, [sys.executable, "-c", child_script])

        # Give curses a predictable, non-edge terminal size. The child has
        # already acquired the PTY as its controlling terminal at this point.
        rows, columns = 12, 80
        winsize = struct.pack("HHHH", rows, columns, 0, 0)
        # tcsetwinsize is the portable Python wrapper; the ioctl payload above
        # remains useful on Python versions where that wrapper is unavailable.
        try:
            termios.tcsetwinsize(master_fd, (rows, columns))
        except (AttributeError, OSError):
            pass
        try:
            import fcntl
            fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
        except (AttributeError, OSError):
            pass

        output = bytearray()
        resized = False
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            readable, _, _ = select.select([master_fd], [], [], 0.2)
            if not readable:
                continue
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            output.extend(chunk)
            if not resized and b"PTY_READY" in output:
                resized = True
                narrow_winsize = struct.pack("HHHH", 20, 24, 0, 0)
                try:
                    termios.tcsetwinsize(master_fd, (20, 24))
                except (AttributeError, OSError):
                    pass
                try:
                    import fcntl
                    fcntl.ioctl(master_fd, termios.TIOCSWINSZ, narrow_winsize)
                except (AttributeError, OSError):
                    pass
                os.write(master_fd, b"x")
        else:
            os.kill(pid, 9)

        _, status = os.waitpid(pid, 0)
        os.close(master_fd)
        decoded = output.decode("utf-8", "replace")
        self.assertTrue(os.WIFEXITED(status), decoded)
        self.assertEqual(os.waitstatus_to_exitcode(status), 0, decoded)
        self.assertIn("PTY_READY", decoded, decoded)
        self.assertIn("PTY_FINAL=", decoded, decoded)
        self.assertIn('"footer": "38s', decoded, decoded)
        self.assertNotIn("Traceback", decoded, decoded)


if __name__ == "__main__":
    unittest.main()
