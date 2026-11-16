import threading

class FrameBuffer:
    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()
        self._novo_frame = threading.Event()

    def atualizar(self, frame):
        with self._lock:
            self._frame = frame.copy()
        self._novo_frame.set()

    def obter(self, timeout=1.0):
        if not self._novo_frame.wait(timeout):
            return None
        with self._lock:
            frame = self._frame
            self._novo_frame.clear()
        return frame.copy() if frame is not None else None


frame_buffer = FrameBuffer()