"""Compatibility shim for ``webrtcvad`` on modern Python packaging stacks.

The upstream ``webrtcvad`` Python wrapper imports ``pkg_resources`` only to
expose ``__version__``. Recent setuptools releases no longer ship that module,
which breaks import on otherwise functional installations. This shim preserves
the small API surface this repo uses and falls back to the C extension
directly when the upstream wrapper cannot be imported.
"""

from __future__ import annotations

from importlib import metadata

try:
    import webrtcvad as _webrtcvad_module
except ModuleNotFoundError as exc:
    if exc.name != "pkg_resources":
        raise

    import _webrtcvad

    __version__ = metadata.version("webrtcvad")

    class Vad:
        """Mirror the upstream wrapper using the native extension directly."""

        def __init__(self, mode: int | None = None):
            self._vad = _webrtcvad.create()
            _webrtcvad.init(self._vad)
            if mode is not None:
                self.set_mode(mode)

        def set_mode(self, mode: int) -> None:
            _webrtcvad.set_mode(self._vad, mode)

        def is_speech(
            self, buf: bytes, sample_rate: int, length: int | None = None
        ) -> bool:
            length = length or int(len(buf) / 2)
            if length * 2 > len(buf):
                raise IndexError(
                    "buffer has %s frames, but length argument was %s"
                    % (int(len(buf) / 2.0), length)
                )
            return _webrtcvad.process(self._vad, sample_rate, buf, length)

    def valid_rate_and_frame_length(rate: int, frame_length: int) -> bool:
        return _webrtcvad.valid_rate_and_frame_length(rate, frame_length)

else:
    Vad = _webrtcvad_module.Vad
    valid_rate_and_frame_length = _webrtcvad_module.valid_rate_and_frame_length
    __version__ = getattr(_webrtcvad_module, "__version__", "unknown")
