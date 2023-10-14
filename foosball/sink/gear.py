from typing import Callable

from vidgear.gears import StreamGear

from . import Sink


class StreamSink(Sink):

    def __init__(self, name='frame', live: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        stream_params = {
            '-pix_fmt': 'yuv420p',
            "-vcodec": "libx264",
            "-preset:v": "veryfast",
            "-f": "hls",
            "-hls_flags": "delete_segments",
            '-livestream': live,
            '-clear_prev_assets': True

        }
        self.streamer = StreamGear(output="stream.mpd", **stream_params)

    def render(self, callbacks: dict[int, Callable] = None):
        pass

    def stop(self):
        self.streamer.terminate()

    def show(self, frame):
        self.streamer.stream(frame=frame, rgb_mode=False)
