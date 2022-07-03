from vidgear.gears import StreamGear


class StreamDisplay:

    def __init__(self, name='frame', *args, **kwargs):
        self.name = name
        output_params = {
            '-pix_fmt': 'yuv420p',
            "-vcodec": "libx264",
            "-preset:v": "veryfast",
            "-f": "flv",
            '-livestream': True,
            '-clear_prev_assets': True

        }
        self.streamer = StreamGear(output="stream.mpd", **output_params)

    def stop(self):
        self.streamer.terminate()

    def show(self, frame):
        self.streamer.stream(frame=frame, rgb_mode=True)

    @staticmethod
    def render(reset_cb=None):
        pass
