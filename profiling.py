from cProfile import Profile
import io
import pstats
from pstats import SortKey

from foosball import ai, capture


class Args:
    args = {
        'off': True,
        'video': '../data/fps200_2.MP4',
        'buffer': 64
    }

    def get(self, key):
        if key in self.args:
            return self.args[key]
        else:
            return None


args = Args()
pr = Profile()
pr.enable()
ai.process_video(args, cap=capture.Capture(args.get('video')))
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
