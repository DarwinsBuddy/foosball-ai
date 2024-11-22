import logging
import os
import random

from foosball.hooks import Hook


class AudioHook(Hook):

    def __init__(self, folder: str):
        super().__init__()
        self.folder = folder

    def start(self, *args, **kwargs):
        pass

    def stop(self, *args, **kwargs):
        pass

    def invoke(self, *args, **kwargs):
        AudioHook.play_random_sound(self.folder)

    @staticmethod
    def play_sound(sound_file: str):
        from playsound import playsound
        if os.path.isfile(sound_file):
            playsound(sound_file, block=False)
        else:
            logging.warning(f"Audio not found: {sound_file}")

    @staticmethod
    def play_random_sound(folder: str, prefix: str = './assets/audio'):
        path = f'{prefix}/{folder}'
        audio_file = random.choice(os.listdir(path))
        AudioHook.play_sound(f"{path}/{audio_file}")
