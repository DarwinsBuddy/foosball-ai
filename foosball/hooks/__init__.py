import logging
import os
import random
import threading
from abc import ABC, abstractmethod
from typing import Mapping, Self

import urllib3
import yaml


class Hook(ABC):
    @abstractmethod
    def invoke(self, *args, **kwargs):
        pass


class AudioHook(Hook):

    def __init__(self, folder: str):
        super().__init__()
        self.folder = folder

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


class Webhook(Hook):

    def __init__(self, method: str = None, url: str = None, json: dict = None, headers: Mapping[str, str] = None, *args, **kwargs):
        super().__init__()
        self.method: str = method
        self.url: str = url
        self.json: dict = json
        self.headers: Mapping[str, str] = headers

    def as_dict(self, json: dict = None) -> dict:
        d = vars(self)
        if json is not None:
            d['json'] = d['json'] | json
        return d

    def invoke(self, json: dict, *args, **kwargs):
        threading.Thread(
            target=Webhook.call,
            kwargs=self.as_dict(json),
            daemon=True
        ).start()

    @staticmethod
    def call(method: str, url: str, json: dict = None, headers: Mapping[str, str] = None):
        try:
            headers = {} if headers is None else headers
            if json is not None and "content-type" not in headers:
                headers['content_type'] = 'application/json'
            response = urllib3.request(method, url, json=json, headers=headers)
            logging.debug(f"webhook response: {response.status}")
        except Exception as e:
            logging.error(f"Webhook failed - {e}")

    @classmethod
    def load_webhook(cls, filename: str) -> Self:
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                wh = yaml.safe_load(f)
                return Webhook(**wh)
        else:
            logging.info("No goal webhook configured under 'goal_webhook.yaml'")
