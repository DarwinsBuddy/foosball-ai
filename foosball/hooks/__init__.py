import dataclasses
import logging
import os
import random
import threading
import traceback
from typing import Mapping, Callable

import urllib3
import yaml

from foosball.models import Team

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Webhook:
    method: str
    url: str
    json: dict = None
    headers: Mapping[str, str] = None


def load_goal_webhook() -> Callable[[Team], Webhook]:
    filename = 'goal_webhook.yaml'

    def to_webhook(webhook_dict: dict, team: Team):
        webhook_dict['json']['team'] = team.value
        return Webhook(**webhook_dict)
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            wh = yaml.safe_load(f)
            return lambda team: to_webhook(wh, team)
    else:
        logger.info("No goal webhook configured under 'goal_webhook.yaml'")


generate_goal_webhook = load_goal_webhook()


def webhook(whook: Webhook):
    threading.Thread(
        target=_webhook,
        args=[whook],
        daemon=True
    ).start()


def _webhook(whook: Webhook):
    try:
        headers = {} if whook.headers is None else whook.headers
        if whook.json is not None and "content-type" not in headers:
            headers['content_type'] = 'application/json'
        response = urllib3.request(whook.method, whook.url, json=whook.json, headers=headers)
        logger.debug(f"webhook response: {response.status}")
    except Exception as e:
        logger.error(f"Webhook failed - {e}")
        traceback.print_exc()


def play_random_sound(folder: str, prefix: str = './assets/audio'):
    path = f'{prefix}/{folder}'
    audio_file = random.choice(os.listdir(path))
    play_sound(f"{path}/{audio_file}")


def play_sound(sound_file: str):
    from playsound import playsound
    if os.path.isfile(sound_file):
        playsound(sound_file, block=False)
    else:
        logger.warning(f"Audio not found: {sound_file}")
