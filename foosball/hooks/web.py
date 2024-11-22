import logging
import os
import threading
from typing import Mapping

import urllib3
import yaml

from foosball.hooks import Hook


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
    def load_webhook(cls, filename: str):
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                wh = yaml.safe_load(f)
                return Webhook(**wh)
        else:
            logging.info("No goal webhook configured under 'goal_webhook.yaml'")
