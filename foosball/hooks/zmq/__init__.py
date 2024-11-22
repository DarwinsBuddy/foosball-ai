import json
import logging

import zmq
from .. import Hook

class ZMQHook(Hook):

    def __init__(self, host="localhost", port=5555, topic="ws"):
        super().__init__()
        self.zmq_pub = ZMQPub(address=f"tcp://{host}:{port}")
        self.topic = topic

    def invoke(self, message: dict | str, *args, **kwargs):
        # TODO: send here FrameStats
        self.zmq_pub.publish(message, topic=self.topic)

    def start(self, *args, **kwargs):
        pass

    def stop(self, *args, **kwargs):
        self.zmq_pub.close()


class ZMQPub:
    def __init__(self, address="tcp://127.0.0.1:5555"):
        self.address = address
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        logging.info(f"ZMQ Publisher started at {self.address}")
        self.socket.bind(self.address)  # Bind the socket to the address

    def publish(self, message: str | dict, topic=None):
        msg = json.dumps(message) if type(message) is dict else message
        logging.debug(f"ZMQ publish {msg} to {topic} ({self.address})")
        if topic is not None:
            self.socket.send_string(f"{topic} {msg}")
        else:
            # Send the message
            self.socket.send_string(msg)

    def close(self):
        self.socket.close()
        logging.info(f"ZMQ Publisher closed for {self.address}")
