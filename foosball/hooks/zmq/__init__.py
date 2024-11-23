import json
import logging
import threading
import time
from multiprocessing import Queue
from queue import Full, Empty

import zmq

from .. import Hook


class ZMQHook(Hook):

    def __init__(self, host="localhost", port=5555, topic="ws"):
        super().__init__()
        self.zmq = ZMQDispatcher(host, port, topic)
        self.started = False
        self.start_lock = threading.Lock()

    def invoke(self, message: dict | str, *args, **kwargs):
        self.zmq.send(message)

    def start(self, *args, **kwargs):
        with self.start_lock:
            self.started = True
            self.zmq.start()

    def stop(self, *args, **kwargs):
        self.zmq.stop()

class ZMQDispatcher(threading.Thread):
    def __init__(self, host="localhost", port=5555, topic="ws"):
        super().__init__(daemon=True)
        self.zmq_pub = ZMQPub(address=f"tcp://{host}:{port}")
        self.topic = topic
        self.q = Queue()
        self.stopped = threading.Event()

    def send(self, message: dict | str, *args, **kwargs):
        try:
            if not self.stopped.is_set():
                self.q.put_nowait(message)
        except Full:
            logging.error("ZMQHook queue is full")

    def run(self):
        logging.debug("ZMQDispatcher started")
        try:
            self.zmq_pub.start()
            while not self.stopped.is_set():
                try:
                    message = self.q.get(timeout=0.1)
                    self.zmq_pub.publish(message, topic=self.topic)
                except Empty:
                    pass
        except Exception as e:
            logging.error(f"Error in ZMQDispatcher {e}")
            self.stopped.set()
        finally:
            try:
                while True:
                    self.q.get_nowait()
            except Empty:
                pass
            self.zmq_pub.close()

    def stop(self):
        self.stopped.is_set()


class ZMQPub:
    def __init__(self, address="tcp://127.0.0.1:5555"):
        self.address = address
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        logging.info(f"ZMQ Publisher started at {self.address}")

    def start(self):
        self.socket.bind(self.address)  # Bind the socket to the address
        time.sleep(0.5)

    def publish(self, message: str | dict, topic=None):
        try:
            msg = json.dumps(message) if type(message) is dict else message
            # logging.debug(f"ZMQ publish {msg} to {topic} ({self.address})")
            if topic is not None:
                self.socket.send_string(f"{topic} {msg}")
            else:
                # Send the message
                self.socket.send_string(msg)
        except Exception as e:
            logging.error(f"Error in ZMQ Publisher {e}")

    def close(self):
        self.socket.close()
        logging.info(f"ZMQ Publisher closed for {self.address}")
