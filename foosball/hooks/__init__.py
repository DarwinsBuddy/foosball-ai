import threading
from abc import ABC, abstractmethod

import asyncio


class Hook(ABC):

    def __init__(self):
        pass
    @abstractmethod
    def invoke(self, *args, **kwargs):
        pass

    @abstractmethod
    def start(self, *args, **kwargs):
        pass

    @abstractmethod
    def stop(self, *args, **kwargs):
        pass


class HookManager(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.hooks = []
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.stop_event = asyncio.Event()

    def run(self):
        for hook in self.hooks:
            self.loop.call_soon_threadsafe(hook.start)
        self.loop.run_until_complete(self.stop_event.wait())

    def extend(self, hooks: [Hook], start=False):
        self.hooks.extend(hooks)
        for hook in hooks:
            if start:
                self.loop.call_soon_threadsafe(hook.start)

    def add(self, hook: Hook, start=False):
        self.hooks.append(hook)
        if start:
            self.loop.call_soon_threadsafe(hook.start)

    def remove(self, hook: Hook):
        self.loop.call_soon_threadsafe(hook.stop)
        self.hooks.remove(hook)

    def invoke(self, *args, **kwargs):
        for hook in self.hooks:
            self.loop.call_soon_threadsafe(hook.invoke(*args, **kwargs))

    def stop(self):
        for hook in self.hooks:
            self.loop.call_soon_threadsafe(hook.stop)
        self.hooks = []
        self.stop_event.set()