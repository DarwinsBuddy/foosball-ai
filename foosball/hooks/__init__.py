from abc import ABC, abstractmethod


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


class HookManager:
    def __init__(self, hooks=None):
        self.hooks = hooks or []

    def start(self):
        for hook in self.hooks:
            hook.start()

    def invoke(self, *args, **kwargs):
        for hook in self.hooks:
            hook.invoke(*args, **kwargs)

    def stop(self):
        for hook in self.hooks:
            hook.stop()