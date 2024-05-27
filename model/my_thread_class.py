from threading import Thread

class my_custom_thread(Thread):

    def __init__(self, target=None, group=None, name=None, args=(), kwargs={}):
        Thread.__init__(target, group, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return
