from __future__ import absolute_import

# A dummy event for neutering profiling code
class DummyEvent(object):
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        pass

    def begin(self):
        pass

    def end(self):
        pass
