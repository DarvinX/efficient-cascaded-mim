"""Defer termination until the current unit of work has a durable checkpoint."""
import signal

requested = False


def _request(signum, frame):
    global requested
    requested = True


def install():
    signal.signal(signal.SIGTERM, _request)
    signal.signal(signal.SIGINT, _request)
