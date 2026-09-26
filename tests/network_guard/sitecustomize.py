"""Deny network access in Python subprocesses launched by tests and CLI smoke."""

import sys


def deny_network(event, args):
    if event in {"socket.__new__", "socket.connect", "socket.bind", "socket.getaddrinfo"}:
        raise AssertionError("Tests must not open network sockets; inject a transport fixture")


sys.addaudithook(deny_network)
