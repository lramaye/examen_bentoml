import pytest


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    envoi le message success
    """
    if exitstatus == 0:
        terminalreporter.write_sep("=", "SUCCESS")
