import os
import sys
from pathlib import Path
import pytest

# Configuration Pytest placée désormais dans tests/conftest.py
# - Ajuste sys.path pour permettre:
#   * import service
#   * from src import service

def pytest_sessionstart(session):
    # Ajouter les chemins nécessaires pour que `import service` et `from src import service` fonctionnent
    repo_root = Path(__file__).resolve().parents[1]  # projet racine
    src_dir = repo_root / "src"

    # S'assurer que le projet racine est dans sys.path (permet `from src import service`)
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # S'assurer que le dossier src est aussi dans sys.path (permet `import service`)
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    Affiche un message SUCCESS si toute la suite de tests passe
    """
    if exitstatus == 0:
        terminalreporter.write_sep("=", "SUCCESS")
