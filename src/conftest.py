import os
os.environ.setdefault("BENTOML_CONFIG_OPTIONS", "api_server.metrics.enabled=false,tracing.enabled=false")
import sys
from pathlib import Path
import pytest

# Désactiver Prometheus/Metrics pour toute la session Pytest, avant tout import de BentoML

def pytest_sessionstart(session):
    # Désactivation explicite des métriques pour différentes versions de BentoML
    # Ajouter les chemins nécessaires pour que `import service` et `from src import service` fonctionnent
    src_dir = Path(__file__).resolve().parent  # .../src
    repo_root = src_dir.parent                 # projet racine

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
    envoi le message success
    """
    if exitstatus == 0:
        terminalreporter.write_sep("=", "SUCCESS")
