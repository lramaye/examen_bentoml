import os
import shutil
import tempfile
import pytest

# Désactiver Prometheus/Metrics pour toute la session Pytest, avant tout import de BentoML
_TMP_PROM_DIR = None

def pytest_sessionstart(session):
    global _TMP_PROM_DIR
    # Désactivation explicite des métriques pour différentes versions de BentoML
    os.environ.setdefault("BENTOML_DISABLE_PROMETHEUS", "true")
    os.environ.setdefault("BENTOML_DISABLE_METRICS", "1")

    # Assurer un répertoire multiprocess Prometheus valide si jamais activé par défaut
    _TMP_PROM_DIR = tempfile.mkdtemp(prefix="pytest_prom_multiproc_")
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = _TMP_PROM_DIR


def pytest_sessionfinish(session, exitstatus):
    # Nettoyage du dossier multiprocess si créé
    global _TMP_PROM_DIR
    if _TMP_PROM_DIR and os.path.isdir(_TMP_PROM_DIR):
        shutil.rmtree(_TMP_PROM_DIR, ignore_errors=True)
        _TMP_PROM_DIR = None


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    envoi le message success
    """
    if exitstatus == 0:
        terminalreporter.write_sep("=", "SUCCESS")
