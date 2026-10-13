"""Load each module independently so unavailable dependencies affect only its nodes."""
import importlib
import logging

LOGGER = logging.getLogger(__name__)


def load_nodes(package, modules):
    nodes, loaded, failures = {}, {}, {}
    for module, names in modules:
        try:
            imported = importlib.import_module("." + module, package)
            classes = {name: getattr(imported, name) for name in names}
        except Exception as error:
            failures[module] = str(error)
            LOGGER.exception("DKST: could not load %s; continuing with other nodes. "
                             "Check dependencies with this pack's install.py.", module)
            continue
        loaded[module] = imported
        nodes.update(classes)
    return nodes, loaded, failures
