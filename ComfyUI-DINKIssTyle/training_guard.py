"""One GPU training reservation shared by the Comfy-side trainer services."""
import threading

GPU_LOCK = threading.RLock()
SERVICES = {}


def register_trainer(name, service):
    SERVICES[name] = service


def active_training(exclude=None):
    for name, get_service in tuple(SERVICES.items()):
        if name == exclude:
            continue
        job_id = get_service().active()
        if job_id:
            return {"family": name, "id": job_id}
    return None


def require_other_trainers_idle(name):
    state = active_training(exclude=name)
    if state:
        raise ValueError(f"Finish or pause the active {state['family']} training job before starting {name}.")
