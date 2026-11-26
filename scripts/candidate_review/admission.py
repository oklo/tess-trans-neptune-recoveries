"""Yield the traveling Mac between public leases and local pixel reductions."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess

from p9search.storage import read_json, write_json

SERVICE = f"gui/{os.getuid()}/org.oklo.p9.usable-pixel-recovery"


def available_memory():
    level = subprocess.check_output(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True)
    return level.strip() == "1"


def worker_pid():
    result = subprocess.run(["launchctl", "print", SERVICE], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Public worker service is not loaded")
    match = re.search(r"^\s*pid = (\d+)$", result.stdout, re.M)
    return int(match[1]) if match else None


def between_leases(work):
    state = read_json(work / "worker-state.json")
    if state["state"] in {"queue_wait", "capacity_wait", "approval_wait"}:
        return True
    task = state.get("task_id", "")
    return (state["state"] == "submitting" and re.fullmatch("[0-9a-f]{64}", task)
            and (work / task / "submission.json").is_file())


YIELDED = Path(__file__).resolve().parents[2] / "outputs/planet_nine_backlog_service/public-worker-yielded"


@contextmanager
def handoff_lock():
    with YIELDED.with_name("handoff.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def local_slot(work):
    with handoff_lock():
        return take_slot(work)


def take_slot(work):
    pid = worker_pid()
    if pid:
        if not between_leases(work):
            return None, "public worker has the search slot"
        # Freeze the parent only at an idle boundary and recheck for a racing claim.
        try:
            os.kill(pid, signal.SIGSTOP)
            try:
                if worker_pid() == pid and between_leases(work):
                    children = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True)
                    if children.returncode == 1:
                        YIELDED.touch()
                        os.kill(pid, signal.SIGINT)
            finally:
                os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        return None, "waiting for public worker to yield"
    if not available_memory():
        return None, "waiting for normal system memory pressure"
    lock = (work / "worker.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None, "another participant owns the search slot"
    return lock, None


def resume_public(work):
    with handoff_lock():
        if not YIELDED.exists() or worker_pid() is not None or not available_memory():
            return
        with (work / "worker.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
        subprocess.run(["launchctl", "kickstart", SERVICE], check=True)
        YIELDED.unlink(missing_ok=True)


def request_transport_retry(work):
    """Hand a released, interrupted archive download back to the usual worker."""
    with handoff_lock():
        if YIELDED.exists() or worker_pid() is not None:
            return
        state = read_json(work / 'worker-state.json')
        if state.get('state') != 'held':
            return
        prefix = re.escape(f'Search exited 1; inspect {work}/')
        match = re.fullmatch(prefix + r'([0-9a-f]{64})/worker\.log', state.get('detail', ''))
        if not match:
            return
        task = match[1]
        folder = work / task
        if (folder/'result.json').exists() or not (folder/'task.json').is_file() or not (folder/'worker.log').is_file():
            return
        if read_json(folder/'task.json')['task_id'] != task:
            return
        with (folder/'worker.log').open('rb') as stream:
            stream.seek(max(0, os.fstat(stream.fileno()).st_size - 2048))
            lines = stream.read().splitlines()
        if not lines or not lines[-1].startswith(b'p9: Response payload is not completed:') or b'Not enough data to satisfy content length header' not in lines[-1]:
            return
        path = YIELDED.with_name('download-retries.json')
        records = read_json(path) if path.exists() else {}
        attempts = records.get(task, [])
        if state['updated'] in attempts or len(attempts) >= 2:
            return
        records[task] = attempts + [state['updated']]
        write_json(path, records)
        YIELDED.touch()
        return True
