"""Notice a submission receipt while a prepared image review is waiting."""
from contextlib import closing
import os
from pathlib import Path
import select
import sys
import time

from admission import between_leases, handoff_lock, take_slot, worker_pid
from p9search.storage import read_json

def yield_after_receipt(work, review, timeout=1800):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and worker_pid():
        if not review.exists() or read_json(review).get('reason') != 'public worker has the search slot':
            return
        task = read_json(work / 'worker-state.json').get('task_id', '')
        if len(task) != 64 or not (work / task).is_dir():
            time.sleep(1)
            continue
        try:
            fd = os.open(work / task, os.O_RDONLY)
        except FileNotFoundError:
            continue
        try:
            with closing(select.kqueue()) as queue:
                event = select.kevent(fd, filter=select.KQ_FILTER_VNODE, flags=select.KQ_EV_ADD|select.KQ_EV_CLEAR,
                                     fflags=select.KQ_NOTE_WRITE|select.KQ_NOTE_DELETE|select.KQ_NOTE_RENAME)
                queue.control([event],0,0)
                while time.monotonic() < deadline and (work / task).is_dir():
                    receipt = work / task / 'submission.json'
                    if receipt.exists() and between_leases(work):
                        with handoff_lock():
                            state = read_json(work / 'worker-state.json')
                            if (worker_pid() and state.get('state') == 'submitting' and state.get('task_id') == task
                                    and read_json(review).get('reason') == 'public worker has the search slot'
                                    and receipt.exists() and read_json(receipt).get('accepted') is True):
                                allocation, reason = take_slot(work)
                                if allocation:
                                    allocation.close()
                                print(reason,flush=True)
                                return
                    if not queue.control(None,1,5) and (not worker_pid() or read_json(review).get('state') != 'waiting'):
                        return
        finally:
            os.close(fd)

if __name__ == '__main__':
    yield_after_receipt(Path(sys.argv[1]),Path(sys.argv[2]))
