import atexit
import json
import os
import threading
import time
from pathlib import Path
from typing import Callable
from warnings import warn

import psutil


class ResourceTracker:
    """
    Tracks system resources

    Parameters
    ----------
    callback : Callable
        Function to call on each resource refresh. Signature must accept:

            callable(info : dict) -> None

            where the info dict consists of:
                pid : int
                    Main process ID
                name : str
                    Main process name
                cores : int
                    Number of CPU cores available via os.cpu_count()
                mem : float
                    Main process memory used
                mem_total : float
                    Total memory of the system
                cpu : float
                    Main process CPU percentage
                timestamp : float
                    Timestamp of the resource record via time.time()
                children : list[dict]
                    Information of child processes:
                        pid : int
                            Child process ID
                        name : str
                            Child process name
                        mem : float
                            Child process memory used
                        cpu : float
                            Child process CPU percentage
            if summarize is enabled, these will also be included:
                mem_app : float
                    Total memory of the main process + children
                mem_used : float
                    Memory in use by the system, excluding the app
                mem_avail : float
                    Remaining available memory, defined as free + reclaimable
            Total memory of the system is the sum([mem_app, mem_used, mem_free])
    interval : int | float, default=2
        Interval frequency in seconds to check resources
        Must be greater than 0. Values less than 0.1 risk high CPU usage and skewing
        polled results
        The CPU usage is calculated as the percentage of CPU used over this interval
    round : int | bool, default=2
        Round the memory variables to this many decimals. Set to False or 0 to disable
        True will be set to 1
    summarize : bool, default=True
        Includes summary statistics such as the sum of all children
    allow_unsafe : bool, default=False
        Bypasses the exception and allows unsafe interval values (less than 0.1)
        Not recommended
    """

    thread = None

    def __init__(
        self,
        callback: Callable,
        interval: float = 2,
        round: bool | int = 2,
        summarize: bool = True,
        allow_unsafe: bool = False,
    ):
        if not callable(callback):
            raise AttributeError(f"The callback parameter must be a callable")

        if not isinstance(interval, (int, float)):
            raise AttributeError(f"The interval parameter must be an integer")
        if interval <= 0:
            raise AttributeError(f"The interval parameter must be greater than 0")
        if interval < 0.1:
            msg = "High CPU usage risk with an interval less than 0.1"
            if allow_unsafe:
                warn(msg)
            else:
                msg += " - If this is intended, set allow_unsafe=True"
                raise ValueError(msg)

        if isinstance(round, bool):
            round = int(round)
        if not isinstance(round, int):
            raise AttributeError(f"The round parameter must be an integer")

        self.callback = callback
        self.interval = interval
        self.round = round
        self.summarize = summarize

        self.unitLabel = "GB"
        self.unitValue = 1024**3

    def _track(self):
        """
        System resource tracker intended to be set in a thread
        """
        sys = psutil.virtual_memory()
        proc = psutil.Process()
        info = {
            "pid": proc.pid,
            "name": proc.name(),
            "cores": os.cpu_count(),
            "mem_total": sys.total / self.unitValue,
        }

        while not self.stopEvent.is_set():
            # Main process
            info["cpu"] = proc.cpu_percent()
            info["status"] = proc.status()
            info["timestamp"] = time.time()

            # Reset children every loop
            children = []
            info["children"] = children

            # Memory
            info["mem"] = proc.memory_info().rss / self.unitValue

            # Retrieve child processes' info (ray workers, etc)
            childProcs = []
            for child in proc.children(recursive=True):
                try:
                    children.append(
                        {
                            "pid": child.pid,
                            "name": child.name(),
                            "cpu": child.cpu_percent(),  # This will always be 0 on first call
                            "mem": child.memory_info().rss / self.unitValue,
                            "status": child.status(),
                        }
                    )
                    childProcs.append(child)
                except psutil.NoSuchProcess:
                    continue

            time.sleep(self.interval)

            # Get the children CPU info after the sleep
            for i, child in enumerate(childProcs):
                children[i]["cpu"] = child.cpu_percent()

            if self.summarize:
                # Snapshot memory usage
                sys = psutil.virtual_memory()

                # Total app memory usage
                info["mem_app"] = sum([p["mem"] for p in children]) + info["mem"]

                # System memory used minus app
                used = sys.used / self.unitValue
                info["mem_used"] = used - info["mem_app"]

                # Remaining available memory
                info["mem_avail"] = sys.available / self.unitValue

            if self.round:
                for key, value in info.items():
                    if "mem" in key:
                        info[key] = round(value, self.round)

                for child in children:
                    child["mem"] = round(child["mem"], self.round)

            self.callback(info)

    def start(self):
        """
        Starts the _track function in a thread
        """
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(
            target=self._track, daemon=True, name=self.__class__.__name__
        )
        self.thread.start()

        # Kill the thread even if stop isn't manually called
        atexit.register(self.stop)

    def is_running(self):
        """
        Checks if there is a thread running
        """
        if self.thread is not None:
            return self.thread.is_alive()
        return False

    def stop(self):
        """
        Sets the stop event to kill any running threads
        """
        self.stopEvent.set()


class FileResources(ResourceTracker):
    """
    Subclass of ResourceTracker with built in file handling

    Parameters
    ----------
    file : str
        Path to a JSONL file to log resource information to
    reset : bool, default=False
        If the file exists, reset it
    """

    def __init__(self, file: str, /, reset: bool = False, **kwargs):
        if "callback" in kwargs:
            raise AttributeError(
                f"{self.__class__.__name__} does not accept a callback parameter"
            )

        super().__init__(callback=self.write, **kwargs)

        self.file = Path(file)
        self.file.parent.mkdir(exist_ok=True, parents=True)

        if reset:
            self.io = open(self.file, "w")
        else:
            self.io = open(self.file, "a")

        atexit.register(self.io.close)

    def write(self, info: dict) -> None:
        """
        Writes the resource information as a JSON object per line

        Parameters
        ----------
        info : dict
            A dictionary containing resource information to log
        """
        data = json.dumps(info) + "\n"

        self.io.write(data)
        self.io.flush()


def stream(file: str, sleep: float = 0.2) -> dict:
    """
    Generator that yields parsed JSONL objects from a growing json file produced by
    FileResources

    Parameters
    ----------
    file : str
        Path to the JSONL file being written to
    sleep : float, default=0.2
        How long to wait (in seconds) between polling for new lines

    Yields
    ------
    dict
        Parsed JSONL object from each line
    """
    with open(file, "r") as f:
        while True:
            line = f.readline()
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Incomplete line or corrupted; wait for next write
                time.sleep(sleep)
