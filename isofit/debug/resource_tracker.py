import atexit
import json
import os
import threading
import time
from pathlib import Path
from typing import Callable, Literal
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

            the first info dict will contain the non-changing values:
                cores : int
                    Number of CPU cores in use. See the 'cores' parameter for more
                mem_total : float
                    Total memory of the system
                mem_unit : str
                    Unit label that the memory values are in
                mem_value : float
                    The value used to convert the bytes to the mem_unit. This may be
                    used to reverse the conversion
                poll_interval : float
                    Resource polling interval
                timestamp : float
                    The start timestamp of the resource tracker via time.time()

            all calls afterwards will consist of:
                pid : int
                    Main process ID
                name : str
                    Main process name
                mem : float
                    Main process memory used
                cpu : float
                    Main process CPU percentage over the interval
                timestamp : float
                    Timestamp of the resource record via time.time()
                status : str
                    Main process status, eg. 'running', 'sleeping'
                children : list[dict]
                    Information of child processes:
                        pid : int
                            Child process ID
                        name : str
                            Child process name
                        mem : float
                            Child process memory used
                        cpu : float
                            Child process CPU percentage over the interval
                        status : str
                            Child process status, eg. 'running', 'sleeping'

            if summarize is enabled, these will also be included:
                mem_app : float
                    Total memory of the main process + children
                mem_used : float
                    Memory in use by the system, excluding the app
                mem_avail : float
                    Remaining available memory, defined as free + reclaimable
                cpu_avg : float
                    Average CPU percentage calculated as: sum(main + children) / cores

            Total memory of the system is the sum([mem_app, mem_used, mem_free])
    interval : int | float, default=2
        Interval frequency in seconds to check resources
        Must be greater than 0. Values less than 0.1 risk high CPU usage and skewing
        polled results
        The CPU usage is calculated as the percentage of CPU used over this interval
    units : tuple[str, float], default=("GB", 1024**3)
        Units to convert the memory values to. Must be in the form of (str, float)
        where the float is used to divide the bytes values that psutil returns
        Some possible conversions:
            - ('b', 1/8)      # Convert to bits (multiply by 8)
            - ('B', 1)        # No conversion, leave as the default bytes
            - ('KB', 1024)    # Kilobytes
            - ('MB', 1024**2) # Megabytes
            - ('GB', 1024**3) # Gigabytes, default
    cores : int | 'all', default=1
        Number of cores being used by the source program. This is used for calculating
        the average CPU percentage. Can be passed 'all' to retrieve the os.cpu_count()
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
        units: tuple = ("GB", 1024**3),
        cores: int | Literal["all"] = None,
        round: bool | int = 2,
        summarize: bool = True,
        allow_unsafe: bool = False,
    ):
        # Check the 'callback' parameter
        if not callable(callback):
            raise AttributeError(f"The 'callback' parameter must be a callable")

        # Check the 'interval' parameter
        if not isinstance(interval, (int, float)):
            raise AttributeError(f"The 'interval' parameter must be an integer")
        if interval <= 0:
            raise AttributeError(f"The 'interval' parameter must be greater than 0")
        if interval < 0.1:
            msg = "High CPU usage risk with an interval less than 0.1"
            if allow_unsafe:
                warn(msg)
            else:
                msg += " - If this is intended, set allow_unsafe=True"
                raise ValueError(msg)

        # Check the 'round' parameter
        if isinstance(round, bool):
            round = int(round)
        if not isinstance(round, int):
            raise AttributeError(f"The 'round' parameter must be an integer")

        # Check the 'cores' parameter
        if isinstance(cores, str):
            if cores == "all":
                cores = os.cpu_count()
            else:
                raise AttributeError(
                    "The 'cores' parameter must be either an int or 'all'"
                )
        elif isinstance(cores, int):
            if cores <= 0:
                raise AttributeError("The 'cores' parameter must be greater than 0")
        else:
            raise AttributeError("The 'cores' parameter must be either an int or 'all'")

        # Check the 'units' parameter
        if (
            not isinstance(units, (tuple, list))
            or len(units) != 2
            or not isinstance(units[0], str)
            or not isinstance(units[1], (int, float))
        ):
            raise AttributeError(
                "The 'units' parameter must be a two item tuple consisting of (str label, float divisor)"
            )
        if units[1] == 0:
            raise AttributeError(
                "The divisor in the 'units' parameter must not be zero"
            )

        self.callback = callback
        self.interval = interval
        self.cores = cores
        self.round = round
        self.summarize = summarize
        self.unitLabel, self.unitValue = units

    def _track(self):
        """
        System resource tracker intended to be set in a thread
        """
        sys = psutil.virtual_memory()
        proc = psutil.Process()
        info = {
            "pid": proc.pid,
            "name": proc.name(),
        }

        # Record non-changing values as the first line
        self.callback(
            {
                "cores": self.cores,
                "mem_unit": self.unitLabel,
                "mem_value": self.unitValue,
                "mem_total": sys.total / self.unitValue,
                "poll_interval": self.interval,
                "timestamp": time.time(),
            }
        )

        # Establish a baseline for CPU usage
        proc.cpu_percent()
        psutil.cpu_percent(percpu=True)

        while not self.stopEvent.is_set():
            # Establish baselines for child processes
            childProcs = []
            for child in proc.children(recursive=True):
                try:
                    child.cpu_percent()
                    childProcs.append(child)
                except psutil.NoSuchProcess:
                    continue

            # CPU usage is calculated as the percentage used over this interval
            time.sleep(self.interval)

            # Main process
            info["mem"] = proc.memory_info().rss / self.unitValue
            info["status"] = proc.status()
            info["timestamp"] = time.time()

            # Get the system CPU usage per core
            info["cpu"] = proc.cpu_percent()
            info["sys_cpu"] = psutil.cpu_percent(percpu=True)

            # Reset children every loop
            children = []
            info["children"] = children

            # Retrieve child processes' info (ray workers, etc)
            for child in childProcs:
                try:
                    children.append(
                        {
                            "pid": child.pid,
                            "name": child.name(),
                            "cpu": child.cpu_percent(),
                            "mem": child.memory_info().rss / self.unitValue,
                            "status": child.status(),
                        }
                    )
                except psutil.NoSuchProcess:
                    continue

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

                # Average CPU usage
                info["cpu_avg"] = sum([p["cpu"] for p in children]) + info["cpu"]
                info["cpu_avg"] /= self.cores

            if self.round:
                for key, value in info.items():
                    if "mem" in key or "cpu" in key:
                        info[key] = round(value, self.round)

                for child in children:
                    child["mem"] = round(child["mem"], self.round)
                    child["cpu"] = round(child["cpu"], self.round)

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
