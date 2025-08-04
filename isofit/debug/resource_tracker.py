import atexit
import json
import threading
import time
from pathlib import Path
from types import FunctionType, MethodType

import psutil


class ResourceTracker:
    """
    Tracks system resources

    Parameters
    ----------
    callback : function
        Function to call on each resource refresh. Signature must be:

            function(info : dict) -> None

            where the info dict consists of:
                pid : int
                    Main process ID
                name : str
                    Main process name
                mem : float
                    Main process memory used
                cpu : float
                    Main process CPU percentage
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
                mem_used : float
                    Memory in use by the system, excluding the app
                mem_free : float
                    Remaining free memory
            Total memory of the system is the sum([mem, mem_used, mem_free])
    interval : int | float, default=2
        Interval frequency in seconds to check resources
    round : int | bool, default=2
        Round the memory variables to this many decimals. Set to False or 0 to disable
        True will be set to 1
    summarize : bool, default=True
        Includes summary statistics such as the sum of all children
    """

    thread = None

    def __init__(
        self,
        callback: FunctionType | MethodType,
        interval: float = 2,
        round: float = 2,
        summarize: bool = True,
    ):
        if not isinstance(callback, (FunctionType, MethodType)):
            raise AttributeError(
                f"The callback parameter must be a function, got {type(callback)} instead"
            )

        if not isinstance(interval, (int, float)):
            raise AttributeError(f"The interval parameter must be an integer")
        if interval <= 0:
            raise AttributeError(f"The interval parameter must be greater than 0")

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

        sys = psutil.virtual_memory()
        self.total = sys.total / self.unitValue

    def _track(self):
        """
        Memory tracker intended to be set in a thread
        """
        proc = psutil.Process()
        info = {"pid": proc.pid, "name": proc.name()}

        while not self.stopEvent.is_set():
            # Main process
            info["cpu"] = proc.cpu_percent()
            info["status"] = proc.status()

            # Reset children every loop
            children = []
            info["children"] = children

            # Memory
            sys = psutil.virtual_memory()
            info["mem"] = proc.memory_info().rss / self.unitValue

            # Retrieve child processes' info (ray workers, etc)
            for child in proc.children(recursive=True):
                try:
                    children.append(
                        {
                            "pid": child.pid,
                            "name": child.name(),
                            "cpu": child.cpu_percent(),
                            "memory": child.memory_info().rss / self.unitValue,
                            "status": child.status(),
                        }
                    )
                except psutil.NoSuchProcess:
                    continue

            if self.summarize:
                # Total app memory usage
                app = sum([p["mem"] for p in children]) + info["mem"]
                info["mem_used"] = sys.used / self.unitValue - app

                # Remaining free memory
                info["mem_free"] = self.total - info["mem_used"]

            if self.round:
                info["mem"] = round(info["mem"], self.round)

                for child in children:
                    child["mem"] = round(child["mem"], self.round)

                if self.summarize:
                    info["mem_used"] = round(info["mem_used"], self.round)
                    info["mem_free"] = round(info["mem_free"], self.round)

            self.callback(info)

            time.sleep(self.interval)

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
        Path to a JSON file to log resource information to
    reset : bool, default=False
        If the file exists, reset it
    """

    def __init__(self, file: str, reset: bool = False, *args, **kwargs):
        super().__init__(*args, callback=self.write, **kwargs)

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
            A dictionary containing resource information to log.
        """
        info["ts"] = time.time()
        data = json.dumps(info) + "\n"

        self.io.write(data)
        self.io.flush()


def stream(file: str, sleep: float = 0.2) -> dict:
    """
    Generator that yields parsed JSONL objects from a growing json file produced by
    FileMemory

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
        # Move to the end of file initially
        f.seek(0, 2)

        while True:
            line = f.readline()
            if not line:
                time.sleep(sleep)
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Incomplete line or corrupted; wait for next write
                time.sleep(sleep)
