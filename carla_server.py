import os
import socket
import subprocess
import time
from typing import Callable, Coroutine, Tuple
import nvidia_smi as nvml
import asyncio
from asyncio.subprocess import PIPE, Process
import signal


LOCK_FILE = "/tmp/episode_manager_carla_server_build.lock"


def is_build_locked():
    return os.path.exists(LOCK_FILE)


def lock_build():
    with open(LOCK_FILE, "w") as lock_file:
        lock_file.write("locked")


def unlock_build():
    if is_build_locked():
        os.remove(LOCK_FILE)


def main():
    server = CarlaServer()

    def on_exit(return_code, stdout, stderr):
        print("Server exited with return code: ", return_code)

    host, port, tm_port = server.start_server(on_exit)

    print("Server started on ", host, port, tm_port)

    time.sleep(10)
    server.stop_server()


class CarlaServer:
    def start_server(
        self, on_exit: Callable[[int, str, str], None]
    ) -> Tuple[str, int, int]:
        self.loop = asyncio.get_event_loop()

        current_directory = os.path.abspath(os.path.dirname(__file__))

        lock_build()
        command = f"cd {current_directory} &&  make build-carla"
        subprocess.run(command, shell=True, check=True)
        unlock_build()

        port = find_available_port()
        device = get_gpu_with_most_vram()

        command = f"cd {current_directory} && make run-carla CARLA_SERVER_PORT={port} CARLA_SERVER_GPU_DEVICE={device}"

        self.process, self.future = self.loop.run_until_complete(
            run_subprocess(command, on_exit)
        )

        tm_port = find_available_port()

        time.sleep(10)

        return ("127.0.0.1", port, tm_port)

    def stop_server(self):
        if self.process.returncode is None:
            try:
                print("PID: ", self.process.pid)
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

            except ProcessLookupError:
                print("Process already terminated")
        else:
            print("WAITING FOR PROCESS TO EXIT")
            self.loop.run_until_complete(self.future)

        return


def find_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


async def run_subprocess(
    command, on_process_exit: Callable[[int, str, str], None]
) -> Tuple[Process, asyncio.Task]:
    process = await asyncio.create_subprocess_shell(
        command, stdout=PIPE, stderr=PIPE, preexec_fn=os.setsid
    )

    async def on_exit(process):
        try:
            await asyncio.wait_for(process.communicate(), timeout=None)
            print("Process completed normally")
        except asyncio.TimeoutError:
            print("Process was killed before completion")
        else:
            on_process_exit(process.returncode, process.stdout, process.stderr)

    future = asyncio.ensure_future(on_exit(process))
    return process, future


def get_gpu_with_most_vram():
    nvml.nvmlInit()
    device_count = nvml.nvmlDeviceGetCount()

    max_vram = -1
    max_vram_gpu = -1

    for i in range(device_count):
        gpu = nvml.nvmlDeviceGetHandleByIndex(i)
        info = nvml.nvmlDeviceGetMemoryInfo(gpu)

        if info.free > max_vram:
            max_vram = info.free
            max_vram_gpu = i

    nvml.nvmlShutdown()

    if max_vram_gpu == -1:
        raise ValueError("No GPU devices found.")
    else:
        return max_vram_gpu


if __name__ == "__main__":
    main()
