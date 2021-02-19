import sys
import subprocess

commands = ["-m pip install -r gesticulator/requirements.txt",
            "-m pip install -e .",
            "-m pip install -e gesticulator/visualization"] 

for cmd in commands:
    subprocess.check_call([sys.executable] + cmd.split())
