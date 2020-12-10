import sys
import subprocess

commands = ["-m pip install -r gesticulator/requirements.txt",
            "-m pip install bert-embedding==1.0.1",
            "-m pip install numpy==1.18.2",
            "-m pip install -e .",
            "-m pip install -e gesticulator/visualization"] 

for cmd in commands:
    subprocess.check_call([sys.executable] + cmd.split())
