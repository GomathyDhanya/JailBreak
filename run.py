import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def execute(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", default="all", choices=["all", "task1", "task2", "task3", "full-gcg", "train", "data", "check"])
    args = parser.parse_args()
    if args.task == "data":
        execute("data_creation.py")
    elif args.task == "check":
        execute("verify.py")
    elif args.task == "task1":
        execute("task1.py", "--mode", "evaluate-cached")
    elif args.task == "task2":
        execute("task2.py")
    elif args.task == "task3":
        execute("task3.py", "--mode", "saved")
    elif args.task == "full-gcg":
        execute("task1.py", "--mode", "full")
    elif args.task == "train":
        execute("task3.py", "--mode", "train")
    else:
        execute("task1.py", "--mode", "evaluate-cached")
        execute("task2.py")
        execute("task3.py", "--mode", "saved")


if __name__ == "__main__":
    main()
