#!/usr/bin/env python3
import argparse
import subprocess
import sys


def build_command(args, remaining):
    if args.no_conda:
        prefix = []
    else:
        prefix = ["conda", "run", "--no-capture-output", "-n", args.env]

    only_kwargs = all(a.startswith("-") for a in remaining)

    if not remaining or only_kwargs:
        # No args or only keyword/flag args -> Jupyter
        cmd = prefix + ["jupyter", "lab"] + remaining
    elif remaining[0].endswith(".py") or remaining[0] in ['-c', '-m']:
        # A python script or command
        cmd = prefix + ["python"] + remaining
    else:
        # Forward everything as-is
        cmd = prefix + remaining

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Forward commands into a conda environment (or run them directly).",
        add_help=False,  # let -h/--help pass through to the subprocess unless before --
    )
    parser.add_argument(
        "-n", "--env",
        default="cmcc-rxn",
        help="Conda environment name (default: cmcc-rxn).",
    )
    parser.add_argument(
        "--no-conda",
        action="store_true",
        help="Run the command directly without 'conda run'.",
    )

    # Split our own options from the forwarded command. Everything after the
    # first non-option (or after '--') is forwarded untouched.
    args, remaining = parser.parse_known_args()

    cmd = build_command(args, remaining)

    if not cmd:
        parser.error("Nothing to run.")

    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        sys.stderr.write(f"Command not found: {cmd[0]}\n")
        return 127

'''
./cli.py                          # -> conda run ... jupyter lab
./cli.py --port 9999              # -> conda run ... jupyter lab --port 9999
./cli.py script.py --foo bar      # -> conda run ... python script.py --foo bar
./cli.py mycommand arg1 arg2      # -> conda run ... mycommand arg1 arg2
./cli.py -n other-env script.py   # swap environment
./cli.py --no-conda script.py     # run without conda
'''

if __name__ == "__main__":
    sys.exit(main())