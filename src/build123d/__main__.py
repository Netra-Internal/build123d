"""Allow ``python -m build123d probe …`` as a fallback to the ``b123d`` script."""

from build123d.cli import console_main

if __name__ == "__main__":
    console_main()
