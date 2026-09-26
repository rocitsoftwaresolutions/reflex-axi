"""Keep the version probe free of the command graph and third-party imports."""

import sys

from . import __version__


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] in {"-v", "-V", "--version"}:
        print(__version__)
        return
    from .cli import run

    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
