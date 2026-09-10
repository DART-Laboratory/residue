"""`python -m residue ...` — the same entry point as the installed `residue` script, for use
from a checkout with nothing installed."""

from residue.cli import main

if __name__ == "__main__":
    main()
