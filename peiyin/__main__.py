"""Entry point: `python -m peiyin` launches the web interface."""

import argparse
import sys


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="peiyin",
        description="Dub authorized media into Mandarin for private language learning.")
    p.add_argument("--no-browser", action="store_true",
                   help="do not open a browser window on start")
    p.add_argument("--version", action="store_true", help="print version and exit")
    args = p.parse_args(argv)

    if args.version:
        from peiyin import __version__
        print(__version__)
        return 0

    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        p.error("Use Python 3.10–3.12 (3.12 recommended).")

    from peiyin.ui import launch_ui
    launch_ui(open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
