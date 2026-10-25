"""Entry script of the packaged NWB converter (PyInstaller starts here)."""
from pyside_app.nwb_converter import main

if __name__ == "__main__":
    raise SystemExit(main())
