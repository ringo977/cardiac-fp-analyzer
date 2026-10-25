"""Build the stand-alone NWB converter with PyInstaller — Windows (.exe), macOS (.app in a zip), Linux (folder).

    pip install -e ".[gui,nwb]" pyinstaller
    python packaging/build_nwb_converter.py

PyInstaller does not cross-compile: run this on each system (the GitHub workflow ``build-nwb-converter.yml``
does it on a Windows and a macOS runner and publishes the results).  Output in ``dist/``.  The build ends by
running the program with ``--self-test`` (converts a synthetic recording and reads it back), so a build that
lacks pynwb's schema files, h5py or the time-zone database fails here and not on the operator's computer.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "NWB Converter"


def build() -> Path:
    import PyInstaller.__main__ as pyi
    system = platform.system()
    args = [
        str(ROOT / "packaging" / "nwb_converter_app.py"),
        "--name", NAME, "--noconfirm", "--clean", "--windowed",
        "--paths", str(ROOT),
        "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build" / "pyinstaller"),
        "--specpath", str(ROOT / "build" / "pyinstaller"),
        # pynwb/hdmf read their schema (.yaml/.json) at run time; h5py loads its hdf5 libraries dynamically
        "--collect-all", "pynwb", "--collect-all", "hdmf", "--collect-all", "h5py", "--collect-data", "tzdata",
        "--hidden-import", "pyside_app.nwb_converter", "--hidden-import", "cardiac_fp_analyzer.translations.it",
        # not needed by the converter: keep the package small
        "--exclude-module", "pyqtgraph", "--exclude-module", "tkinter",
        "--exclude-module", "IPython", "--exclude-module", "pytest",
    ]
    if system == "Windows":
        args += ["--onefile"]            # one .exe to copy around
    elif system == "Darwin":
        args += ["--onedir", "--osx-bundle-identifier", "org.cardiacfp.nwbconverter"]
    else:
        args += ["--onedir"]
    pyi.run(args)
    dist = ROOT / "dist"
    if system == "Windows":
        return dist / f"{NAME}.exe"
    if system == "Darwin":
        app = dist / f"{NAME}.app"
        zipped = shutil.make_archive(str(dist / f"{NAME.replace(' ', '-')}-macOS"), "zip", dist, app.name)
        print("zipped:", zipped)
        return app
    return dist / NAME


def smoke_test(artifact: Path) -> None:
    exe = {"Windows": artifact, "Darwin": artifact / "Contents" / "MacOS" / NAME}.get(platform.system(), artifact / NAME)
    out = subprocess.run([str(exe), "--self-test"], capture_output=True, text=True, timeout=300)
    print(out.stdout, out.stderr)
    if out.returncode != 0 or "self-test OK" not in out.stdout:
        sys.exit(f"self-test of the packaged program failed (exit {out.returncode})")


if __name__ == "__main__":
    smoke_test(build())
