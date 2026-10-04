"""Build MANUALE_Cardiac_FP_Analyzer.docx from its markdown source.

    python docs/build_manual.py

Needs pandoc (>= 2.9) and python-docx. The markdown file is the source
of truth; this script adds a static index (a TOC field would stay empty
until Word refreshes it), styles the reference document and runs pandoc
with ``--columns=40`` so every pipe table receives explicit column
widths (otherwise tables with short lines collapse in LibreOffice).
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'MANUALE_Cardiac_FP_Analyzer.md'
OUT = HERE / 'MANUALE_Cardiac_FP_Analyzer.docx'


def with_index(text: str) -> str:
    """Insert a static 'Indice' (levels 1–2) before the first H1."""
    body = text.split('\n---\n', 2)[-1] if text.startswith('---') else text
    items = []
    for line in body.split('\n'):
        m = re.match(r'^(#{1,2}) (.+)$', line)
        if m:
            indent = '' if len(m.group(1)) == 1 else '    '
            items.append(f'{indent}- {m.group(2)}')
    index = '# Indice\n\n' + '\n'.join(items) + '\n\n'
    first_h1 = re.search(r'^# ', body, flags=re.M)
    body = body[:first_h1.start()] + index + body[first_h1.start():]
    head = text[:len(text) - len(text.split('\n---\n', 2)[-1])] if text.startswith('---') else ''
    return head + body


def reference_docx(path: Path) -> None:
    from docx import Document
    from docx.shared import Cm, Pt, RGBColor

    subprocess.run(['pandoc', '-o', str(path), '--print-default-data-file', 'reference.docx'], check=True)
    d = Document(str(path))

    def setf(name, size=None, bold=None, color=None, font='Calibri'):
        try:
            s = d.styles[name]
        except KeyError:
            return
        s.font.name = font
        if size:
            s.font.size = Pt(size)
        if bold is not None:
            s.font.bold = bold
        if color:
            s.font.color.rgb = RGBColor.from_string(color)

    for n in ('Normal', 'Body Text', 'First Paragraph', 'Compact'):
        setf(n, 10.5)
    setf('Title', 26, True, '1F3864')
    setf('Subtitle', 13, False, '404040')
    setf('Author', 11)
    setf('Date', 11)
    setf('Heading 1', 16, True, '1F3864')
    setf('Heading 2', 13, True, '2E5597')
    setf('Heading 3', 11.5, True, '2E5597')
    for n in ('Source Code', 'Verbatim Char'):
        setf(n, 9, font='Consolas')
    for sec in d.sections:
        sec.left_margin = sec.right_margin = Cm(2.2)
        sec.top_margin = sec.bottom_margin = Cm(2.0)
    d.save(str(path))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ref = tmp / 'reference.docx'
        reference_docx(ref)
        src = tmp / 'manual.md'
        src.write_text(with_index(SRC.read_text(encoding='utf-8')), encoding='utf-8')
        subprocess.run(['pandoc', str(src), '-o', str(OUT), f'--reference-doc={ref}', '--columns=40'],
                       check=True)
    print(OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
