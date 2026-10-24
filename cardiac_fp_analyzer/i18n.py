"""Language of the texts meant for the operator (v3.19, Oct 2026) — no Qt.

English is the source language of every text in the code.  ``tr(text)`` returns ``text`` in the current
language, looked up in the catalog of ``cardiac_fp_analyzer.translations`` (``it.py`` for Italian: a dict
English → Italian), or ``text`` itself when the catalog has no entry or the language is English.  Texts
with values are templates: ``tr('{0} beats included').format(n)`` — the placeholders stay in the
translation.  ``set_language('it')`` switches the whole process (package and GUI); the default is
English.  The GUI reads the operator's choice from its settings and calls ``set_language`` at start-up
(``pyside_app.i18n``).

Machine keys never pass through here: the sidecar statuses (``da_rivedere`` …), the beat sources
(``aggiunto`` …), the quality levels (``verde`` …) and the dict keys of the results stay what they are;
only their display words are translated.
"""
from __future__ import annotations

import importlib

LANGUAGES = ('en', 'it')
LANGUAGE_NAMES = {'en': 'English', 'it': 'Italiano'}
DEFAULT_LANGUAGE = 'en'

_state = {'language': DEFAULT_LANGUAGE, 'catalog': {}}
_catalogs: dict[str, dict] = {}


def normalise(code) -> str:
    """``'it'``, ``'it_IT'``, ``'it-IT'``, ``'Italian'`` → ``'it'``; anything unknown → English."""
    c = str(code or '').strip().lower().replace('-', '_')
    c = c.split('_')[0].split('.')[0]
    if c.startswith('ital'):
        c = 'it'
    return c if c in LANGUAGES else DEFAULT_LANGUAGE


def catalog(code) -> dict:
    """The catalog (English → language) of ``code``; empty for English."""
    code = normalise(code)
    if code == DEFAULT_LANGUAGE:
        return {}
    if code not in _catalogs:
        mod = importlib.import_module(f'cardiac_fp_analyzer.translations.{code}')
        _catalogs[code] = dict(mod.CATALOG)
    return _catalogs[code]


def set_language(code) -> str:
    """Switch the language of every text produced from now on; returns the code in use."""
    code = normalise(code)
    _state['language'] = code
    _state['catalog'] = catalog(code)
    return code


def language() -> str:
    return _state['language']


def tr(text: str) -> str:
    """``text`` (English source) in the current language."""
    cat = _state['catalog']
    if not cat:
        return text
    return cat.get(text, text)


__all__ = ['tr', 'set_language', 'language', 'catalog', 'normalise', 'LANGUAGES', 'LANGUAGE_NAMES',
           'DEFAULT_LANGUAGE']
