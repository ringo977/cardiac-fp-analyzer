"""Language of the interface (v3.19).

The operator's choice lives in ``QSettings`` under ``ui/language`` (``'en'`` or ``'it'``; English when
absent).  ``apply_language()`` — called by ``main()`` before any widget exists — switches the package's
catalog (``cardiac_fp_analyzer.i18n.set_language``) and installs Qt's own Italian translations
(``qtbase_it``: standard buttons, file dialogs) so the whole window speaks one language.  A change made
from the menu is saved at once and takes effect at the next start: the widgets set their texts when
they are built.

``tr`` is the package's dictionary lookup (English source → current language) and is what every GUI
module imports in place of ``QObject.tr``.  Number formatting is not touched: ``QLocale()`` keeps the
system locale, so the clipboard copies still use the decimal separator Excel expects on this computer.
"""
from __future__ import annotations

from PySide6.QtCore import QCoreApplication, QLibraryInfo, QLocale, QSettings, QTranslator

from cardiac_fp_analyzer.i18n import (
    DEFAULT_LANGUAGE,
    LANGUAGE_NAMES,
    LANGUAGES,
    language,
    normalise,
    set_language,
    tr,
)

SETTINGS_KEY = 'ui/language'
_translators: list[QTranslator] = []


def configured_language(settings: QSettings | None = None) -> str:
    """The language saved in the settings (``'en'`` when absent or unknown)."""
    value = (settings or QSettings()).value(SETTINGS_KEY, DEFAULT_LANGUAGE)
    code = str(value) if value is not None else DEFAULT_LANGUAGE
    return code if code in LANGUAGES else DEFAULT_LANGUAGE


def save_language(code: str, settings: QSettings | None = None) -> str:
    """Persist the operator's choice; returns the normalised code."""
    code = normalise(code)
    (settings or QSettings()).setValue(SETTINGS_KEY, code)
    return code


def apply_language(app: QCoreApplication | None = None, code: str | None = None) -> str:
    """Switch the package catalog and Qt's translations to ``code`` (default: the saved choice)."""
    code = set_language(code if code is not None else configured_language())
    app = app or QCoreApplication.instance()
    if app is not None:
        for t in _translators:
            app.removeTranslator(t)
        _translators.clear()
        if code != DEFAULT_LANGUAGE:
            path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
            for base in ('qtbase', 'qt'):
                t = QTranslator(app)
                if t.load(QLocale(code), base, '_', path):
                    app.installTranslator(t)
                    _translators.append(t)
    return code


__all__ = ['tr', 'language', 'set_language', 'configured_language', 'save_language', 'apply_language',
           'LANGUAGES', 'LANGUAGE_NAMES', 'DEFAULT_LANGUAGE', 'SETTINGS_KEY']
