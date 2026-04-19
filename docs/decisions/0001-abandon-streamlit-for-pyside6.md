# ADR-0001: Abandon Streamlit, migrate UI to PySide6 + PyQtGraph

- **Status:** Proposed (awaiting proof-of-concept validation)
- **Date:** 2026-04-19
- **Deciders:** Marco Rasponi
- **Supersedes:** —
- **Superseded by:** —

## Context

`cardiac-fp-analyzer` ships a Streamlit-based GUI (`app.py`, `ui/*.py`) on
top of the scientific core (`cardiac_fp_analyzer/*.py`). Streamlit has
served the project well for rapid prototyping and internal use, but a
series of UX and architectural limits have accumulated as the tool has
matured toward external and regulatory-adjacent audiences.

Pain points we have hit in practice:

1. **Rerun-on-every-interaction model.** Streamlit re-runs the whole
   script on every widget change. For long signals and multi-tab
   analyses this is wasteful and forces defensive caching. Complex
   interactions (beat editor, click-picking) become awkward.

2. **Broken event model for click picking** (task #70, April 2026).
   `st.plotly_chart(on_select='rerun')` does not forward single-click
   events, and in practice it also fails to forward box-select events
   reliably: `plot_state.selection` stays empty even after a real
   drag-box. Workarounds were attempted: invisible marker overlay,
   different `selection_mode` values, `dragmode='select'`, and finally
   the third-party `streamlit-plotly-events` component — which works in
   isolation but renders the plot inside an iframe with its own theme
   (breaks the dark theme and, for our configuration, failed to render
   data at all). Work is parked on branch `wip/click-picking-ux`.

3. **No native desktop affordances.** No real menus, no keyboard
   shortcuts, no multi-window, no system tray, no drag-drop from
   Finder/Explorer in a first-class way.

4. **Deployment story is poor for non-technical users.** The current
   path (`pip install` + `streamlit run app.py`) does not match the
   "double-click-and-go" expectation of researchers, collaborators, or
   pharma clients. Streamlit-in-a-bundle options exist but always drag
   a browser + a local web server along for the ride.

5. **Licensing / presentability.** For prospective pharma clients, a
   browser-tab tool with Streamlit chrome reads as a prototype, not as
   a regulatory-adjacent product.

Audience trajectory: internal group today → external collaborators
near-term → potentially pharma clients medium-term. The UI needs to
scale up in polish and packaging as that audience widens.

## Decision

Migrate the UI layer from **Streamlit** to **PySide6 + PyQtGraph**.

- **PySide6** is the official LGPL Qt6 binding maintained by the Qt
  Project. LGPL permits distribution of commercial, closed-source
  software — important for a future pharma-client scenario.
- **PyQtGraph** is a scientific plotting library built directly on Qt,
  designed for real-time data visualization. Pan/zoom/click on long
  signals are fluid, full event model is native.
- **Packaging:** PyInstaller for Mac `.app` + Windows `.exe`/`.msi`.
  Apple Developer account (~99 €/yr) for Gatekeeper signing when we
  start distributing outside the group. Code-signing on Windows is a
  later step when the audience justifies it.

The scientific core (`cardiac_fp_analyzer/` — ~7-8 k LOC of algorithms,
I/O, QC, FPD, CDISC export) is **not touched** by this migration. Only
`app.py` + `ui/*.py` are rewritten against the new toolkit.

## Alternatives considered

| Option | Why rejected |
|---|---|
| Stay on Streamlit + workarounds | Core limits are structural (rerun model, event forwarding). Each workaround costs days and leaves brittle code. |
| DearPyGui | Non-native look-and-feel. Acceptable for internal tools, not acceptable when demoing to pharma clients. |
| Flet (Python + Flutter) | Too young as a project for a multi-year roadmap with commercial stakes. |
| Tkinter + matplotlib | Outdated look, matplotlib interactive plots are significantly less fluid than PyQtGraph on long signals. |
| Electron (Python backend + web frontend) | Large memory footprint, double-stack complexity (Python + JS + Node), no benefit over Qt for our use case. |
| NiceGUI / Reflex / Gradio | Still web-based, still a local web server — inherits Streamlit-class deployment friction. |
| Keep Streamlit + add a custom component | Custom Streamlit components are iframes (the `streamlit-plotly-events` experience today proved that). Does not solve the architectural problem. |

## Consequences

### Positive

- Proper event model: single clicks, drags, keyboard shortcuts,
  context menus, multi-window all work natively.
- Much better performance on long signals (no rerun, state persists
  between interactions, PyQtGraph is GPU-aware).
- Native look on each OS. Presentable to external collaborators and,
  eventually, pharma audiences.
- Packaging produces a real installable artifact per platform.
- Qt is a skill that transfers to any future scientific tool.

### Negative / costs

- **Learning curve:** ~3-5 days to internalize Qt's signals/slots
  pattern + layout managers before migration velocity picks up.
- **Migration effort:** ~4 weeks for feature-parity (see roadmap
  below), assuming the PoC validates the approach.
- **Packaging / signing cost:** Apple Developer account (~99 €/yr)
  when distribution begins. Windows code-signing cert is a later
  investment.
- **Dependency footprint grows:** Qt6 runtime ships inside each
  PyInstaller bundle (~60-100 MB per platform). Acceptable for a
  desktop scientific tool.
- **Risk of scope creep:** Migration is a good moment to "also
  refactor X" — must be resisted. In-scope: UI layer rewrite.
  Out-of-scope: algorithms, I/O, QC, FPD pipeline.

### Neutral

- Streamlit code (`app.py`, `ui/*.py`) is kept in git history on branch
  `main` until the PyQt build achieves parity; then it is removed in a
  single commit with a clean deprecation note.

## Roadmap (indicative, to be refined after PoC)

### Phase 0 — Proof of concept (2-3 days)

Branch `feat/pyside-poc`. Deliverables:

- Main window with menu bar.
- CSV loading via `QFileDialog`.
- `pyqtgraph.PlotWidget` rendering the filtered signal.
- Beat-marker overlay with rosso / grigio-chiaro / grigio-scuro / verde
  color coding matching the current Streamlit plot.
- Click-to-add and click-to-remove beat interactions working on a real
  signal (this is the bar Streamlit failed to clear).
- Integration with `cardiac_fp_analyzer.analyze()` so the signal comes
  from the real analysis pipeline, not a fixture.

Gate: Marco runs the PoC on a real signal and decides go / no-go.

### Phase 1 — Core viewer (~Week 1)

- Main window shell + tab widget (Signal / Battiti / Parametri /
  Aritmie, mirroring current Streamlit tabs).
- Signal viewer parity: filtered + raw overlay, beat markers, repol
  peaks, hover tooltips, click-picking (add / remove / inspect modes).
- Session state: result object held in-memory, no Streamlit-style
  reruns; UI updates via Qt signals/slots.

### Phase 2 — Editor + parameters (~Week 2)

- Beat editor: `QTableView` with include-checkbox, sortable columns,
  bulk select / include / exclude actions.
- Template plot + unrepresentativeness banner (mirrors task #61).
- Parameters tab: metrics grid + sortable per-beat table.
- Arrhythmia tab: classifier results + topology visualization.

### Phase 3 — Batch + export (~Week 3)

- Batch-analysis mode: file picker → `QThread` worker → progress bar
  → result summary (does not freeze UI; current Streamlit path blocks
  the script).
- Excel export (xlsxwriter, unchanged backend).
- CDISC SEND `.xpt` export (pyreadstat, unchanged backend).

### Phase 4 — Packaging + polish (~Week 4)

- PyInstaller spec for Mac `.app` + Windows `.exe`.
- Icons, about dialog, version banner.
- Smoke-test install on a clean Mac and a clean Windows VM.
- Draft end-user install instructions.
- Gatekeeper signing decision point (Apple Developer account).

## Open questions to resolve during PoC

1. PyQtGraph vs plotly-in-a-QWebEngineView: PyQtGraph is preferred for
   performance; if any plot type we need is missing, we fall back to
   QWebEngineView for that plot only.
2. Internationalization: current Streamlit UI uses `ui/i18n.py` with a
   `T()` helper. Qt has a first-class i18n system (`QTranslator`,
   `.ts` files). Open question: port the `T()` dict to `.ts` now, or
   keep `T()` temporarily and migrate later.
3. Multi-document model: single-window per analysis, or one main
   window with tabbed analyses. PoC will default to single-window.

## References

- Task #70 — Debug "niente succede" sulla nuova UX click-picking
  (completed as parked; branch `wip/click-picking-ux`).
- Task #69 — Redesign click-picking UX (mode + immediate action).
- Task #67 — Fix click-not-captured bug in plot_signal.
- `ui/display.py` — current Streamlit plot implementation (reference
  for PyQtGraph rewrite).
- `cardiac_fp_analyzer/analyze.py` — scientific entry point, unchanged
  by this migration.
