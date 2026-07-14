# Frontend Test Harness — Design

Status: **phases 1–2 implemented** (2026-07-13); phase 3 (Galata) remains
deferred. Written following the code audit (`AUDIT-2026-07-12.md`): three
rounds of substantial frontend fixes (comm routing, listener lifecycle,
path migration, notice rendering) were verified only by compile + lint
because `src/` had zero tests.

Implementation notes vs. this design:

- The `@jupyterlab/testing` preset's `transformIgnorePatterns` had to be
  overridden to also whitelist `@jupyterlab/`, `@lumino/`, and
  `@jupyter/` themselves (inside Lab's monorepo they resolve to source;
  downstream they arrive as ESM) — same override the official extension
  template uses. See `jest.config.js` + `babel.config.js`.
- The parity fixture immediately caught real drift: Python's
  `ReasonType` has 11 values; the TS `BackendReasonType` union listed 8.
  The three violation-derived reasons (`no_read_and_write`,
  `write_before_read`, `unrecoverable_mutation`) were falling into
  reasonformat's generic default branch. Union + table extended.
- The `IExecutionSignals` seam lives in `cellhighlighter.ts` (exported;
  `executionhook.ts` already imports from there, avoiding a cycle).

## 1. Goals and non-goals

**Goals**

1. Lock in the audit-era bug fixes as executable regressions (comm bound to
   the owning panel, listener disposal, `cell_edited` flush semantics,
   pending-command buffering, pathChanged migration, HTML escaping,
   post-run metadata wait).
2. Unit-test the pure logic that today is only exercised indirectly
   (`cellindexutils`, the TS `▷` relation in `types.ts`, `reasonformat.ts`,
   `escapeHtml`).
3. Keep the TS `▷` conflict relation and reason vocabulary provably in
   sync with the Python side (this repo's spec-sync culture, applied
   across the language boundary).
4. Run fast enough to sit in the default developer loop (`jlpm test`
   under ~30 s) with no Jupyter server or kernel required.

**Non-goals (for the initial harness)**

- Pixel/visual testing of the React panels (`metadatapanel.tsx`,
  `dependenciespanel.tsx` with ReactFlow). Layer 3 covers their behavior
  end-to-end; component-level React testing is deferred.
- Testing JupyterLab itself (session lifecycle, Y.js sync). We test OUR
  managers' reactions to signals, not Lab's emission of them — that is
  Layer 3's job against a real Lab.

## 2. Current state and constraints

- `package.json`: JupyterLab `^4.0.0`, TypeScript `~5.4`, no jest, no
  test script, no CI workflow in the repo. Node 22 locally.
- The audit refactors are what MAKE this testable now: listeners are
  named and stored in per-panel/per-cell maps, the comm handler is bound
  to its owning panel at connect time, and reason formatting is a pure
  table in `reasonformat.ts`. Before those changes, most of this design
  would have required refactoring first.
- Known pain point: JupyterLab 4 / Lumino 2 / yjs packages ship ESM;
  naïve jest configs fail on `transformIgnorePatterns`. This is the main
  reason to use the official `@jupyterlab/testing` preset rather than a
  hand-rolled jest setup.

## 3. Architecture: three layers

```
Layer 1  Pure unit tests            jest, no fakes           ~ms each
Layer 2  Manager/lifecycle tests    jest + narrow fakes      ~ms each
Layer 3  Golden-path E2E            Galata (Playwright)      ~s each, opt-in
```

Layers 1–2 run in `jlpm test` with no external processes. Layer 3 needs a
running JupyterLab with `flowbook_kernel` and is a separate opt-in target
(local + eventual CI nightly), not part of the default loop.

## 4. Layer 1 — Jest infrastructure

### Packages (devDependencies)

```
@jupyterlab/testing   ^4.x     (jest base config; version-matched to @jupyterlab/*)
jest                  ^29
@types/jest           ^29
```

`@jupyterlab/testing` pins its own jest/babel plumbing and, critically,
ships the `transformIgnorePatterns`/moduleNameMapper set that makes
`@jupyterlab/*`, `@lumino/*`, `@jupyter/ydoc`, `yjs`, `nanoid` etc.
importable under jest. Keep its major version locked to the
`@jupyterlab/application` major.

### Config

`jest.config.js` at repo root:

```js
const jestJupyterLab = require('@jupyterlab/testing');
const baseConfig = jestJupyterLab.jestConfig();

module.exports = {
  ...baseConfig,
  testRegex: 'src/.*/tests/.*\\.spec\\.ts[x]?$',
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/_archived/**',
    '!src/**/tests/**'
  ],
  coveragePathIgnorePatterns: ['/node_modules/', '/lib/']
};
```

### Layout — mirror the Python convention

The repo rule for Python is "tests live in a `tests/` subdirectory of the
package they test". Mirror it:

```
src/tests/                      # cellindexutils, handler
src/shared/tests/               # kerneldetection
src/flowbook/tests/             # the plugin managers
src/flowbook/tests/fixtures/    # shared JSON fixtures (see §6)
src/flowbook/tests/testutils.ts # narrow fakes (see §5)
```

Spec files: `<module>.spec.ts` (e.g. `src/flowbook/tests/executionhook.spec.ts`).

### Build/lint integration

- `tsconfig.json`: add `"src/**/tests/**"` to `exclude` so `jlpm build:lib`
  does not compile specs into `lib/` (jest compiles them itself via the
  preset). Type-checking of specs happens through jest and through a new
  `tsconfig.test.json` (`extends` base, includes tests) used by
  `jlpm test:types` if wanted — optional.
- `package.json` scripts:
  ```json
  "test": "jest",
  "test:cov": "jest --coverage",
  "watch:test": "jest --watch"
  ```
- ESLint: add a `src/**/tests/**` override with `env: { jest: true }`.
- `eslintIgnore`/prettier: no change needed (specs should be lint-clean).

## 5. Layer 2 — narrow fakes, not full mocks

### Why hand-rolled fakes

`@jupyterlab/testing` also exposes heavyweight mocks (mock services,
`NBTestUtils`), but they drag in the full services stack, need jsdom
plumbing, and obscure the property we most want to assert: **connection
hygiene**. Our managers consume narrow structural slices of the Lab API,
and TypeScript's structural typing means a small fake object typed as the
interface is accepted directly. More importantly, fakes we own can COUNT
connections — turning "no listener leaks" from a hope into an assertion.

Use the official mocks only if a specific test genuinely needs deep Lab
behavior; expected to be rare.

### The fakes (`src/flowbook/tests/testutils.ts`)

All signals are REAL Lumino `Signal` instances (no DOM, no async), wrapped
in a `CountingSignal<T, U>` that records `connect`/`disconnect` calls:

```ts
class CountingSignal<T, U> {
  readonly signal: Signal<T, U>;
  connects = 0;
  disconnects = 0;
  get live(): number {
    return this.connects - this.disconnects;
  }
  emit(args: U): void;
}
```

Built on that:

| Fake                 | Implements (structurally)                                                                                                           | Notes                                                                                     |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------- |
| `FakeCellModel`      | `id`, `type`, `sharedModel.{getSource,setSource,changed}`, `getMetadata`/`setMetadata`/`deleteMetadata`, `outputs` (for code cells) | metadata is a plain Map; `sharedModel.changed` emits `CellChange`-shaped args             |
| `FakeOutputsModel`   | `length`, `get(i).toJSON()`, `fromJSON(list)`                                                                                       | backs notice-manager tests                                                                |
| `FakeComm`           | `send`, `open`, `close`, `onMsg`, `onClose`, `isDisposed`                                                                           | records `sent: unknown[]`; test can invoke `onMsg(msg)` to deliver kernel→client messages |
| `FakeKernel`         | `createComm(target)` → new `FakeComm` (all retained in `comms: FakeComm[]`)                                                         | asserting "superseded comm closed" = `comms[0].closed === true`                           |
| `FakeSessionContext` | `session.kernel`, `kernelChanged`, `statusChanged`, `ready` (resolved promise)                                                      | `setKernel(k)` emits kernelChanged; `setStatus('restarting'                               | 'idle')` emits statusChanged |
| `FakeContext`        | `path`, `pathChanged`, `saveState`                                                                                                  | `rename(newPath)` emits pathChanged                                                       |
| `FakePanel`          | `content` (widgets list + `model.cells.changed` + `activeCell`), `sessionContext`, `context`, `isDisposed`, `disposed`              | `disposeNow()` flips isDisposed and emits disposed                                        |
| `FakeTracker`        | `currentWidget`, `currentChanged`, `activeCell`, `activeCellChanged`                                                                | `setCurrent(panel)` emits currentChanged                                                  |

Roughly 250–350 lines total; they live in one file and evolve with the
managers.

### One production seam: execution signals

`ReproducibilityExecutionHookManager` and `ReproducibilityCellHighlighter`
connect to the STATIC `NotebookActions.executed` / `executionScheduled`
signals, which cannot be emitted from outside Lab. Add a minimal injection
seam (no behavior change):

```ts
export interface IExecutionSignals {
  executed: ISignal<unknown, { notebook: Notebook; cell: Cell }>;
  executionScheduled: ISignal<unknown, { notebook: Notebook; cell: Cell }>;
}
// constructor(tracker, highlighter, signals: IExecutionSignals = NotebookActions)
```

Production call sites pass nothing; tests pass a `FakeExecutionSignals`
with emittable signals. This is the only production change the harness
requires. (Alternative — invoking private handlers via `(hook as any)` —
is rejected: it skips the connect/disconnect wiring, which is exactly
what the leak tests must cover.)

### Timers

`cell_edited` debounce (1 s), `waitForFlowbookMetadata` polling (50 ms /
2 s cap): jest fake timers (`jest.useFakeTimers()` +
`jest.advanceTimersByTimeAsync`). No real sleeps anywhere in Layers 1–2.

## 6. Cross-language parity fixtures

Two vocabularies exist in both Python and TypeScript and have already
drifted once each (the `'copy'` attr; reason-message wording):

1. **The ▷ conflict relation** — `write_conflicts_read` in
   `flowbook/kernel/locations.py` vs `writeConflictsRead` in
   `src/flowbook/types.ts`.
2. **Backend reason types** — `ReasonType` in `flowbook/kernel/models.py`
   vs the case handling in `src/flowbook/reasonformat.ts`.

Mechanism (same pattern as the existing `tool_catalog.md` drift guard):

- `src/flowbook/tests/fixtures/conflict_cases.json` — a hand-written,
  exhaustive case table for the 5×5 ▷ matrix (including qualifier
  match/mismatch cases): `{ write: {...}, read: {...}, conflicts: bool }`.
- `src/flowbook/tests/fixtures/reason_types.json` — the list of backend
  reason type strings.
- Jest side: table-driven spec asserting `writeConflictsRead` over every
  case; a spec asserting `reasonformat.ts` produces a non-default message
  for every listed reason type.
- Python side: one new pytest module
  (`flowbook/kernel/tests/test_frontend_parity.py`) loading the same
  files by repo-relative path (skip with a clear message if the `src/`
  tree is absent, i.e. installed-package runs) and asserting
  `write_conflicts_read` agrees case-by-case and `ReasonType` values ⊆
  fixture list.

Either side drifting now fails a test on that side.

## 7. Layer 3 — Galata end-to-end (deferred, scoped)

`@jupyterlab/galata` (Playwright) drives a real JupyterLab with the real
`flowbook_kernel`. This is the only layer that tests the comm protocol,
Y.js, and kernel behavior together.

- Separate directory `ui-tests/` (galata's conventional layout) with its
  own `package.json`; `jlpm test:e2e` spins `jupyter lab` via galata's
  config. Requires the `flowbook` conda env.
- **Golden paths only** (5–7 tests, each asserting user-visible state):
  1. Run `x=1` then `y=x+1`; re-run cell 1 with a new value → cell 2 gets
     the stale class + notice text.
  2. Edit an executed cell, wait >1 s → stale marker appears (kernel
     received `cell_edited`).
  3. `x = x + 1` cell → violation notice rendered (and HTML-escaped
     column-name case).
  4. Switch kernel to python3 → flowbook UI deactivates; switch back →
     reactivates, no duplicated notices.
  5. Rename the notebook → staleness state survives.
  6. Run-next-stale toolbar button walks cells in order.
- Policy: Layer 3 failures never block the fast loop; it is a separate
  target and (if CI is added) a nightly job. Flaky tests get quarantined
  or deleted — the value is smoke coverage, not breadth.

## 8. Test inventory (initial), mapped to audit findings

### Phase 1 — infrastructure + pure units (Layer 1)

| Spec                     | Covers                                                                                                          | Audit ref            |
| ------------------------ | --------------------------------------------------------------------------------------------------------------- | -------------------- |
| `cellindexutils.spec.ts` | alpha↔index round-trip, >26 columns, invalid input                                                              | —                    |
| `types.spec.ts`          | ▷ matrix via fixture; `findConflictingReads`; `formatReadLoc`; `escapeHtml` (incl. `<img onerror>` column name) | item 7 escaping      |
| `reasonformat.spec.ts`   | every reason type × {loc, causer, direction above/below, deleted cell}; both views agree on content             | low-sev dedup        |
| `protocol.spec.ts`       | client/kernel message shapes match `flowbook/kernel/protocol.py` expectations (field presence)                  | audit protocol check |

### Phase 2 — lifecycle regressions (Layer 2)

| Spec                                  | Key assertions                                                                                                                                                                         | Audit ref                  |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| `executionhook.comm.spec.ts`          | msg delivered after tab switch applies to OWNING panel; kernel switch closes superseded comm (`comms[0].closed`); `onClose` clears ref; buffer: queue→flush order, 100-cap drop-oldest | F1, F2, buffering          |
| `executionhook.edits.spec.ts`         | debounce fires at 1 s; scheduling FLUSHES (message sent before run); dispose flushes; external (comm-delivered) execution enables edit reporting                                       | debounce, `_executedCells` |
| `executionhook.lifecycle.spec.ts`     | N× setup on same panel → `live === 1` per signal (CountingSignal); dispose → `live === 0` and emissions are no-ops                                                                     | F3                         |
| `cellhighlighter.spec.ts`             | monitor handlers connect once; dispose disconnects; zombie emission after dispose mutates nothing; pathChanged migrates all five maps (old key absent, state preserved)                | F3, rename                 |
| `stalenessmanager.spec.ts`            | restart clears; dispose disconnects + `Signal.clearData`                                                                                                                               | item 3                     |
| `kerneldetection.spec.ts`             | panel-identity keying: rename + reopen old path ≠ shared entry                                                                                                                         | lifecycle batch            |
| `noticemanagers.spec.ts`              | staleness/violation notice built with escaped HTML; dedup (same text → no rewrite); removal; violation suppresses staleness notice; FakeOutputsModel round-trip                        | escaping, notices          |
| `waitformetadata.spec.ts`             | resolves on execution_seq bump; first-appearance; 2 s timeout                                                                                                                          | metadata race              |
| `plugin.activation.spec.ts` (stretch) | null widget deactivates; second flowbook notebook gets per-notebook setup                                                                                                              | lifecycle batch            |

### Phase 3 — Galata (Layer 3)

Golden paths listed in §7.

## 9. Phasing and effort

| Phase | Content                                                                                          | Estimate                      |
| ----- | ------------------------------------------------------------------------------------------------ | ----------------------------- |
| 1     | jest infra, scripts, tsconfig/eslint wiring, Layer-1 specs, parity fixtures + Python parity test | ~½ day                        |
| 2     | `testutils.ts` fakes, execution-signals seam, Layer-2 specs                                      | 1–2 days                      |
| 3     | galata scaffold + golden paths                                                                   | ~1 day, deferred until wanted |

Acceptance for phases 1–2: `jlpm test` green in <30 s, coverage report
produced, the parity pytest green, `jlpm build`/`lint:check` unaffected.

## 10. Risks and mitigations

- **ESM/transform breakage** is the classic failure mode: mitigated by
  using the `@jupyterlab/testing` preset verbatim and version-locking it
  to the Lab major. If a package still slips through, extend
  `transformIgnorePatterns` in one place (`jest.config.js`).
- **Fake drift** (fakes diverge from real Lab behavior): fakes are typed
  against the real interfaces (`NotebookPanel`-shaped structural types),
  so API changes surface as compile errors in `testutils.ts`; behavioral
  drift is what Layer 3 exists to catch.
- **Version skew** on Lab upgrades: bump `@jupyterlab/testing` in the
  same commit as `@jupyterlab/*`; the preset is the only jest coupling.
- **Private-member testing**: specs may read private state via bracket
  access for assertions (`hook['_editTimers']`) but must never CALL
  private methods to simulate events — events go through the fakes'
  signals so the wiring is what's tested. (One documented exception:
  none currently.)

## 11. CLAUDE.md updates once implemented

- Testing section: add `jlpm test` (frontend unit) beside `pytest
flowbook/`, and the `src/**/tests/` convention.
- Note the parity fixtures and that adding a ReasonType or ▷ rule
  requires updating `src/flowbook/tests/fixtures/`.
