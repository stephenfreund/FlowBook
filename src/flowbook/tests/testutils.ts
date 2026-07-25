/**
 * Narrow counting fakes for the frontend test harness.
 *
 * See FRONTEND_TESTING.md §5. The managers consume small structural slices
 * of the JupyterLab API; these fakes implement exactly those slices on top
 * of REAL Lumino Signals — no DOM, no services, no kernels. Because we own
 * them, every signal counts connect/disconnect calls, which turns
 * "no listener leaks" into a direct assertion (`sig.live === 0`).
 *
 * Fakes are passed to production code via `as unknown as <RealType>` in
 * specs; the compile-time contract is that the fake carries every member
 * the code under test actually touches. When Lab's API surface used by a
 * manager grows, the corresponding fake fails at runtime in an obvious way
 * (undefined member) rather than silently.
 */

import { Signal } from '@lumino/signaling';

/**
 * A real Lumino Signal wrapped with connect/disconnect counters.
 */
export class CountingSignal<T, U> {
  readonly raw = new Signal<T, U>(this as unknown as T);
  connects = 0;
  disconnects = 0;

  /** Currently connected slot count (connects - disconnects). */
  get live(): number {
    return this.connects - this.disconnects;
  }

  connect = (
    slot: (sender: T, args: U) => void,
    thisArg?: unknown
  ): boolean => {
    this.connects += 1;
    return this.raw.connect(slot as never, thisArg as never);
  };

  disconnect = (
    slot: (sender: T, args: U) => void,
    thisArg?: unknown
  ): boolean => {
    this.disconnects += 1;
    return this.raw.disconnect(slot as never, thisArg as never);
  };

  emit(args: U): void {
    this.raw.emit(args);
  }
}

// ---------------------------------------------------------------------------
// Cells and outputs
// ---------------------------------------------------------------------------

export interface IFakeOutput {
  output_type: string;
  data?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
}

/** Backs the notice managers: length / get(i).toJSON() / fromJSON(list). */
export class FakeOutputsModel {
  list: IFakeOutput[] = [];

  get length(): number {
    return this.list.length;
  }

  get(i: number): { toJSON(): IFakeOutput } {
    const out = this.list[i];
    return { toJSON: () => out };
  }

  fromJSON(outputs: IFakeOutput[]): void {
    this.list = [...outputs];
  }
}

export class FakeSharedModel {
  readonly changed = new CountingSignal<unknown, unknown>();
  private _source: string;

  constructor(source = '') {
    this._source = source;
  }

  getSource(): string {
    return this._source;
  }

  setSource(src: string): void {
    this._source = src;
    this.changed.emit({ sourceChange: [{ insert: src }] });
  }
}

export class FakeCellModel {
  readonly id: string;
  readonly type: string;
  readonly sharedModel: FakeSharedModel;
  readonly outputs = new FakeOutputsModel();
  executionCount: number | null = null;
  private _metadata = new Map<string, unknown>();

  constructor(id: string, type = 'code', source = '') {
    this.id = id;
    this.type = type;
    this.sharedModel = new FakeSharedModel(source);
  }

  getMetadata(key: string): unknown {
    return this._metadata.get(key);
  }

  setMetadata(key: string, value: unknown): void {
    this._metadata.set(key, value);
  }

  deleteMetadata(key: string): void {
    this._metadata.delete(key);
  }
}

/** A fake Cell widget: model + a real DOM-ish node stand-in. */
export class FakeCell {
  readonly model: FakeCellModel;
  readonly node = {
    classList: {
      _set: new Set<string>(),
      add(c: string) {
        this._set.add(c);
      },
      remove(c: string) {
        this._set.delete(c);
      },
      contains(c: string) {
        return this._set.has(c);
      }
    }
  };

  constructor(id: string, type = 'code', source = '') {
    this.model = new FakeCellModel(id, type, source);
  }
}

// ---------------------------------------------------------------------------
// Kernel and comm
// ---------------------------------------------------------------------------

export class FakeComm {
  readonly targetName: string;
  sent: unknown[] = [];
  opened = false;
  closed = false;
  isDisposed = false;
  onMsg: ((msg: unknown) => void) | null = null;
  onClose: (() => void) | null = null;

  constructor(targetName: string) {
    this.targetName = targetName;
  }

  open(): void {
    this.opened = true;
  }

  send(msg: unknown): void {
    if (this.closed || this.isDisposed) {
      throw new Error('send on closed comm');
    }
    this.sent.push(msg);
  }

  close(): void {
    this.closed = true;
    this.isDisposed = true;
  }

  /** Deliver a kernel→client message through the registered handler. */
  deliver(data: unknown): void {
    if (!this.onMsg) {
      throw new Error('no onMsg handler registered');
    }
    this.onMsg({ content: { data } });
  }
}

export class FakeKernel {
  /** Every comm ever created, in creation order. */
  readonly comms: FakeComm[] = [];

  /** Kernel spec name — consulted by KernelDetector. */
  name: string;

  constructor(name = 'flowbook_kernel') {
    this.name = name;
  }

  createComm(targetName: string): FakeComm {
    const comm = new FakeComm(targetName);
    this.comms.push(comm);
    return comm;
  }

  get lastComm(): FakeComm {
    return this.comms[this.comms.length - 1];
  }
}

// ---------------------------------------------------------------------------
// Session context, document context, panel, tracker
// ---------------------------------------------------------------------------

export class FakeSessionContext {
  readonly kernelChanged = new CountingSignal<unknown, unknown>();
  readonly statusChanged = new CountingSignal<unknown, string>();
  readonly ready: Promise<void> = Promise.resolve();
  private _kernel: FakeKernel | null;

  constructor(kernel: FakeKernel | null = new FakeKernel()) {
    this._kernel = kernel;
  }

  get session(): { kernel: FakeKernel | null } | null {
    return { kernel: this._kernel };
  }

  setKernel(kernel: FakeKernel | null): void {
    this._kernel = kernel;
    this.kernelChanged.emit({});
  }

  setStatus(status: string): void {
    this.statusChanged.emit(status);
  }
}

export class FakeContext {
  readonly pathChanged = new CountingSignal<unknown, string>();
  private _path: string;

  constructor(path: string) {
    this._path = path;
  }

  get path(): string {
    return this._path;
  }

  rename(newPath: string): void {
    this._path = newPath;
    this.pathChanged.emit(newPath);
  }
}

export class FakeCellList {
  readonly changed = new CountingSignal<unknown, { type: string }>();
}

export class FakeNotebookContent {
  widgets: FakeCell[] = [];
  readonly model = { cells: new FakeCellList() };
  activeCell: FakeCell | null = null;
}

export class FakePanel {
  readonly content = new FakeNotebookContent();
  readonly sessionContext: FakeSessionContext;
  readonly context: FakeContext;
  readonly disposed = new CountingSignal<unknown, void>();
  isDisposed = false;

  constructor(path = 'nb.ipynb', kernel: FakeKernel | null = new FakeKernel()) {
    this.sessionContext = new FakeSessionContext(kernel);
    this.context = new FakeContext(path);
  }

  addCell(id: string, type = 'code', source = ''): FakeCell {
    const cell = new FakeCell(id, type, source);
    this.content.widgets.push(cell);
    return cell;
  }

  disposeNow(): void {
    this.isDisposed = true;
    this.disposed.emit(undefined);
  }

  get kernel(): FakeKernel | null {
    return this.sessionContext.session?.kernel ?? null;
  }
}

export class FakeTracker {
  readonly currentChanged = new CountingSignal<unknown, unknown>();
  readonly activeCellChanged = new CountingSignal<unknown, unknown>();
  readonly widgetAdded = new CountingSignal<unknown, FakePanel>();
  readonly widgets: FakePanel[] = [];
  currentWidget: FakePanel | null = null;
  activeCell: FakeCell | null = null;

  setCurrent(panel: FakePanel | null): void {
    this.currentWidget = panel;
    this.currentChanged.emit(panel);
  }

  /** Register a panel with the tracker and announce it via widgetAdded. */
  addWidget(panel: FakePanel): void {
    this.widgets.push(panel);
    this.widgetAdded.emit(panel);
  }

  /** Iterate tracked panels (KernelDetector scans these at construction). */
  forEach = (fn: (panel: FakePanel) => void): void => {
    this.widgets.forEach(p => fn(p));
  };
}

// ---------------------------------------------------------------------------
// Execution signals (the production seam — see cellhighlighter.ts)
// ---------------------------------------------------------------------------

export interface IFakeExecutedArgs {
  notebook: unknown;
  cell: FakeCell;
  success?: boolean;
}

export class FakeExecutionSignals {
  readonly executed = new CountingSignal<unknown, IFakeExecutedArgs>();
  readonly executionScheduled = new CountingSignal<
    unknown,
    { notebook: unknown; cell: FakeCell }
  >();
}

// ---------------------------------------------------------------------------
// Highlighter fake for execution-hook specs (only what the hook touches)
// ---------------------------------------------------------------------------

/** Staleness-manager stub used by FakeHighlighter. */
export class FakeHookStalenessManager {
  staleCells = new Set<string>();
  reasons = new Map<string, unknown>();

  setReason(cellId: string, reason: unknown): void {
    this.reasons.set(cellId, reason);
  }

  updateFromMetadata(meta: { stale_cells: string[] }): void {
    this.staleCells = new Set(meta.stale_cells);
  }
}

export class FakeHighlighter {
  managers = new Map<unknown, FakeHookStalenessManager>();
  cellUpdates: Array<{ cellId: string; path: string }> = [];
  statusUpdates: Array<{ icon: string; text: string }> = [];

  getStalenessManager(panel: unknown): FakeHookStalenessManager {
    let m = this.managers.get(panel);
    if (!m) {
      m = new FakeHookStalenessManager();
      this.managers.set(panel, m);
    }
    return m;
  }

  updateCell(
    cell: { model: { id: string } },
    _mgr: unknown,
    _order: string[],
    path: string
  ): void {
    this.cellUpdates.push({ cellId: cell.model.id, path });
  }

  refreshDependencies(): void {
    // no-op
  }

  updateStatus(icon: string, text: string): void {
    this.statusUpdates.push({ icon, text });
  }
}

/** Build a minimal kernel "metadata" comm payload for a cell. */
export function metadataFor(
  cellId: string,
  staleCells: string[] = []
): unknown {
  return {
    type: 'metadata',
    cell_id: cellId,
    execution_seq: 1,
    read_locs: [],
    write_locs: [],
    changed_locs: [],
    stale_cells: staleCells,
    cell_order: [cellId]
  };
}

// ---------------------------------------------------------------------------
// Casting helpers — one place for the unavoidable structural casts
// ---------------------------------------------------------------------------

/* eslint-disable @typescript-eslint/no-explicit-any */
export const asTracker = (t: FakeTracker): any => t as any;
export const asPanel = (p: FakePanel): any => p as any;
export const asSignals = (s: FakeExecutionSignals): any => s as any;
export const asCell = (c: FakeCell): any => c as any;
export const asHighlighter = (h: FakeHighlighter): any => h as any;
/* eslint-enable @typescript-eslint/no-explicit-any */
