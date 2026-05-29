/**
 * AI-attribution helpers.
 *
 * FlowBook marks the notebook activity it performs on behalf of an LLM (the
 * in-product fix applier, the NBI bridge handlers, the MCP server) so that an
 * *optional* observer — currently LogBook — can attribute it to the AI rather
 * than the user. This is deliberately a general, one-way mechanism: FlowBook
 * imports nothing from any observer and depends on no observer being present.
 *
 * Two decoupled contracts, both no-ops when unobserved:
 *
 *  1. Edits → `aiTransact` tags the underlying Yjs `Doc` transaction with
 *     `FLOWBOOK_TX_ORIGIN`. An observer that watches Yjs transaction origins
 *     (LogBook does) attributes the resulting edits to the AI. Tagging is done
 *     on the raw Yjs doc (the stable `transact(fn, origin)` API) rather than
 *     the JupyterLab shared-model wrapper, whose `transact` signature varies by
 *     version; nested mutations (e.g. `setSource`) inherit the origin.
 *
 *  2. Executions / out-of-process activity → `emitAiActivity` dispatches a DOM
 *     CustomEvent (`AI_ACTIVITY_EVENT`). Cell executions are not shared-model
 *     edits, so the Yjs-origin mechanism can't cover them; the event lets an
 *     observer attribute an AI-driven run (e.g. the MCP server running a cell
 *     on the shared kernel). With no listener the dispatch is a harmless no-op.
 */

/**
 * Yjs transaction origin tag for FlowBook's AI-driven edits. An observer
 * recognizes this string to attribute the edit to the AI.
 */
export const FLOWBOOK_TX_ORIGIN = 'flowbook';

/** Minimal structural type for "something exposing a Yjs Doc". */
interface IHasYDoc {
  ydoc?: { transact?: (f: () => void, origin?: unknown) => void };
}

/**
 * Run `fn` (which mutates `sharedModel`) inside a Yjs transaction tagged as
 * AI-originated. Falls back to calling `fn` directly when no Yjs doc is
 * reachable (e.g. a non-collaborative model in tests), so callers can wrap
 * unconditionally.
 */
export function aiTransact(sharedModel: unknown, fn: () => void): void {
  const ydoc = (sharedModel as IHasYDoc | null | undefined)?.ydoc;
  if (ydoc && typeof ydoc.transact === 'function') {
    ydoc.transact(fn, FLOWBOOK_TX_ORIGIN);
  } else {
    fn();
  }
}

/**
 * Name of the general DOM event announcing AI-driven notebook activity that is
 * NOT a shared-model edit — primarily a cell *execution* driven by an
 * out-of-process agent (the MCP server running a cell on the shared kernel).
 *
 * A one-way, dependency-free contract: FlowBook dispatches; any observer (e.g.
 * LogBook) may `addEventListener` for it. With no listener the dispatch is a
 * harmless no-op, so FlowBook never depends on an observer being present.
 */
export const AI_ACTIVITY_EVENT = 'ai-notebook-activity';

/** Payload of an {@link AI_ACTIVITY_EVENT}. */
export interface IAiActivityDetail {
  /** Announcing extension; lets observers tell producers apart. */
  source: 'flowbook';
  /** Notebook path the activity occurred in. */
  path: string;
  /** Cell the activity concerned, if applicable. */
  cellId?: string;
  /** What happened — currently 'execute'. */
  kind: 'execute';
  /**
   * Execution outcome. Observers need this because frontend execution signals
   * don't fire for out-of-process runs, so this event is the only record.
   */
  status?: 'ok' | 'error';
  /** The cell's post-run execution count, if known. */
  executionCount?: number | null;
  /** Number of outputs produced, if known. */
  outputCount?: number;
}

/**
 * Announce AI-driven activity via {@link AI_ACTIVITY_EVENT}. Safe to call when
 * no DOM / no listener is present (e.g. headless tests) — it never throws.
 */
export function emitAiActivity(detail: Omit<IAiActivityDetail, 'source'>): void {
  try {
    if (typeof document !== 'undefined' && typeof CustomEvent !== 'undefined') {
      const full: IAiActivityDetail = { source: 'flowbook', ...detail };
      document.dispatchEvent(new CustomEvent(AI_ACTIVITY_EVENT, { detail: full }));
    }
  } catch {
    // Best-effort: attribution must never break notebook execution.
  }
}
