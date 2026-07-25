/**
 * Tests for waitForFlowbookMetadata (types.ts) — the post-run metadata
 * race guard: resolves when a new execution's metadata lands (higher
 * execution_seq or first appearance), resolves (never rejects) at the
 * ~2 s cap, and polls on a 50 ms cadence. All timing via jest fake timers.
 * See FRONTEND_TESTING.md §8, phase 2.
 */

import { IReproducibilityMetadata, waitForFlowbookMetadata } from '../types';

function meta(seq: number): IReproducibilityMetadata {
  return {
    cell_id: 'c1',
    execution_seq: seq,
    read_locs: [],
    write_locs: [],
    changed_locs: [],
    stale_cells: [],
    cell_order: ['c1']
  };
}

type IResult = IReproducibilityMetadata | undefined | 'pending';

function makeModel(initial?: IReproducibilityMetadata): {
  state: { meta: IReproducibilityMetadata | undefined };
  model: { getMetadata: (key: string) => unknown };
} {
  const state = { meta: initial };
  const model = {
    getMetadata: (key: string): unknown =>
      key === 'flowbook' ? state.meta : undefined
  };
  return { state, model };
}

describe('waitForFlowbookMetadata', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('resolves once execution_seq increases', async () => {
    const before = meta(1);
    const { state, model } = makeModel(before);
    let result: IResult = 'pending';
    waitForFlowbookMetadata(model, before).then(r => {
      result = r;
    });

    // Metadata unchanged — keeps polling.
    await jest.advanceTimersByTimeAsync(200);
    expect(result).toBe('pending');

    state.meta = meta(2);
    await jest.advanceTimersByTimeAsync(50);
    expect(result).toEqual(meta(2));
  });

  it('resolves on first appearance when there was no prior metadata', async () => {
    const { state, model } = makeModel(undefined);
    let result: IResult = 'pending';
    waitForFlowbookMetadata(model, undefined).then(r => {
      result = r;
    });

    await jest.advanceTimersByTimeAsync(150);
    expect(result).toBe('pending');

    state.meta = meta(1);
    await jest.advanceTimersByTimeAsync(50);
    expect(result).toEqual(meta(1));
  });

  it('resolves (not rejects) at the timeout cap when nothing changes', async () => {
    const before = meta(1);
    const { model } = makeModel(before);
    let result: IResult = 'pending';
    let rejected = false;
    waitForFlowbookMetadata(model, before).then(
      r => {
        result = r;
      },
      () => {
        rejected = true;
      }
    );

    await jest.advanceTimersByTimeAsync(2100);

    expect(rejected).toBe(false);
    // Returns the freshest (still stale) metadata it observed.
    expect(result).toBe(before);
  });

  it('polls on a 50 ms cadence', async () => {
    const before = meta(1);
    const { state, model } = makeModel(before);
    let result: IResult = 'pending';
    waitForFlowbookMetadata(model, before).then(r => {
      result = r;
    });

    // Fresh metadata lands right after the call — but the poller only
    // re-reads on its 50 ms tick.
    state.meta = meta(2);
    await jest.advanceTimersByTimeAsync(49);
    expect(result).toBe('pending');

    await jest.advanceTimersByTimeAsync(1);
    expect(result).toEqual(meta(2));
  });
});
