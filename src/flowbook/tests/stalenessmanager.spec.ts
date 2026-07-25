/**
 * Tests for StalenessManager: updateFromMetadata set/diff semantics,
 * kernel-restart clearing, and dispose hygiene (audit item 3).
 * See FRONTEND_TESTING.md §8, phase 2.
 */

import { StalenessManager, IStalenessChange } from '../stalenessmanager';
import { IReproducibilityMetadata } from '../types';
import { FakePanel, asPanel } from './testutils';

function meta(
  staleCells: string[],
  reasons?: IReproducibilityMetadata['staleness_reasons']
): IReproducibilityMetadata {
  return {
    cell_id: 'c0',
    execution_seq: 1,
    read_locs: [],
    write_locs: [],
    changed_locs: [],
    stale_cells: staleCells,
    cell_order: staleCells,
    staleness_reasons: reasons
  };
}

function makeManager(): {
  panel: FakePanel;
  manager: StalenessManager;
  events: IStalenessChange[];
} {
  const panel = new FakePanel('a.ipynb');
  const manager = new StalenessManager(asPanel(panel));
  const events: IStalenessChange[] = [];
  manager.stalenessChanged.connect((_, change) => {
    events.push(change);
  });
  return { panel, manager, events };
}

describe('updateFromMetadata', () => {
  it('sets and clears stale cells, emitting added/removed diffs', () => {
    const { manager, events } = makeManager();

    manager.updateFromMetadata(meta(['a', 'b']));
    expect(manager.isCellStale('a')).toBe(true);
    expect(manager.isCellStale('b')).toBe(true);
    expect(events).toEqual([
      { added: ['a', 'b'], removed: [], current: ['a', 'b'] }
    ]);

    manager.updateFromMetadata(meta(['b']));
    expect(manager.isCellStale('a')).toBe(false);
    expect(manager.isCellStale('b')).toBe(true);
    expect(events[1]).toEqual({ added: [], removed: ['a'], current: ['b'] });

    // Identical update — no emission.
    manager.updateFromMetadata(meta(['b']));
    expect(events).toHaveLength(2);
    manager.dispose();
  });

  it('stores reasons from metadata and drops them when cells freshen', () => {
    const { manager } = makeManager();

    manager.updateFromMetadata(
      meta(['a'], { a: [{ type: 'forward_stale', loc: 'x' }] })
    );
    expect(manager.getReason('a')).toEqual({ type: 'forward_stale', loc: 'x' });

    manager.updateFromMetadata(meta([]));
    expect(manager.getReason('a')).toBeUndefined();
    manager.dispose();
  });
});

describe('kernel restart', () => {
  it('clears staleness on restarting and autorestarting statuses', () => {
    const { panel, manager, events } = makeManager();
    manager.updateFromMetadata(meta(['a']));

    panel.sessionContext.setStatus('restarting');
    expect(manager.staleCells.size).toBe(0);
    expect(events[1]).toEqual({ added: [], removed: ['a'], current: [] });

    manager.updateFromMetadata(meta(['b']));
    panel.sessionContext.setStatus('autorestarting');
    expect(manager.staleCells.size).toBe(0);
    manager.dispose();
  });
});

describe('dispose', () => {
  it('disconnects statusChanged and further emissions are no-ops', () => {
    const { panel, manager, events } = makeManager();
    expect(panel.sessionContext.statusChanged.live).toBe(1);
    manager.updateFromMetadata(meta(['a']));
    const eventsBefore = events.length;

    manager.dispose();

    expect(panel.sessionContext.statusChanged.live).toBe(0);
    expect(manager.staleCells.size).toBe(0);

    // Disconnected — no clear-triggered signal, no throw.
    panel.sessionContext.setStatus('restarting');
    expect(events).toHaveLength(eventsBefore);
  });
});
