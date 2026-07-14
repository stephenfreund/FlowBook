/**
 * Unit tests for the canonical backend-reason formatting table.
 *
 * The reason vocabulary comes from the shared parity fixture
 * (tests/fixtures/reason_types.json); the Python side asserts its
 * ReasonType enum is a subset of the same list, so a new reason type
 * added on either side fails a test until both are updated.
 */

import {
  backendReasonToFrontend,
  formatBackendReasonMessage
} from '../reasonformat';
import { BackendReasonType, IBackendStalenessReason } from '../types';

import reasonTypes from './fixtures/reason_types.json';

const ALL_TYPES = (reasonTypes as { backend_reason_types: string[] })
  .backend_reason_types as BackendReasonType[];

// Order: [a, stale, b] — causer 'a' is above 'stale', 'b' is below.
const ORDER = ['cell-a', 'cell-stale', 'cell-b'];

function reason(
  type: BackendReasonType,
  loc?: string,
  cellId?: string
): IBackendStalenessReason {
  return {
    type,
    ...(loc ? { loc } : {}),
    ...(cellId ? { cell_id: cellId } : {})
  };
}

describe('reason vocabulary coverage', () => {
  const GENERIC_DEFAULT = 'Dependencies changed by';

  it.each(ALL_TYPES.map(t => [t] as [BackendReasonType]))(
    '%s formats to a specific (non-default) message',
    type => {
      const r = backendReasonToFrontend(
        reason(type, 'x', 'cell-a'),
        ORDER,
        'cell-stale'
      );
      expect(r.message.length).toBeGreaterThan(0);
      expect(r.message.startsWith(GENERIC_DEFAULT)).toBe(false);
      expect(r.message).not.toBe('Cell is stale');
    }
  );

  it('both views agree on the message', () => {
    for (const type of ALL_TYPES) {
      const input = reason(type, 'x', 'cell-a');
      expect(formatBackendReasonMessage(input, ORDER, 'cell-stale')).toBe(
        backendReasonToFrontend(input, ORDER, 'cell-stale').message
      );
    }
  });
});

describe('direction and references', () => {
  it('says above for a causer earlier in the order', () => {
    const r = backendReasonToFrontend(
      reason('forward_stale', 'x', 'cell-a'),
      ORDER,
      'cell-stale'
    );
    expect(r.message).toBe('`x` modified by @A above');
    expect(r.causing_cell).toBe('cell-a');
    expect(r.variables).toEqual(['x']);
  });

  it('says below for a causer later in the order', () => {
    const r = backendReasonToFrontend(
      reason('no_read_before_write', 'x', 'cell-b'),
      ORDER,
      'cell-stale'
    );
    expect(r.message).toBe('Reads `x` written by @C below');
  });

  it('falls back to "a deleted cell" when the causer is gone', () => {
    const r = backendReasonToFrontend(
      reason('forward_stale', 'x', 'gone'),
      ORDER,
      'cell-stale'
    );
    expect(r.message).toBe('`x` modified by a deleted cell');
  });

  it('omits direction when the current cell is unknown', () => {
    const msg = backendReasonToFrontend(
      reason('forward_stale', 'x', 'cell-a'),
      ORDER
    ).message;
    expect(msg).toBe('`x` modified by @A');
  });
});

describe('specific wordings', () => {
  const cases: Array<[BackendReasonType, string]> = [
    ['never_executed', 'Cell has never been executed'],
    ['code_changed', 'Source code was edited'],
    ['order_changed', 'Cell order changed'],
    ['write_overlap', '`x` also written by @A above'],
    ['backward_stale', 'Write conflict on `x` with @A above'],
    ['no_write_after_read', 'Writes `x` already read by @A above'],
    ['no_read_and_write', 'Reads and writes `x`'],
    ['write_before_read', '`x` is not defined by any cell above'],
    [
      'unrecoverable_mutation',
      '`x` was modified in place, which violates rerun consistency'
    ]
  ];

  it.each(cases)('%s', (type, expected) => {
    expect(
      formatBackendReasonMessage(
        reason(type, 'x', 'cell-a'),
        ORDER,
        'cell-stale'
      )
    ).toBe(expected);
  });
});
