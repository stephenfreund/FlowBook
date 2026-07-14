/**
 * Tests for the notice managers (stalenessnotice.ts, violationnotice.ts):
 * notice insertion at output index 0, HTML escaping of user-controlled
 * location names, dedup (no fromJSON rewrite when the text is unchanged),
 * removal, and violation-suppresses-staleness precedence.
 * See FRONTEND_TESTING.md §8, phase 2.
 */

import { StalenessNoticeManager } from '../stalenessnotice';
import { ViolationNoticeManager } from '../violationnotice';
import { StalenessManager } from '../stalenessmanager';
import { IFrontendStalenessReason, IReproducibilityError } from '../types';
import { FakeCell, FakePanel, asCell, asPanel } from './testutils';

const XSS = '<img src=x onerror=alert(1)>';
const XSS_ESCAPED = '&lt;img src=x onerror=alert(1)&gt;';
const ORDER = ['c0', 'c1'];
const STREAM = { output_type: 'stream', name: 'stdout', text: 'hi' };

function setup(): {
  panel: FakePanel;
  manager: StalenessManager;
  cell: FakeCell;
  notice: StalenessNoticeManager;
} {
  const panel = new FakePanel('nb.ipynb');
  const manager = new StalenessManager(asPanel(panel));
  const cell = panel.addCell('c1', 'code', 'x = 1');
  const notice = new StalenessNoticeManager();
  return { panel, manager, cell, notice };
}

function setReason(
  manager: StalenessManager,
  cell: FakeCell,
  variables: string[]
): void {
  const reason: IFrontendStalenessReason = {
    type: 'variable_modified',
    causing_cell: 'c0',
    variables,
    message: 'placeholder'
  };
  manager.setReason(cell.model.id, reason);
}

function violation(
  loc: string,
  errorType = 'write_before_read',
  causer?: string
): IReproducibilityError {
  return {
    error_type: errorType,
    cell_id: 'c1',
    locations: [loc],
    message: 'violation',
    causer_cell: causer
  };
}

function stalenessNotices(cell: FakeCell): unknown[] {
  return cell.model.outputs.list.filter(
    o =>
      (o.metadata as Record<string, unknown> | undefined)
        ?.flowbook_staleness_notice === true
  );
}

describe('staleness notice', () => {
  it('inserts an escaped notice at output index 0, preserving other outputs', () => {
    const { manager, cell, notice } = setup();
    cell.model.outputs.fromJSON([STREAM]);
    setReason(manager, cell, [XSS]);

    notice.updateStalenessNotice(asCell(cell), true, manager, ORDER);

    const outputs = cell.model.outputs.list;
    expect(outputs).toHaveLength(2);
    expect(outputs[0].metadata).toEqual({ flowbook_staleness_notice: true });
    const html = (outputs[0].data as Record<string, string>)['text/html'];
    expect(html).toContain(XSS_ESCAPED);
    expect(html).not.toContain(XSS);
    expect(outputs[1]).toEqual(STREAM);
  });

  it('does not rewrite outputs when the notice is already up to date', () => {
    const { manager, cell, notice } = setup();
    setReason(manager, cell, ['x']);
    const fromJSON = jest.spyOn(cell.model.outputs, 'fromJSON');

    notice.updateStalenessNotice(asCell(cell), true, manager, ORDER);
    notice.updateStalenessNotice(asCell(cell), true, manager, ORDER);

    expect(fromJSON).toHaveBeenCalledTimes(1);
  });

  it('removes the notice when no longer stale, preserving other outputs', () => {
    const { manager, cell, notice } = setup();
    cell.model.outputs.fromJSON([STREAM]);
    setReason(manager, cell, ['x']);
    notice.updateStalenessNotice(asCell(cell), true, manager, ORDER);
    expect(stalenessNotices(cell)).toHaveLength(1);

    notice.updateStalenessNotice(asCell(cell), false, manager, ORDER);

    expect(stalenessNotices(cell)).toHaveLength(0);
    expect(cell.model.outputs.list).toEqual([STREAM]);
  });

  it('is suppressed when the cell has violations', () => {
    const { manager, cell, notice } = setup();
    setReason(manager, cell, ['x']);
    notice.updateStalenessNotice(asCell(cell), true, manager, ORDER);
    expect(stalenessNotices(cell)).toHaveLength(1);

    cell.model.setMetadata('flowbook', { errors: [violation('x')] });
    notice.updateStalenessNotice(asCell(cell), true, manager, ORDER);

    // The violation is more specific — the stale notice is removed, not
    // re-added, even though the cell IS stale.
    expect(stalenessNotices(cell)).toHaveLength(0);
  });
});

describe('violation notice', () => {
  it('renders violations with escaped locations at output index 0', () => {
    const { cell } = setup();
    const violations = new ViolationNoticeManager();
    cell.model.outputs.fromJSON([STREAM]);
    cell.model.setMetadata('flowbook', { errors: [violation(XSS)] });

    const hasViolations = violations.updateViolationNotice(asCell(cell), ORDER);

    expect(hasViolations).toBe(true);
    const outputs = cell.model.outputs.list;
    expect(outputs).toHaveLength(2);
    expect(
      (outputs[0].metadata as Record<string, unknown>).flowbook_violation_notice
    ).toBe(true);
    const html = (outputs[0].data as Record<string, string>)['text/html'];
    expect(html).toContain(XSS_ESCAPED);
    expect(html).not.toContain(XSS);
    expect(outputs[1]).toEqual(STREAM);
  });

  it('resolves causer cells to @A references', () => {
    const { cell } = setup();
    const violations = new ViolationNoticeManager();
    cell.model.setMetadata('flowbook', {
      errors: [violation('x', 'no_write_after_read', 'c0')]
    });

    violations.updateViolationNotice(asCell(cell), ORDER);

    const html = (cell.model.outputs.list[0].data as Record<string, string>)[
      'text/html'
    ];
    expect(html).toContain('Writes <code>x</code> read by @A above');
  });

  it('clears the notice when errors disappear', () => {
    const { cell } = setup();
    const violations = new ViolationNoticeManager();
    cell.model.outputs.fromJSON([STREAM]);
    cell.model.setMetadata('flowbook', { errors: [violation('x')] });
    expect(violations.updateViolationNotice(asCell(cell), ORDER)).toBe(true);

    cell.model.setMetadata('flowbook', { errors: [] });

    expect(violations.updateViolationNotice(asCell(cell), ORDER)).toBe(false);
    expect(cell.model.outputs.list).toEqual([STREAM]);
  });
});
