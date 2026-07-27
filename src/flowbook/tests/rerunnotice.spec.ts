/**
 * Tests for the rerun warning notice manager (rerunnotice.ts): the
 * orange notice the run-until-clean loop places under a cell whose
 * rerun backward-marked an earlier cell. Covers insertion at output
 * index 0, HTML escaping, dedup (no fromJSON rewrite when the text is
 * unchanged), removal, clearAll at sweep start, and non-code cells.
 */

import { RerunWarningNoticeManager } from '../rerunnotice';
import { FakeCell, FakePanel, asCell, asPanel } from './testutils';

const XSS = '<img src=x onerror=alert(1)>';
const XSS_ESCAPED = '&lt;img src=x onerror=alert(1)&gt;';
const STREAM = { output_type: 'stream', name: 'stdout', text: 'hi' };

function warningNotices(cell: FakeCell): unknown[] {
  return cell.model.outputs.list.filter(
    o =>
      (o.metadata as Record<string, unknown> | undefined)
        ?.flowbook_rerun_warning_notice === true
  );
}

function setup(): {
  panel: FakePanel;
  cell: FakeCell;
  notice: RerunWarningNoticeManager;
} {
  const panel = new FakePanel('nb.ipynb');
  const cell = panel.addCell('c1', 'code', 'x = 1');
  return { panel, cell, notice: new RerunWarningNoticeManager() };
}

describe('rerun warning notice', () => {
  it('inserts an escaped notice at output index 0, preserving other outputs', () => {
    const { cell, notice } = setup();
    cell.model.outputs.fromJSON([STREAM]);

    notice.showWarning(asCell(cell), `dropped the write of ${XSS}`);

    const outputs = cell.model.outputs.list;
    expect(outputs).toHaveLength(2);
    expect(outputs[0].metadata).toEqual({
      flowbook_rerun_warning_notice: true
    });
    const html = (outputs[0].data as Record<string, string>)['text/html'];
    expect(html).toContain('flowbook-rerun-warning-notice');
    expect(html).toContain(XSS_ESCAPED);
    expect(html).not.toContain(XSS);
    expect(outputs[1]).toEqual(STREAM);
  });

  it('renders backtick-wrapped names as code', () => {
    const { cell, notice } = setup();

    notice.showWarning(asCell(cell), 'this run wrote `a` but not `b`');

    const html = (cell.model.outputs.list[0].data as Record<string, string>)[
      'text/html'
    ];
    expect(html).toContain('<code>a</code>');
    expect(html).toContain('<code>b</code>');
  });

  it('does not rewrite outputs when the notice is already up to date', () => {
    const { cell, notice } = setup();
    const fromJSON = jest.spyOn(cell.model.outputs, 'fromJSON');

    notice.showWarning(asCell(cell), 'run 2 of at most 10');
    notice.showWarning(asCell(cell), 'run 2 of at most 10');

    expect(fromJSON).toHaveBeenCalledTimes(1);
  });

  it('replaces the notice when the message changes (run counter)', () => {
    const { cell, notice } = setup();

    notice.showWarning(asCell(cell), 'run 2 of at most 10');
    notice.showWarning(asCell(cell), 'run 3 of at most 10');

    expect(warningNotices(cell)).toHaveLength(1);
    const html = (cell.model.outputs.list[0].data as Record<string, string>)[
      'text/html'
    ];
    expect(html).toContain('run 3 of at most 10');
  });

  it('clearWarning removes the notice, preserving other outputs', () => {
    const { cell, notice } = setup();
    cell.model.outputs.fromJSON([STREAM]);
    notice.showWarning(asCell(cell), 'msg');
    expect(warningNotices(cell)).toHaveLength(1);

    notice.clearWarning(asCell(cell));

    expect(warningNotices(cell)).toHaveLength(0);
    expect(cell.model.outputs.list).toEqual([STREAM]);
  });

  it('clearWarning without a notice does not rewrite outputs', () => {
    const { cell, notice } = setup();
    cell.model.outputs.fromJSON([STREAM]);
    const fromJSON = jest.spyOn(cell.model.outputs, 'fromJSON');

    notice.clearWarning(asCell(cell));

    expect(fromJSON).not.toHaveBeenCalled();
  });

  it('clearAll removes notices from every cell (sweep start)', () => {
    const { panel, cell, notice } = setup();
    const other = panel.addCell('c2', 'code', 'y = 2');
    notice.showWarning(asCell(cell), 'msg');
    notice.showWarning(asCell(other), 'msg');

    notice.clearAll(asPanel(panel));

    expect(warningNotices(cell)).toHaveLength(0);
    expect(warningNotices(other)).toHaveLength(0);
  });

  it('ignores non-code cells', () => {
    const { panel, notice } = setup();
    const md = panel.addCell('m1', 'markdown', '# title');

    notice.showWarning(asCell(md), 'msg');

    expect(md.model.outputs.list).toHaveLength(0);
  });
});
