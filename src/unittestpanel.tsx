import { Widget } from '@lumino/widgets';
import { Cell } from '@jupyterlab/cells';
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { IUnitTest, IUnitTests } from './types';

interface IUnitTestPanelProps {
  cell: Cell | null;
}

const UnitTestEditor: React.FC<IUnitTestPanelProps> = ({ cell }) => {
  const [tests, setTests] = React.useState<IUnitTest[]>([]);
  const [cellId, setCellId] = React.useState<string | null>(null);

  // Load tests from cell metadata when cell changes
  React.useEffect(() => {
    if (!cell) {
      setTests([]);
      setCellId(null);
      return;
    }

    // Get ferret metadata using the proper API
    const ferretMetadata = cell.model.getMetadata('ferret') as any;
    const unitTests = ferretMetadata?.unit_tests as IUnitTests | undefined;

    console.log('[UnitTestPanel] Loading tests from cell', cell.model.id, unitTests);

    // Update both tests and cellId together
    if (unitTests && unitTests.tests) {
      setTests([...unitTests.tests]);
    } else {
      setTests([]);
    }
    setCellId(cell.model.id);
  }, [cell]);

  // Save tests to cell metadata whenever tests change
  React.useEffect(() => {
    // Don't save if no cell or if cell changed (cellId mismatch)
    if (!cell || !cellId || cell.model.id !== cellId) {
      return;
    }

    // Retrieve existing metadata for ferret, or create an empty object
    const ferretMetadata = cell.model.getMetadata('ferret') as any || {};

    // Update the unit_tests property
    ferretMetadata.unit_tests = {
      tests: tests.map(t => ({
        title: t.title,
        description: t.description,
        setup_code: t.setup_code,
        assertion_code: t.assertion_code
      }))
    };

    // Set the updated metadata back
    cell.model.setMetadata('ferret', ferretMetadata);

    console.log('[UnitTestPanel] Saved tests to cell', cell.model.id, 'tests:', JSON.stringify(tests));

    // Verify it was written
    setTimeout(() => {
      const checkFerret = cell.model.getMetadata('ferret') as any;
      console.log('[UnitTestPanel] Verify: ferret.unit_tests =', checkFerret?.unit_tests);
    }, 100);
  }, [tests, cell, cellId]);

  const addTest = () => {
    const newTest: IUnitTest = {
      title: 'New Test',
      description: '',
      setup_code: '',
      assertion_code: 'assert True  # Add your assertions here'
    };
    setTests([...tests, newTest]);
  };

  const deleteTest = (index: number) => {
    const updated = tests.filter((_, i) => i !== index);
    setTests(updated);
  };

  const updateTest = (index: number, field: keyof IUnitTest, value: string) => {
    const updated = [...tests];
    updated[index] = { ...updated[index], [field]: value };
    setTests(updated);
  };

  if (!cell) {
    return (
      <div className="ferret-unittest-empty">
        <p>No cell selected.</p>
        <p>Select a code cell to view or edit its unit tests.</p>
      </div>
    );
  }

  return (
    <div className="ferret-unittest-content">
      <div className="ferret-unittest-header">
        <div className="ferret-unittest-cell-id">Cell: {cell.model.id.substring(0, 8)}...</div>
      </div>

      <div className="ferret-unittest-actions">
        <button onClick={addTest} className="ferret-unittest-btn-add">
          + Add Test
        </button>
        <button className="ferret-unittest-btn-generate" disabled>
          ⚡ Auto-generate (Coming Soon)
        </button>
      </div>

      <div className="ferret-unittest-list">
        {tests.length === 0 ? (
          <div className="ferret-unittest-empty-list">
            No tests defined. Click &quot;+ Add Test&quot; to create one.
          </div>
        ) : (
          tests.map((test, index) => (
            <div key={index} className="ferret-unittest-item">
              <div className="ferret-unittest-item-header">
                <input
                  type="text"
                  className="ferret-unittest-title-input"
                  value={test.title}
                  onChange={(e) => updateTest(index, 'title', e.target.value)}
                  placeholder="Test title"
                />
                <button
                  onClick={() => deleteTest(index)}
                  className="ferret-unittest-btn-delete"
                >
                  🗑️
                </button>
              </div>

              <div className="ferret-unittest-field">
                <label>Description:</label>
                <textarea
                  className="ferret-unittest-textarea"
                  value={test.description}
                  onChange={(e) => updateTest(index, 'description', e.target.value)}
                  placeholder="What does this test verify?"
                  rows={2}
                />
              </div>

              <div className="ferret-unittest-field">
                <label>Setup Code:</label>
                <textarea
                  className="ferret-unittest-code-editor"
                  value={test.setup_code}
                  onChange={(e) => updateTest(index, 'setup_code', e.target.value)}
                  placeholder="# Set up test globals here"
                  rows={4}
                  spellCheck={false}
                />
              </div>

              <div className="ferret-unittest-field">
                <label>Assertion Code:</label>
                <textarea
                  className="ferret-unittest-code-editor"
                  value={test.assertion_code}
                  onChange={(e) => updateTest(index, 'assertion_code', e.target.value)}
                  placeholder="# Add assertions here"
                  rows={4}
                  spellCheck={false}
                />
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export class UnitTestPanel extends Widget {
  private _cell: Cell | null = null;

  constructor() {
    super();
    this.id = 'ferret-unittest-panel';
    this.addClass('ferret-unittest-panel');
    this.title.label = 'Unit Tests';
    this.title.closable = true;
    this.title.caption = 'Edit unit tests for the selected cell';

    this.render();
  }

  private render(): void {
    ReactDOM.render(<UnitTestEditor cell={this._cell} />, this.node);
  }

  public updateCell(cell: Cell | null): void {
    this._cell = cell;
    this.render();
  }

  public clear(): void {
    this._cell = null;
    this.render();
  }

  dispose(): void {
    if (this.isDisposed) {
      return;
    }
    ReactDOM.unmountComponentAtNode(this.node);
    super.dispose();
  }
}
