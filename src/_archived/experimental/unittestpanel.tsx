import { Widget } from '@lumino/widgets';
import { Cell } from '@jupyterlab/cells';
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { IUnitTest, IUnitTests } from './types';
import { JupyterFrontEnd } from '@jupyterlab/application';
import { INotebookTracker } from '@jupyterlab/notebook';

interface IUnitTestPanelProps {
  cell: Cell | null;
  app: JupyterFrontEnd | null;
  tracker: INotebookTracker | null;
}

const UnitTestEditor: React.FC<IUnitTestPanelProps> = ({
  cell,
  app,
  tracker
}) => {
  const [tests, setTests] = React.useState<IUnitTest[]>([]);
  const [cellId, setCellId] = React.useState<string | null>(null);
  const [isGenerating, setIsGenerating] = React.useState<boolean>(false);

  // Load tests from cell metadata when cell changes
  React.useEffect(() => {
    if (!cell) {
      setTests([]);
      setCellId(null);
      return;
    }

    // Get flowbook metadata using the proper API
    const flowbookMetadata = cell.model.getMetadata('flowbook') as any;
    const unitTests = flowbookMetadata?.unit_tests as IUnitTests | undefined;

    console.log(
      '[UnitTestPanel] Loading tests from cell',
      cell.model.id,
      unitTests
    );

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

    // Retrieve existing metadata for flowbook, or create an empty object
    const flowbookMetadata = (cell.model.getMetadata('flowbook') as any) || {};

    // Update the unit_tests property
    flowbookMetadata.unit_tests = {
      tests: tests.map(t => ({
        title: t.title,
        description: t.description,
        setup_code: t.setup_code,
        assertion_code: t.assertion_code
      }))
    };

    // Set the updated metadata back
    cell.model.setMetadata('flowbook', flowbookMetadata);

    console.log(
      '[UnitTestPanel] Saved tests to cell',
      cell.model.id,
      'tests:',
      JSON.stringify(tests)
    );

    // Verify it was written
    setTimeout(() => {
      const checkFlowbook = cell.model.getMetadata('flowbook') as any;
      console.log(
        '[UnitTestPanel] Verify: flowbook.unit_tests =',
        checkFlowbook?.unit_tests
      );
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

  const handleGenerateTests = async () => {
    if (!cell || !app || !tracker) {
      console.warn(
        '[UnitTestPanel] Cannot generate tests: missing cell, app, or tracker'
      );
      return;
    }

    setIsGenerating(true);

    try {
      // Execute the generate_tests command with the current cell ID
      await app.commands.execute('flowbook:generate_tests', {
        cellId: cell.model.id
      });

      // After command completes, reload tests from cell metadata
      // The command will have updated the metadata, so we need to refresh
      setTimeout(() => {
        const flowbookMetadata = cell.model.getMetadata('flowbook') as any;
        const unitTests = flowbookMetadata?.unit_tests as
          | IUnitTests
          | undefined;

        if (unitTests && unitTests.tests) {
          setTests([...unitTests.tests]);
          console.log(
            '[UnitTestPanel] Reloaded tests after generation:',
            unitTests.tests.length
          );
        }
        setIsGenerating(false);
      }, 500); // Small delay to ensure metadata is saved
    } catch (error) {
      console.error('[UnitTestPanel] Error generating tests:', error);
      setIsGenerating(false);
    }
  };

  if (!cell) {
    return (
      <div className="flowbook-unittest-empty">
        <p>No cell selected.</p>
        <p>Select a code cell to view or edit its unit tests.</p>
      </div>
    );
  }

  return (
    <div className="flowbook-unittest-content">
      <div className="flowbook-unittest-header">
        <div className="flowbook-unittest-cell-id">
          Cell: {cell.model.id.substring(0, 8)}...
        </div>
      </div>

      <div className="flowbook-unittest-actions">
        <button onClick={addTest} className="flowbook-unittest-btn-add">
          + Add Test
        </button>
        <button
          className="flowbook-unittest-btn-generate"
          onClick={handleGenerateTests}
          disabled={isGenerating}
        >
          {isGenerating ? '⏳ Generating...' : '⚡ Auto-generate'}
        </button>
      </div>

      <div className="flowbook-unittest-list">
        {tests.length === 0 ? (
          <div className="flowbook-unittest-empty-list">
            No tests defined. Click &quot;+ Add Test&quot; to create one.
          </div>
        ) : (
          tests.map((test, index) => (
            <div key={index} className="flowbook-unittest-item">
              <div className="flowbook-unittest-item-header">
                <input
                  type="text"
                  className="flowbook-unittest-title-input"
                  value={test.title}
                  onChange={e => updateTest(index, 'title', e.target.value)}
                  placeholder="Test title"
                />
                <button
                  onClick={() => deleteTest(index)}
                  className="flowbook-unittest-btn-delete"
                >
                  🗑️
                </button>
              </div>

              <div className="flowbook-unittest-field">
                <label>Description:</label>
                <textarea
                  className="flowbook-unittest-textarea"
                  value={test.description}
                  onChange={e =>
                    updateTest(index, 'description', e.target.value)
                  }
                  placeholder="What does this test verify?"
                  rows={2}
                />
              </div>

              <div className="flowbook-unittest-field">
                <label>Setup Code:</label>
                <textarea
                  className="flowbook-unittest-code-editor"
                  value={test.setup_code}
                  onChange={e =>
                    updateTest(index, 'setup_code', e.target.value)
                  }
                  placeholder="# Set up test globals here"
                  rows={4}
                  spellCheck={false}
                />
              </div>

              <div className="flowbook-unittest-field">
                <label>Assertion Code:</label>
                <textarea
                  className="flowbook-unittest-code-editor"
                  value={test.assertion_code}
                  onChange={e =>
                    updateTest(index, 'assertion_code', e.target.value)
                  }
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
  private _app: JupyterFrontEnd | null = null;
  private _tracker: INotebookTracker | null = null;

  constructor(app: JupyterFrontEnd, tracker: INotebookTracker) {
    super();
    this.id = 'flowbook-unittest-panel';
    this.addClass('flowbook-unittest-panel');
    this.title.label = 'Unit Tests';
    this.title.closable = true;
    this.title.caption = 'Edit unit tests for the selected cell';

    this._app = app;
    this._tracker = tracker;

    this.render();
  }

  private render(): void {
    ReactDOM.render(
      <UnitTestEditor
        cell={this._cell}
        app={this._app}
        tracker={this._tracker}
      />,
      this.node
    );
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
