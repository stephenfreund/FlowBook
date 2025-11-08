/**
 * History Panel for Ferret - Photoshop-style undo/redo
 */

import React, { useState, useEffect } from 'react';
import { ReactWidget } from '@jupyterlab/apputils';
import { INotebookTracker } from '@jupyterlab/notebook';
import { NotebookHistoryManager } from './history';
import { IHistoryEntry } from './types';
import { LabIcon } from '@jupyterlab/ui-components';

interface HistoryPanelProps {
  tracker: INotebookTracker;
  historyManager: NotebookHistoryManager;
}

/**
 * React component for the history panel
 */
function HistoryPanelComponent(props: HistoryPanelProps): JSX.Element {
  const { tracker, historyManager } = props;
  const [entries, setEntries] = useState<IHistoryEntry[]>([]);
  const [currentIndex, setCurrentIndex] = useState<number>(-1);
  const [notebookPath, setNotebookPath] = useState<string>('');

  // Track current widget for notebook change listener
  const [currentWidget, setCurrentWidget] = useState(tracker.currentWidget);

  // Update history when active notebook changes or history changes
  useEffect(() => {
    const updateHistory = () => {
      const current = tracker.currentWidget;
      setCurrentWidget(current || null);
      if (current) {
        const path = current.context.path;
        setNotebookPath(path);
        setEntries(historyManager.getHistory(path));
        setCurrentIndex(historyManager.getCurrentIndex(path));
      } else {
        setNotebookPath('');
        setEntries([]);
        setCurrentIndex(-1);
      }
    };

    // Initial update
    updateHistory();

    // Listen to tracker changes
    tracker.currentChanged.connect(updateHistory);

    // Listen to history changes
    const onHistoryChanged = (sender: NotebookHistoryManager, path: string) => {
      if (path === tracker.currentWidget?.context.path) {
        updateHistory();
      }
    };
    historyManager.historyChanged.connect(onHistoryChanged);

    return () => {
      tracker.currentChanged.disconnect(updateHistory);
      historyManager.historyChanged.disconnect(onHistoryChanged);
    };
  }, [tracker, historyManager]);

  // Listen to notebook cell changes to update displayed indices
  useEffect(() => {
    if (!currentWidget?.content.model) {
      return;
    }

    const onNotebookChanged = () => {
      // Force a re-render by updating state to refresh displayed indices
      setEntries([...historyManager.getHistory(currentWidget.context.path)]);
    };

    currentWidget.content.model.cells.changed.connect(onNotebookChanged);

    return () => {
      if (currentWidget?.content.model) {
        try {
          currentWidget.content.model.cells.changed.disconnect(onNotebookChanged);
        } catch (e) {
          // Ignore if already disconnected
        }
      }
    };
  }, [currentWidget, historyManager]);

  const handleEntryClick = (index: number) => {
    const current = tracker.currentWidget;
    if (current && notebookPath) {
      historyManager.jumpToEntry(notebookPath, index, current);
    }
  };

  const handleUndo = () => {
    const current = tracker.currentWidget;
    if (current && notebookPath) {
      historyManager.undo(notebookPath, current);
    }
  };

  const handleRedo = () => {
    const current = tracker.currentWidget;
    if (current && notebookPath) {
      historyManager.redo(notebookPath, current);
    }
  };

  const canUndo = historyManager.canUndo(notebookPath);
  const canRedo = historyManager.canRedo(notebookPath);

  if (!notebookPath) {
    return (
      <div className="ferret-history-panel">
        <div className="ferret-history-empty">
          No notebook open
        </div>
      </div>
    );
  }

  if (entries.length === 0) {
    return (
      <div className="ferret-history-panel">
        <div className="ferret-history-empty">
          No history yet
        </div>
      </div>
    );
  }

  // Get current notebook snapshot for dynamic descriptions
  const currentNotebook = tracker.currentWidget?.content.model?.toJSON();

  return (
    <div className="ferret-history-panel">
      <div className="ferret-history-controls">
        <button
          className="ferret-history-button"
          onClick={handleUndo}
          disabled={!canUndo}
          title="Undo"
        >
          ← Undo
        </button>
        <button
          className="ferret-history-button"
          onClick={handleRedo}
          disabled={!canRedo}
          title="Redo"
        >
          Redo →
        </button>
      </div>
      <div className="ferret-history-list">
        {entries.map((entry, index) => {
          const isCurrent = index === currentIndex;
          const isFuture = index > currentIndex;
          const entryClass = `ferret-history-entry ${isCurrent ? 'current' : ''} ${isFuture ? 'future' : ''}`.trim();

          // Get dynamic description based on current notebook state
          const description = currentNotebook
            ? historyManager.getDynamicDescription(entry, currentNotebook)
            : entry.description;

          if (index === 0) {
            console.log('[HistoryPanel] First entry - index:', index, 'currentIndex:', currentIndex, 'isCurrent:', isCurrent, 'class:', entryClass);
          }

          return (
            <div
              key={entry.id}
              className={entryClass}
              onClick={() => handleEntryClick(index)}
              title={description}
            >
              <div className="ferret-history-entry-icon">
                <LabIcon.resolveReact
                  icon={entry.icon}
                  className="ferret-history-icon"
                  tag="span"
                />
              </div>
              <div className="ferret-history-entry-content">
                <div className="ferret-history-entry-label">
                  {entry.type === 'command' ? entry.commandLabel : 'User Edit'}
                </div>
                <div className="ferret-history-entry-description">
                  {description}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/**
 * Widget wrapper for the history panel
 */
export class HistoryPanel extends ReactWidget {
  private tracker: INotebookTracker;
  private historyManager: NotebookHistoryManager;

  constructor(tracker: INotebookTracker, historyManager: NotebookHistoryManager) {
    super();
    this.id = 'ferret-history-panel';
    this.tracker = tracker;
    this.historyManager = historyManager;
    this.addClass('ferret-history-widget');
    this.title.label = 'Ferret History';
    this.title.icon = LabIcon.resolve({ icon: 'ui-components:list' });
    this.title.closable = true;
  }

  render(): JSX.Element {
    return (
      <HistoryPanelComponent
        tracker={this.tracker}
        historyManager={this.historyManager}
      />
    );
  }
}
