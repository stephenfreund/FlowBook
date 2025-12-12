/**
 * SDC Metadata Panel - Shows SDC-specific cell metadata
 */

import { Widget } from '@lumino/widgets';
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { ISDCMetadata } from './types';
import { indexToAlpha } from '../cellindexutils';

interface ISDCMetadataDisplayProps {
  metadata: ISDCMetadata | null;
  cellId: string | null;
  currentCellOrder: string[];  // Current cell order from notebook (not historical)
}

/**
 * Convert cell ID to reference (@A notation) using cell order.
 * @param cellId - The cell ID to convert
 * @param cellOrder - Array of cell IDs in notebook order
 * @returns Cell reference in @A notation, or the original cell ID if not found
 */
function cellIdToReference(cellId: string, cellOrder: string[]): string {
  const index = cellOrder.indexOf(cellId);
  if (index === -1) {
    // Cell not in order, return ID as-is
    return cellId;
  }
  try {
    return indexToAlpha(index);
  } catch (e) {
    // Fallback to ID if conversion fails
    return cellId;
  }
}

/**
 * Flatten column tracking to dot notation list.
 * Converts variable-level tracking with column info into a flat list like:
 * ["df.price", "df.quantity", "config"]
 */
function flattenColumnTracking(
  varList: string[],
  columnReads: { [key: string]: string[] } | undefined,
  columnWrites: { [key: string]: string[] } | undefined
): string[] {
  const result: string[] = [];

  // Handle undefined columnReads/columnWrites (backwards compatibility)
  const safeColumnReads = columnReads || {};
  const safeColumnWrites = columnWrites || {};

  for (const varName of varList) {
    const readCols = safeColumnReads[varName] || [];
    const writeCols = safeColumnWrites[varName] || [];
    const allCols = new Set([...readCols, ...writeCols]);

    if (allCols.size > 0) {
      // Add dot notation: "df.price", "df.qty"
      for (const col of Array.from(allCols).sort()) {
        result.push(`${varName}.${col}`);
      }
    } else {
      // No column tracking - just add variable
      result.push(varName);
    }
  }

  return result;
}

const SDCMetadataDisplay: React.FC<ISDCMetadataDisplayProps> = ({ metadata, cellId, currentCellOrder }) => {
  if (!metadata) {
    return (
      <div className="sdc-metadata-empty">
        <p>No SDC metadata available.</p>
        <p>Execute a cell to see dependency tracking.</p>
      </div>
    );
  }

  const hasViolation = metadata.violation !== null;
  const hasStale = metadata.stale_cells.length > 0;

  return (
    <div className="sdc-metadata-content">
      {/* Cell ID Header */}
      {cellId && (
        <>
          <div className="sdc-metadata-header">Cell: {cellIdToReference(cellId, currentCellOrder)}</div>
          <div className="sdc-metadata-section">
            <div className="sdc-metadata-item">
              <span style={{ fontSize: '0.85em', color: '#666' }}>Id: </span>
              <code style={{ fontSize: '0.85em', color: '#666' }}>{cellId}</code>
            </div>
          </div>
          <div className="sdc-metadata-divider" />
        </>
      )}

      {/* Execution Info */}
      <div className="sdc-metadata-section">
        <div className="sdc-metadata-item">
          <strong>Execution #:</strong> {metadata.execution_seq}
        </div>
      </div>

      {/* Timing Info */}
      {(metadata.run_duration_ms !== undefined ||
        metadata.state_duration_ms !== undefined ||
        metadata.check_duration_ms !== undefined) && (
        <>
          <div className="sdc-metadata-divider" />
          <div className="sdc-metadata-section">
            <div className="sdc-metadata-item">
              <strong>Timing:</strong>
              <ul className="sdc-timing-list">
                {metadata.run_duration_ms !== undefined && (
                  <li>Run: <code>{metadata.run_duration_ms.toFixed(0)} ms</code></li>
                )}
                {metadata.state_duration_ms !== undefined && (
                  <li>State: <code>{metadata.state_duration_ms.toFixed(0)} ms</code></li>
                )}
                {metadata.check_duration_ms !== undefined && (
                  <li>Check: <code>{metadata.check_duration_ms.toFixed(0)} ms</code></li>
                )}
              </ul>
            </div>
          </div>
        </>
      )}

      {/* Reads */}
      <div className="sdc-metadata-divider" />
      <div className="sdc-metadata-section">
        <div className="sdc-metadata-item">
          <strong>Variables Read:</strong>
          {(() => {
            // Include variables with either variable-level OR column-level reads
            const readVars = new Set([
              ...metadata.reads,
              ...Object.keys(metadata.column_reads || {})
            ]);
            const flatReads = flattenColumnTracking(
              Array.from(readVars),
              metadata.column_reads,
              {}
            );
            return flatReads.length > 0 ? (
              <ul className="sdc-variable-list">
                {flatReads.map((v, i) => <li key={i}><code>{v}</code></li>)}
              </ul>
            ) : (
              <span className="sdc-none"> None</span>
            );
          })()}
        </div>
      </div>

      {/* Writes */}
      <div className="sdc-metadata-divider" />
      <div className="sdc-metadata-section">
        <div className="sdc-metadata-item">
          <strong>Variables Written:</strong>
          {(() => {
            // Include variables with either variable-level OR column-level writes
            const writeVars = new Set([
              ...metadata.writes,
              ...Object.keys(metadata.column_writes || {})
            ]);
            const flatWrites = flattenColumnTracking(
              Array.from(writeVars),
              {},
              metadata.column_writes
            );
            return flatWrites.length > 0 ? (
              <ul className="sdc-variable-list">
                {flatWrites.map((v, i) => <li key={i}><code>{v}</code></li>)}
              </ul>
            ) : (
              <span className="sdc-none"> None</span>
            );
          })()}
        </div>
      </div>

      {/* Changed Variables */}
      {(() => {
        // Include variables with either variable-level OR column-level changes
        const changedVars = new Set([
          ...metadata.changed_variables,
          ...Object.keys(metadata.column_changed || {})
        ]);
        const flatChanged = flattenColumnTracking(
          Array.from(changedVars),
          {},
          metadata.column_changed
        );
        return flatChanged.length > 0 && (
          <>
            <div className="sdc-metadata-divider" />
            <div className="sdc-metadata-section">
              <div className="sdc-metadata-item">
                <strong>Changed:</strong>
                <ul className="sdc-variable-list sdc-changed">
                  {flatChanged.map((v, i) => <li key={i}><code>{v}</code></li>)}
                </ul>
              </div>
            </div>
          </>
        );
      })()}

      {/* Stale Cells */}
      {hasStale && (
        <>
          <div className="sdc-metadata-divider" />
          <div className="sdc-metadata-section sdc-stale-section">
            <div className="sdc-metadata-item">
              <strong>Stale Cells:</strong>
              <ul className="sdc-cell-list sdc-stale">
                {metadata.stale_cells.map((id, i) => (
                  <li key={i}><code>{cellIdToReference(id, currentCellOrder)}</code></li>
                ))}
              </ul>
            </div>
          </div>
        </>
      )}

      {/* Violation */}
      {hasViolation && metadata.violation && (
        <>
          <div className="sdc-metadata-divider" />
          <div className="sdc-metadata-section sdc-violation-section">
            <div className="sdc-violation-header">SDC Violation</div>
            <div className="sdc-violation-content">
              <p>{metadata.violation.message}</p>
              <div className="sdc-violation-details">
                <strong>Mutating Cell:</strong> <code>{cellIdToReference(metadata.violation.mutating_cell, currentCellOrder)}</code><br />
                <strong>Affected Cell:</strong> <code>{cellIdToReference(metadata.violation.affected_cell, currentCellOrder)}</code><br />
                <strong>Variables:</strong> {metadata.violation.variables.join(', ')}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export class SDCMetadataPanel extends Widget {
  private _metadata: ISDCMetadata | null = null;
  private _cellId: string | null = null;
  private _currentCellOrder: string[] = [];

  constructor() {
    super();
    this.id = 'sdc-metadata-panel';
    this.addClass('sdc-metadata-panel');
    this.title.label = 'SDC Metadata';
    this.title.closable = true;
    this.title.caption = 'Sequential Dataflow Consistency cell metadata';
    this.render();
  }

  private render(): void {
    ReactDOM.render(
      <SDCMetadataDisplay
        metadata={this._metadata}
        cellId={this._cellId}
        currentCellOrder={this._currentCellOrder}
      />,
      this.node
    );
  }

  public updateMetadata(metadata: ISDCMetadata | null, cellId: string | null, currentCellOrder: string[]): void {
    this._metadata = metadata;
    this._cellId = cellId;
    this._currentCellOrder = currentCellOrder;
    this.render();
  }

  public clear(): void {
    this._metadata = null;
    this._cellId = null;
    this._currentCellOrder = [];
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
