/**
 * Reproducibility Panel - Shows reproducibility-specific cell metadata
 */

import { Widget } from '@lumino/widgets';
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { IReproducibilityMetadata } from './types';
import { indexToAlpha } from '../cellindexutils';

interface IReproducibilityMetadataDisplayProps {
  metadata: IReproducibilityMetadata | null;
  cellId: string | null;
  currentCellOrder: string[]; // Current cell order from notebook (not historical)
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
    return indexToAlpha(index, cellId);
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

const ReproducibilityMetadataDisplay: React.FC<
  IReproducibilityMetadataDisplayProps
> = ({ metadata, cellId, currentCellOrder }) => {
  if (!metadata) {
    return (
      <div className="flowbook-metadata-empty">
        <p>No reproducibility metadata available.</p>
        <p>Execute a cell to see dependency tracking.</p>
      </div>
    );
  }

  const hasViolation = metadata.violation !== null;
  const hasStale = metadata.stale_cells.length > 0;

  return (
    <div className="flowbook-metadata-content">
      {/* Cell ID Header */}
      {cellId && (
        <>
          <div className="flowbook-metadata-header">
            Cell: {cellIdToReference(cellId, currentCellOrder)}
          </div>
          <div className="flowbook-metadata-section">
            <div className="flowbook-metadata-item">
              <span style={{ fontSize: '0.85em', color: '#666' }}>Id: </span>
              <code style={{ fontSize: '0.85em', color: '#666' }}>
                {cellId}
              </code>
            </div>
          </div>
          <div className="flowbook-metadata-divider" />
        </>
      )}

      {/* Execution Info */}
      <div className="flowbook-metadata-section">
        <div className="flowbook-metadata-item">
          <strong>Execution #:</strong> {metadata.execution_seq}
        </div>
      </div>

      {/* Timing Info */}
      {(metadata.execute_duration_ms !== undefined ||
        metadata.code_duration_ms !== undefined ||
        metadata.state_duration_ms !== undefined ||
        metadata.check_duration_ms !== undefined) && (
        <>
          <div className="flowbook-metadata-divider" />
          <div className="flowbook-metadata-section">
            <div className="flowbook-metadata-item">
              <strong>Timing:</strong>
              <ul className="flowbook-timing-list">
                {metadata.execute_duration_ms !== undefined && (
                  <li>
                    Execute:{' '}
                    <code>{metadata.execute_duration_ms.toFixed(0)} ms</code>
                  </li>
                )}
                {metadata.code_duration_ms !== undefined && (
                  <li>
                    Code: <code>{metadata.code_duration_ms.toFixed(0)} ms</code>
                  </li>
                )}
                {metadata.state_duration_ms !== undefined && (
                  <li>
                    State:{' '}
                    <code>{metadata.state_duration_ms.toFixed(0)} ms</code>
                  </li>
                )}
                {metadata.check_duration_ms !== undefined && (
                  <li>
                    Check:{' '}
                    <code>{metadata.check_duration_ms.toFixed(0)} ms</code>
                  </li>
                )}
              </ul>
            </div>
          </div>
        </>
      )}

      {/* Reads */}
      <div className="flowbook-metadata-divider" />
      <div className="flowbook-metadata-section">
        <div className="flowbook-metadata-item">
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
              <ul className="flowbook-variable-list">
                {flatReads.map((v, i) => (
                  <li key={i}>
                    <code>{v}</code>
                  </li>
                ))}
              </ul>
            ) : (
              <span className="flowbook-none"> None</span>
            );
          })()}
        </div>
      </div>

      {/* Structural Reads */}
      {metadata.structural_reads &&
        Object.keys(metadata.structural_reads).length > 0 && (
          <>
            <div className="flowbook-metadata-divider" />
            <div className="flowbook-metadata-section">
              <div className="flowbook-metadata-item">
                <strong>Structural Reads:</strong>
                <ul className="flowbook-variable-list flowbook-structural">
                  {Object.entries(metadata.structural_reads).flatMap(
                    ([varName, attrs]) =>
                      attrs.map((attr, i) => (
                        <li key={`${varName}.${attr}`}>
                          <code>
                            {varName}.{attr}
                          </code>
                        </li>
                      ))
                  )}
                </ul>
              </div>
            </div>
          </>
        )}

      {/* Writes */}
      <div className="flowbook-metadata-divider" />
      <div className="flowbook-metadata-section">
        <div className="flowbook-metadata-item">
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
              <ul className="flowbook-variable-list">
                {flatWrites.map((v, i) => (
                  <li key={i}>
                    <code>{v}</code>
                  </li>
                ))}
              </ul>
            ) : (
              <span className="flowbook-none"> None</span>
            );
          })()}
        </div>
      </div>

      {/* File Reads */}
      {metadata.file_reads && metadata.file_reads.length > 0 && (
        <>
          <div className="flowbook-metadata-divider" />
          <div className="flowbook-metadata-section">
            <div className="flowbook-metadata-item">
              <strong>Files Read:</strong>
              <ul className="flowbook-variable-list">
                {metadata.file_reads.map((f, i) => (
                  <li key={i}>
                    <code>{f}</code>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </>
      )}

      {/* File Writes */}
      {metadata.file_writes && metadata.file_writes.length > 0 && (
        <>
          <div className="flowbook-metadata-divider" />
          <div className="flowbook-metadata-section">
            <div className="flowbook-metadata-item">
              <strong>Files Written:</strong>
              <ul className="flowbook-variable-list">
                {metadata.file_writes.map((f, i) => (
                  <li key={i}>
                    <code>{f}</code>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </>
      )}

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
        return (
          flatChanged.length > 0 && (
            <>
              <div className="flowbook-metadata-divider" />
              <div className="flowbook-metadata-section">
                <div className="flowbook-metadata-item">
                  <strong>Changed:</strong>
                  <ul className="flowbook-variable-list flowbook-changed">
                    {flatChanged.map((v, i) => (
                      <li key={i}>
                        <code>{v}</code>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </>
          )
        );
      })()}

      {/* Stale Cells */}
      {hasStale && (
        <>
          <div className="flowbook-metadata-divider" />
          <div className="flowbook-metadata-section flowbook-stale-section">
            <div className="flowbook-metadata-item">
              <strong>Stale Cells:</strong>
              <ul className="flowbook-cell-list flowbook-stale">
                {metadata.stale_cells.map((id, i) => (
                  <li key={i}>
                    <code>{cellIdToReference(id, currentCellOrder)}</code>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </>
      )}

      {/* Structural Warnings */}
      {metadata.structural_warnings &&
        metadata.structural_warnings.length > 0 && (
          <>
            <div className="flowbook-metadata-divider" />
            <div className="flowbook-metadata-section flowbook-warning-section">
              <div className="flowbook-warning-header">Structural Warnings</div>
              <div className="flowbook-warning-content">
                <ul className="flowbook-warning-list">
                  {metadata.structural_warnings.map((warning, i) => (
                    <li key={i}>{warning}</li>
                  ))}
                </ul>
              </div>
            </div>
          </>
        )}

      {/* Violation */}
      {hasViolation && metadata.violation && (
        <>
          <div className="flowbook-metadata-divider" />
          <div
            className={`flowbook-metadata-section ${
              metadata.violation.violation_type === 'forward_dependency'
                ? 'flowbook-forward-dep-section'
                : 'flowbook-violation-section'
            }`}
          >
            <div
              className={
                metadata.violation.violation_type === 'forward_dependency'
                  ? 'flowbook-forward-dep-header'
                  : 'flowbook-violation-header'
              }
            >
              {metadata.violation.violation_type === 'forward_dependency'
                ? 'Forward Dependency'
                : 'Reproducibility Violation'}
            </div>
            <div className="flowbook-violation-content">
              <p>{metadata.violation.message}</p>
              <div className="flowbook-violation-details">
                <strong>
                  {metadata.violation.violation_type === 'forward_dependency'
                    ? 'Writing Cell:'
                    : 'Mutating Cell:'}
                </strong>{' '}
                <code>
                  {cellIdToReference(
                    metadata.violation.mutating_cell,
                    currentCellOrder
                  )}
                </code>
                <br />
                <strong>
                  {metadata.violation.violation_type === 'forward_dependency'
                    ? 'Reading Cell:'
                    : 'Affected Cell:'}
                </strong>{' '}
                <code>
                  {cellIdToReference(
                    metadata.violation.affected_cell,
                    currentCellOrder
                  )}
                </code>
                <br />
                <strong>Variables:</strong>{' '}
                {metadata.violation.variables.join(', ')}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export class ReproducibilityMetadataPanel extends Widget {
  private _metadata: IReproducibilityMetadata | null = null;
  private _cellId: string | null = null;
  private _currentCellOrder: string[] = [];

  constructor() {
    super();
    this.id = 'flowbook-metadata-panel';
    this.addClass('flowbook-metadata-panel');
    this.title.label = 'Reproducibility';
    this.title.closable = true;
    this.title.caption = 'Reproducibility cell metadata';
    this.render();
  }

  private render(): void {
    ReactDOM.render(
      <ReproducibilityMetadataDisplay
        metadata={this._metadata}
        cellId={this._cellId}
        currentCellOrder={this._currentCellOrder}
      />,
      this.node
    );
  }

  public updateMetadata(
    metadata: IReproducibilityMetadata | null,
    cellId: string | null,
    currentCellOrder: string[]
  ): void {
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
