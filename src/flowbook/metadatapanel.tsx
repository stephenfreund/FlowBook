/**
 * Reproducibility Panel - Shows reproducibility-specific cell metadata
 *
 * Displays typed ReadLoc/WriteLoc sets grouped by variable, with loc type
 * annotations for column, attribute, and file access.
 */

import { Widget } from '@lumino/widgets';
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { IReproducibilityMetadata, IReadLoc, IWriteLoc } from './types';
import { indexToAlpha } from '../cellindexutils';

interface IReproducibilityMetadataDisplayProps {
  metadata: IReproducibilityMetadata | null;
  cellId: string | null;
  currentCellOrder: string[]; // Current cell order from notebook (not historical)
}

/**
 * Convert cell ID to reference (@A notation) using cell order.
 */
function cellIdToReference(cellId: string, cellOrder: string[]): string {
  const index = cellOrder.indexOf(cellId);
  if (index === -1) {
    return cellId;
  }
  try {
    return indexToAlpha(index, cellId);
  } catch (e) {
    return cellId;
  }
}

/**
 * Get display name for a loc qualifier (handles both string and numeric loc_ids).
 */
function displayQualifier(loc: { qualifier?: string | number; var_name?: string }): string | undefined {
  if (loc.var_name !== undefined) {
    return loc.var_name;
  }
  if (typeof loc.qualifier === 'string') {
    return loc.qualifier;
  }
  return undefined;
}

/**
 * Group read locs by variable, producing a map from variable name to its sub-locs.
 * Var(x) locs appear as standalone entries. Col/Attr locs are grouped under their qualifier.
 */
function groupReadLocs(
  locs: IReadLoc[]
): Map<string, { types: Map<string, string[]> }> {
  const groups = new Map<string, { types: Map<string, string[]> }>();

  for (const loc of locs) {
    const q = displayQualifier(loc);
    if (q) {
      // Col(d,c) or Attr(d,a) — group under variable name
      let group = groups.get(q);
      if (!group) {
        group = { types: new Map() };
        groups.set(q, group);
      }
      const typeLabel = loc.type === 'col' ? 'Col' : 'Attr';
      let names = group.types.get(typeLabel);
      if (!names) {
        names = [];
        group.types.set(typeLabel, names);
      }
      names.push(loc.name);
    } else {
      // Var(x) or File(p) — standalone
      let group = groups.get(loc.name);
      if (!group) {
        group = { types: new Map() };
        groups.set(loc.name, group);
      }
      const typeLabel = loc.type === 'file' ? 'File' : 'Var';
      if (!group.types.has(typeLabel)) {
        group.types.set(typeLabel, []);
      }
    }
  }

  return groups;
}

/**
 * Group write locs by variable.
 */
function groupWriteLocs(
  locs: IWriteLoc[]
): Map<string, { types: Map<string, string[]> }> {
  const groups = new Map<string, { types: Map<string, string[]> }>();

  const typeLabels: Record<string, string> = {
    var: 'Var',
    col: 'Col',
    col_add: 'ColAdd',
    col_del: 'ColDel',
    rows: 'Rows',
    attr: 'Attr',
    file: 'File'
  };

  for (const loc of locs) {
    const label = typeLabels[loc.type] || loc.type;
    const q = displayQualifier(loc);
    if (q) {
      let group = groups.get(q);
      if (!group) {
        group = { types: new Map() };
        groups.set(q, group);
      }
      let names = group.types.get(label);
      if (!names) {
        names = [];
        group.types.set(label, names);
      }
      if (loc.type !== 'rows') {
        names.push(loc.name);
      }
    } else {
      let group = groups.get(loc.name);
      if (!group) {
        group = { types: new Map() };
        groups.set(loc.name, group);
      }
      if (!group.types.has(label)) {
        group.types.set(label, []);
      }
    }
  }

  return groups;
}

/**
 * Render a grouped loc map as a list.
 */
function renderLocGroups(
  groups: Map<string, { types: Map<string, string[]> }>
): React.ReactElement {
  if (groups.size === 0) {
    return <span className="flowbook-none"> None</span>;
  }

  return (
    <ul className="flowbook-variable-list">
      {Array.from(groups.entries()).map(([varName, group]) => {
        const hasSubItems = Array.from(group.types.values()).some(
          names => names.length > 0
        );

        if (!hasSubItems) {
          // Standalone: Var(x) or File(p) — just show the name with type
          const typeLabel = Array.from(group.types.keys())[0];
          const suffix =
            typeLabel && typeLabel !== 'Var' ? ` (${typeLabel})` : '';
          return (
            <li key={varName}>
              <code>
                {varName}
                {suffix && (
                  <span style={{ color: '#888', fontSize: '0.9em' }}>
                    {suffix}
                  </span>
                )}
              </code>
            </li>
          );
        }

        // Grouped: show variable with sub-items
        return (
          <li key={varName}>
            <code>{varName}</code>
            <ul className="flowbook-loc-sublist">
              {Array.from(group.types.entries()).map(([typeLabel, names]) => {
                if (names.length === 0) {
                  // Type with no sub-names (e.g., Rows)
                  return (
                    <li key={typeLabel}>
                      <span className="flowbook-loc-type">{typeLabel}</span>
                    </li>
                  );
                }
                return (
                  <li key={typeLabel}>
                    <span className="flowbook-loc-type">{typeLabel}:</span>{' '}
                    <code>{names.sort().join(', ')}</code>
                  </li>
                );
              })}
            </ul>
          </li>
        );
      })}
    </ul>
  );
}

/**
 * Check if two write loc lists are identical.
 */
function writeLocsEqual(a: IWriteLoc[], b: IWriteLoc[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    if (
      a[i].type !== b[i].type ||
      a[i].name !== b[i].name ||
      a[i].qualifier !== b[i].qualifier
    ) {
      return false;
    }
  }
  return true;
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

  const hasStale = metadata.stale_cells.length > 0;
  const readGroups = groupReadLocs(metadata.read_locs || []);
  const writeGroups = groupWriteLocs(metadata.write_locs || []);
  const changedGroups = groupWriteLocs(metadata.changed_locs || []);
  const writesAndChangedSame = writeLocsEqual(
    metadata.write_locs || [],
    metadata.changed_locs || []
  );

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
          <strong>Reads:</strong>
          {renderLocGroups(readGroups)}
        </div>
      </div>

      {/* Writes — show as single "Writes" if identical, else show both */}
      <div className="flowbook-metadata-divider" />
      <div className="flowbook-metadata-section">
        <div className="flowbook-metadata-item">
          <strong>{writesAndChangedSame ? 'Writes:' : 'Writes (Intended):'}</strong>
          {renderLocGroups(writeGroups)}
        </div>
      </div>

      {!writesAndChangedSame && (
        <>
          <div className="flowbook-metadata-divider" />
          <div className="flowbook-metadata-section">
            <div className="flowbook-metadata-item">
              <strong>Changed (Actual):</strong>
              {renderLocGroups(changedGroups)}
            </div>
          </div>
        </>
      )}

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
    this.title.label = 'FlowBook';
    this.title.closable = true;
    this.title.caption = 'FlowBook cell metadata';
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
