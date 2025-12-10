/**
 * SDC Metadata Panel - Shows SDC-specific cell metadata
 */

import { Widget } from '@lumino/widgets';
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { ISDCMetadata } from './types';

interface ISDCMetadataDisplayProps {
  metadata: ISDCMetadata | null;
  cellId: string | null;
}

const SDCMetadataDisplay: React.FC<ISDCMetadataDisplayProps> = ({ metadata, cellId }) => {
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
          <div className="sdc-metadata-header">Cell: {cellId}</div>
          <div className="sdc-metadata-divider" />
        </>
      )}

      {/* Execution Info */}
      <div className="sdc-metadata-section">
        <div className="sdc-metadata-item">
          <strong>Execution #:</strong> {metadata.execution_seq}
        </div>
      </div>

      {/* Reads */}
      <div className="sdc-metadata-divider" />
      <div className="sdc-metadata-section">
        <div className="sdc-metadata-item">
          <strong>Variables Read:</strong>
          {metadata.reads.length > 0 ? (
            <ul className="sdc-variable-list">
              {metadata.reads.map((v, i) => <li key={i}><code>{v}</code></li>)}
            </ul>
          ) : (
            <span className="sdc-none"> None</span>
          )}
        </div>
      </div>

      {/* Writes */}
      <div className="sdc-metadata-divider" />
      <div className="sdc-metadata-section">
        <div className="sdc-metadata-item">
          <strong>Variables Written:</strong>
          {metadata.writes.length > 0 ? (
            <ul className="sdc-variable-list">
              {metadata.writes.map((v, i) => <li key={i}><code>{v}</code></li>)}
            </ul>
          ) : (
            <span className="sdc-none"> None</span>
          )}
        </div>
      </div>

      {/* Changed Variables */}
      {metadata.changed_variables.length > 0 && (
        <>
          <div className="sdc-metadata-divider" />
          <div className="sdc-metadata-section">
            <div className="sdc-metadata-item">
              <strong>Changed:</strong>
              <ul className="sdc-variable-list sdc-changed">
                {metadata.changed_variables.map((v, i) => <li key={i}><code>{v}</code></li>)}
              </ul>
            </div>
          </div>
        </>
      )}

      {/* Stale Cells */}
      {hasStale && (
        <>
          <div className="sdc-metadata-divider" />
          <div className="sdc-metadata-section sdc-stale-section">
            <div className="sdc-metadata-item">
              <strong>Stale Cells:</strong>
              <ul className="sdc-cell-list sdc-stale">
                {metadata.stale_cells.map((id, i) => <li key={i}><code>{id}</code></li>)}
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
                <strong>Mutating Cell:</strong> <code>{metadata.violation.mutating_cell}</code><br />
                <strong>Affected Cell:</strong> <code>{metadata.violation.affected_cell}</code><br />
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
      <SDCMetadataDisplay metadata={this._metadata} cellId={this._cellId} />,
      this.node
    );
  }

  public updateMetadata(metadata: ISDCMetadata | null, cellId: string | null): void {
    this._metadata = metadata;
    this._cellId = cellId;
    this.render();
  }

  public clear(): void {
    this._metadata = null;
    this._cellId = null;
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
