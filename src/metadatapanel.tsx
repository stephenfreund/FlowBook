import { Widget } from '@lumino/widgets';
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { IFerretMetadata } from './types';

/**
 * React component for displaying Ferret metadata
 */
interface IMetadataDisplayProps {
  metadata: IFerretMetadata | null;
  cellId: string | null;
}

const MetadataDisplay: React.FC<IMetadataDisplayProps> = ({ metadata, cellId }) => {
  const [profileExpanded, setProfileExpanded] = React.useState(false);
  const [envExpanded, setEnvExpanded] = React.useState(false);
  const [envAfterExpanded, setEnvAfterExpanded] = React.useState(false);

  console.log('[MetadataDisplay] Rendering with metadata:', metadata);

  if (!metadata || (!metadata.optimization_potential && !metadata.profile)) {
    console.log('[MetadataDisplay] No metadata, showing empty state');
    return (
      <div className="ferret-metadata-empty">
        <p>No Ferret metadata available.</p>
        <p>Select a cell with optimization potential or profile data to view its metadata.</p>
      </div>
    );
  }

  console.log('[MetadataDisplay] Has metadata, rendering content');

  return (
    <div className="ferret-metadata-content">
      {/* Cell ID Header */}
      {cellId && (
        <>
          <div className="ferret-metadata-header">Cell ID: {cellId}</div>
          <div className="ferret-metadata-divider" />
        </>
      )}
      {/* Profile Metadata Section */}
      {metadata.profile && (
        <>
          <div className="ferret-metadata-header">Profile Metadata</div>
          <div className="ferret-metadata-divider" />

          <div className="ferret-metadata-section">
            <div className="ferret-metadata-item">
              <strong>Duration:</strong> {metadata.profile.duration.toFixed(3)}s
            </div>
          </div>

            <div className="ferret-metadata-divider" />
            <div className="ferret-metadata-section">
              <div
                className="ferret-metadata-profile-toggle"
                onClick={() => setEnvExpanded(!envExpanded)}
              >
                <strong>Environment (Before)</strong>
                <span className="ferret-metadata-toggle-icon">
                  {envExpanded ? '▼' : '▶'}
                </span>
              </div>
              {envExpanded && (
                <pre className="ferret-metadata-profile-output">
                  {Object.entries(metadata.profile.env).map(([key, value]) => (
                    <div key={key}>
                      <strong>{key}:</strong> {value}
                    </div>
                  ))}
                </pre>
              )}
            </div>

            <div className="ferret-metadata-divider" />
            <div className="ferret-metadata-section">
              <div
                className="ferret-metadata-profile-toggle"
                onClick={() => setEnvAfterExpanded(!envAfterExpanded)}
              >
                <strong>Environment Changes</strong>
                <span className="ferret-metadata-toggle-icon">
                  {envAfterExpanded ? '▼' : '▶'}
                </span>
              </div>
              {envAfterExpanded && (() => {
                const envBefore = metadata.profile.env;
                const envAfter = metadata.profile.env_after;
                const additions = Object.entries(envAfter).filter(([key]) => !(key in envBefore));
                const removals = Object.entries(envBefore).filter(([key]) => !(key in envAfter));

                return (
                  <pre className="ferret-metadata-profile-output">
                    {additions.length > 0 && (
                      <>
                        <div style={{ color: 'green', marginBottom: '0.5em' }}>
                          <strong>Added:</strong>
                        </div>
                        {additions.map(([key, value]) => (
                          <div key={key} style={{ paddingLeft: '1em' }}>
                            <strong>{key}:</strong> {value}
                          </div>
                        ))}
                      </>
                    )}
                    {removals.length > 0 && (
                      <>
                        <div style={{ color: 'red', marginTop: additions.length > 0 ? '0.5em' : '0', marginBottom: '0.5em' }}>
                          <strong>Removed:</strong>
                        </div>
                        {removals.map(([key, value]) => (
                          <div key={key} style={{ paddingLeft: '1em' }}>
                            <strong>{key}:</strong> {value}
                          </div>
                        ))}
                      </>
                    )}
                    {additions.length === 0 && removals.length === 0 && (
                      <div style={{ fontStyle: 'italic', color: '#666' }}>
                        No variables added or removed
                      </div>
                    )}
                  </pre>
                );
              })()}
            </div>

          {metadata.profile.profile && metadata.profile.profile.trim() && (
            <>
              <div className="ferret-metadata-divider" />
              <div className="ferret-metadata-section">
                <div
                  className="ferret-metadata-profile-toggle"
                  onClick={() => setProfileExpanded(!profileExpanded)}
                >
                  <strong>Profile Output</strong>
                  <span className="ferret-metadata-toggle-icon">
                    {profileExpanded ? '▼' : '▶'}
                  </span>
                </div>
                {profileExpanded && (
                  <pre className="ferret-metadata-profile-output">
                    {metadata.profile.profile}
                  </pre>
                )}
              </div>
            </>
          )}
        </>
      )}

      {/* Optimization Potential Metadata Section */}
      {metadata.optimization_potential && (
        <>
          {metadata.optimization_potential && <div className="ferret-metadata-divider" />}

          <div className="ferret-metadata-header">Optimization Potential</div>
          <div className="ferret-metadata-divider" />

          <div className="ferret-metadata-section">
            <div className="ferret-metadata-item">
              <strong>Potential:</strong> {metadata.optimization_potential.potential}
            </div>
          </div>

          {metadata.optimization_potential.optimization_plan &&
            metadata.optimization_potential.optimization_plan.length > 0 && (
              <>
                <div className="ferret-metadata-divider" />
                <div className="ferret-metadata-section">
                  <div className="ferret-metadata-item">
                    <strong>Optimization Plan:</strong>
                  </div>
                  <ul className="ferret-metadata-list">
                    {metadata.optimization_potential.optimization_plan.map((step, index) => (
                      <li key={index}>
                        <strong>Cell {step.target_cell_id}</strong>
                        {step.function_name && (
                          <> / <strong>Function {step.function_name}</strong></>
                        )}
                        : {step.description}
                      </li>
                    ))}
                  </ul>
                </div>
              </>
            )}
        </>
      )}
    </div>
  );
};

/**
 * Panel widget for displaying Ferret cell metadata
 */
export class FerretMetadataPanel extends Widget {
  private _metadata: IFerretMetadata | null = null;
  private _cellId: string | null = null;

  constructor() {
    super();
    this.id = 'ferret-metadata-panel';
    this.addClass('ferret-metadata-panel');
    this.title.label = 'Ferret Metadata';
    this.title.closable = true;
    this.title.caption = 'Ferret cell optimization potential metadata';

    this.render();
  }

  /**
   * Render the React component into the panel
   */
  private render(): void {
    ReactDOM.render(<MetadataDisplay metadata={this._metadata} cellId={this._cellId} />, this.node);
  }

  /**
   * Update the displayed metadata
   */
  public updateMetadata(metadata: IFerretMetadata | null, cellId: string | null): void {
    console.log('[MetadataPanel] updateMetadata called with:', metadata, cellId);
    this._metadata = metadata;
    this._cellId = cellId;
    this.render();
  }

  /**
   * Clear the displayed metadata
   */
  public clear(): void {
    console.log('[MetadataPanel] clear called');
    this._metadata = null;
    this._cellId = null;
    this.render();
  }

  /**
   * Dispose of the panel and clean up resources
   */
  dispose(): void {
    if (this.isDisposed) {
      return;
    }

    ReactDOM.unmountComponentAtNode(this.node);
    super.dispose();
  }
}
