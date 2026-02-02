import { Widget } from '@lumino/widgets';
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { IFlowbookMetadata } from './types';

/**
 * React component for displaying FlowBook metadata
 */
interface IMetadataDisplayProps {
  metadata: IFlowbookMetadata | null;
  cellId: string | null;
}

const MetadataDisplay: React.FC<IMetadataDisplayProps> = ({ metadata, cellId }) => {
  const [profileExpanded, setProfileExpanded] = React.useState(false);
  const [envExpanded, setEnvExpanded] = React.useState(false);
  const [envAfterExpanded, setEnvAfterExpanded] = React.useState(false);
  const [originalCodeExpanded, setOriginalCodeExpanded] = React.useState(false);
  const [optimizedCodeExpanded, setOptimizedCodeExpanded] = React.useState(false);
  const [generatedCodeExpanded, setGeneratedCodeExpanded] = React.useState(false);

  console.log('[MetadataDisplay] Rendering with metadata:', metadata);

  if (!metadata || (!metadata.optimization_potential && !metadata.profile && !metadata.dynamic_dependencies && !metadata.generated && !metadata.optimized && !metadata.optimization_applied)) {
    console.log('[MetadataDisplay] No metadata, showing empty state');
    return (
      <div className="flowbook-metadata-empty">
        <p>No FlowBook metadata available.</p>
        <p>Select a cell with FlowBook metadata to view it.</p>
      </div>
    );
  }

  console.log('[MetadataDisplay] Has metadata, rendering content');

  return (
    <div className="flowbook-metadata-content">
      {/* Cell ID Header */}
      {cellId && (
        <>
          <div className="flowbook-metadata-header">Cell ID: {cellId}</div>
          <div className="flowbook-metadata-divider" />
        </>
      )}
      {/* Profile Metadata Section */}
      {metadata.profile && (
        <>
          <div className="flowbook-metadata-header">Profile Metadata</div>
          <div className="flowbook-metadata-divider" />

          <div className="flowbook-metadata-section">
            <div className="flowbook-metadata-item">
              <strong>Duration:</strong> {metadata.profile.duration.toFixed(3)}s
            </div>
          </div>

            <div className="flowbook-metadata-divider" />
            <div className="flowbook-metadata-section">
              <div
                className="flowbook-metadata-profile-toggle"
                onClick={() => setEnvExpanded(!envExpanded)}
              >
                <strong>Environment (Before)</strong>
                <span className="flowbook-metadata-toggle-icon">
                  {envExpanded ? '▼' : '▶'}
                </span>
              </div>
              {envExpanded && (
                <pre className="flowbook-metadata-profile-output">
                  {Object.entries(metadata.profile.env).map(([key, value]) => (
                    <div key={key}>
                      <strong>{key}:</strong> {value}
                    </div>
                  ))}
                </pre>
              )}
            </div>

            <div className="flowbook-metadata-divider" />
            <div className="flowbook-metadata-section">
              <div
                className="flowbook-metadata-profile-toggle"
                onClick={() => setEnvAfterExpanded(!envAfterExpanded)}
              >
                <strong>Environment Changes</strong>
                <span className="flowbook-metadata-toggle-icon">
                  {envAfterExpanded ? '▼' : '▶'}
                </span>
              </div>
              {envAfterExpanded && (() => {
                const envBefore = metadata.profile.env;
                const envAfter = metadata.profile.env_after;
                const additions = Object.entries(envAfter).filter(([key]) => !(key in envBefore));
                const removals = Object.entries(envBefore).filter(([key]) => !(key in envAfter));

                return (
                  <pre className="flowbook-metadata-profile-output">
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
              <div className="flowbook-metadata-divider" />
              <div className="flowbook-metadata-section">
                <div
                  className="flowbook-metadata-profile-toggle"
                  onClick={() => setProfileExpanded(!profileExpanded)}
                >
                  <strong>Profile Output</strong>
                  <span className="flowbook-metadata-toggle-icon">
                    {profileExpanded ? '▼' : '▶'}
                  </span>
                </div>
                {profileExpanded && (
                  <pre className="flowbook-metadata-profile-output">
                    {metadata.profile.profile}
                  </pre>
                )}
              </div>
            </>
          )}
        </>
      )}

      {/* Dynamic Dependencies Section */}
      {metadata.dynamic_dependencies && (
        <>
          <div className="flowbook-metadata-divider" />
          <div className="flowbook-metadata-header">Dynamic Dependencies</div>
          <div className="flowbook-metadata-divider" />

          <div className="flowbook-metadata-section">
            <div className="flowbook-metadata-item">
              <strong>Reads Before Writes:</strong>
              {metadata.dynamic_dependencies.reads_before_writes.length > 0 ? (
                <ul style={{ marginTop: '0.5em', paddingLeft: '1.5em' }}>
                  {metadata.dynamic_dependencies.reads_before_writes.map((varName, idx) => (
                    <li key={idx}><code>{varName}</code></li>
                  ))}
                </ul>
              ) : (
                <span style={{ fontStyle: 'italic', color: '#666', marginLeft: '0.5em' }}>
                  None
                </span>
              )}
            </div>
          </div>

          <div className="flowbook-metadata-divider" />
          <div className="flowbook-metadata-section">
            <div className="flowbook-metadata-item">
              <strong>Variables Written:</strong>
              {metadata.dynamic_dependencies.writes.length > 0 ? (
                <ul style={{ marginTop: '0.5em', paddingLeft: '1.5em' }}>
                  {metadata.dynamic_dependencies.writes.map((varName, idx) => (
                    <li key={idx}><code>{varName}</code></li>
                  ))}
                </ul>
              ) : (
                <span style={{ fontStyle: 'italic', color: '#666', marginLeft: '0.5em' }}>
                  None
                </span>
              )}
            </div>
          </div>

          {/* Column-level DataFrame reads tracking */}
          {metadata.dynamic_dependencies.column_reads_before_writes &&
           Object.entries(metadata.dynamic_dependencies.column_reads_before_writes).some(([, cols]) => cols.length > 0) && (
            <>
              <div className="flowbook-metadata-divider" />
              <div className="flowbook-metadata-section">
                <div className="flowbook-metadata-item">
                  <strong>DataFrame Columns Read:</strong>
                  <ul style={{ marginTop: '0.5em', paddingLeft: '1.5em' }}>
                    {Object.entries(metadata.dynamic_dependencies.column_reads_before_writes)
                      .filter(([, columns]) => columns.length > 0)
                      .map(([dfName, columns], idx) => (
                        <li key={idx}>
                          <code>{dfName}</code>
                          <ul style={{ paddingLeft: '1.5em', marginTop: '0.25em' }}>
                            {columns.map((col, colIdx) => (
                              <li key={colIdx} style={{ listStyleType: 'circle' }}>
                                <code>{col}</code>
                              </li>
                            ))}
                          </ul>
                        </li>
                      )
                    )}
                  </ul>
                </div>
              </div>
            </>
          )}

          {/* Column-level DataFrame writes tracking */}
          {metadata.dynamic_dependencies.column_writes &&
           Object.entries(metadata.dynamic_dependencies.column_writes).some(([, cols]) => cols.length > 0) && (
            <>
              <div className="flowbook-metadata-divider" />
              <div className="flowbook-metadata-section">
                <div className="flowbook-metadata-item">
                  <strong>DataFrame Columns Written:</strong>
                  <ul style={{ marginTop: '0.5em', paddingLeft: '1.5em' }}>
                    {Object.entries(metadata.dynamic_dependencies.column_writes)
                      .filter(([, columns]) => columns.length > 0)
                      .map(([dfName, columns], idx) => (
                        <li key={idx}>
                          <code>{dfName}</code>
                          <ul style={{ paddingLeft: '1.5em', marginTop: '0.25em' }}>
                            {columns.map((col, colIdx) => (
                              <li key={colIdx} style={{ listStyleType: 'circle' }}>
                                <code>{col}</code>
                              </li>
                            ))}
                          </ul>
                        </li>
                      )
                    )}
                  </ul>
                </div>
              </div>
            </>
          )}
        </>
      )}

      {/* Generated Code Metadata Section */}
      {metadata.generated && (
        <>
          <div className="flowbook-metadata-divider" />
          <div className="flowbook-metadata-header">Generated Code</div>
          <div className="flowbook-metadata-divider" />

          <div className="flowbook-metadata-section">
            <div className="flowbook-metadata-item">
              <strong>Explanation:</strong> {metadata.generated.explanation}
            </div>
          </div>

          <div className="flowbook-metadata-divider" />
          <div className="flowbook-metadata-section">
            <div
              className="flowbook-metadata-profile-toggle"
              onClick={() => setGeneratedCodeExpanded(!generatedCodeExpanded)}
            >
              <strong>Original Specification</strong>
              <span className="flowbook-metadata-toggle-icon">
                {generatedCodeExpanded ? '▼' : '▶'}
              </span>
            </div>
            {generatedCodeExpanded && (
              <pre className="flowbook-metadata-profile-output">
                {metadata.generated.original_spec}
              </pre>
            )}
          </div>
        </>
      )}

      {/* Optimized Code Metadata Section */}
      {metadata.optimized && (
        <>
          <div className="flowbook-metadata-divider" />
          <div className="flowbook-metadata-header">Optimized Code</div>
          <div className="flowbook-metadata-divider" />

          <div className="flowbook-metadata-section">
            <div className="flowbook-metadata-item">
              <strong>Optimizations Applied:</strong>
              <ul style={{ marginTop: '0.5em', paddingLeft: '1.5em' }}>
                {metadata.optimized.optimizations_applied.map((opt, idx) => (
                  <li key={idx}>{opt}</li>
                ))}
              </ul>
            </div>
          </div>

          <div className="flowbook-metadata-divider" />
          <div className="flowbook-metadata-section">
            <div
              className="flowbook-metadata-profile-toggle"
              onClick={() => setOriginalCodeExpanded(!originalCodeExpanded)}
            >
              <strong>Original Code</strong>
              <span className="flowbook-metadata-toggle-icon">
                {originalCodeExpanded ? '▼' : '▶'}
              </span>
            </div>
            {originalCodeExpanded && (
              <pre className="flowbook-metadata-profile-output">
                {metadata.optimized.original_code}
              </pre>
            )}
          </div>

          <div className="flowbook-metadata-divider" />
          <div className="flowbook-metadata-section">
            <div
              className="flowbook-metadata-profile-toggle"
              onClick={() => setOptimizedCodeExpanded(!optimizedCodeExpanded)}
            >
              <strong>Optimized Code</strong>
              <span className="flowbook-metadata-toggle-icon">
                {optimizedCodeExpanded ? '▼' : '▶'}
              </span>
            </div>
            {optimizedCodeExpanded && (
              <pre className="flowbook-metadata-profile-output">
                {metadata.optimized.optimized_code}
              </pre>
            )}
          </div>
        </>
      )}

      {/* Optimization Applied Metadata Section */}
      {metadata.optimization_applied && (
        <>
          <div className="flowbook-metadata-divider" />
          <div className="flowbook-metadata-header">Optimizations Applied</div>
          <div className="flowbook-metadata-divider" />

          <div className="flowbook-metadata-section">
            <div className="flowbook-metadata-item">
              <strong>Modified Cells:</strong>
              <ul style={{ marginTop: '0.5em', paddingLeft: '1.5em' }}>
                {metadata.optimization_applied.modified_cell_ids.map((cellId, idx) => (
                  <li key={idx}>
                    <code>{cellId.substring(0, 8)}...</code>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </>
      )}

      {/* Optimization Potential Metadata Section */}
      {metadata.optimization_potential && (
        <>
          <div className="flowbook-metadata-divider" />

          <div className="flowbook-metadata-header">Optimization Potential</div>
          <div className="flowbook-metadata-divider" />

          <div className="flowbook-metadata-section">
            <div className="flowbook-metadata-item">
              <strong>Potential:</strong> {metadata.optimization_potential.potential}
            </div>
          </div>

          {metadata.optimization_potential.optimization_plan &&
            metadata.optimization_potential.optimization_plan.length > 0 && (
              <>
                <div className="flowbook-metadata-divider" />
                <div className="flowbook-metadata-section">
                  <div className="flowbook-metadata-item">
                    <strong>Optimization Plan:</strong>
                  </div>
                  <div className="flowbook-metadata-optimization-plan">
                    {metadata.optimization_potential.optimization_plan.map((step, index) => (
                      <div key={index} className="flowbook-metadata-optimization-step">
                        <div className="flowbook-metadata-optimization-step-header">
                          <strong>Target: </strong>
                          Cell <code>{step.target_cell_id.substring(0, 8)}...</code>
                          {step.function_name && (
                            <>
                              {' '}/{' '}
                              Function <code>{step.function_name}</code>
                            </>
                          )}
                        </div>
                        <div className="flowbook-metadata-optimization-step-descriptions">
                          <strong>Optimizations:</strong>
                          <ul>
                            {step.description.map((desc, descIndex) => (
                              <li key={descIndex}>{desc}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
        </>
      )}
    </div>
  );
};

/**
 * Panel widget for displaying FlowBook cell metadata
 */
export class FlowbookMetadataPanel extends Widget {
  private _metadata: IFlowbookMetadata | null = null;
  private _cellId: string | null = null;

  constructor() {
    super();
    this.id = 'flowbook-metadata-panel';
    this.addClass('flowbook-metadata-panel');
    this.title.label = 'FlowBook Metadata';
    this.title.closable = true;
    this.title.caption = 'FlowBook cell optimization potential metadata';

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
  public updateMetadata(metadata: IFlowbookMetadata | null, cellId: string | null): void {
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
