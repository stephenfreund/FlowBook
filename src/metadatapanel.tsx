import { Widget } from '@lumino/widgets';
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { IFerretMetadata } from './types';

/**
 * React component for displaying Ferret metadata
 */
interface IMetadataDisplayProps {
  metadata: IFerretMetadata | null;
}

const MetadataDisplay: React.FC<IMetadataDisplayProps> = ({ metadata }) => {
  const [profileExpanded, setProfileExpanded] = React.useState(false);

  console.log('[MetadataDisplay] Rendering with metadata:', metadata);

  if (!metadata || (!metadata.inspect && !metadata.profile)) {
    console.log('[MetadataDisplay] No metadata, showing empty state');
    return (
      <div className="ferret-metadata-empty">
        <p>No Ferret metadata available.</p>
        <p>Select a cell with inspection or profile data to view its metadata.</p>
      </div>
    );
  }

  console.log('[MetadataDisplay] Has metadata, rendering content');

  return (
    <div className="ferret-metadata-content">
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

      {/* Inspection Metadata Section */}
      {metadata.inspect && (
        <>
          {metadata.profile && <div className="ferret-metadata-divider" />}

          <div className="ferret-metadata-header">Inspection Metadata</div>
          <div className="ferret-metadata-divider" />

          <div className="ferret-metadata-section">
            <div className="ferret-metadata-item">
              <strong>Optimizability:</strong> {metadata.inspect.optimizability}
            </div>
            <div className="ferret-metadata-item">
              <strong>Readability:</strong> {metadata.inspect.readability}
            </div>
            <div className="ferret-metadata-item">
              <strong>Complexity:</strong> {metadata.inspect.complexity}
            </div>
          </div>

          {metadata.inspect.improvements &&
            metadata.inspect.improvements.length > 0 && (
              <>
                <div className="ferret-metadata-divider" />
                <div className="ferret-metadata-section">
                  <div className="ferret-metadata-item">
                    <strong>Improvements:</strong>
                  </div>
                  <ul className="ferret-metadata-list">
                    {metadata.inspect.improvements.map((item, index) => (
                      <li key={index}>{item}</li>
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

  constructor() {
    super();
    this.id = 'ferret-metadata-panel';
    this.addClass('ferret-metadata-panel');
    this.title.label = 'Ferret Metadata';
    this.title.closable = true;
    this.title.caption = 'Ferret cell inspection metadata';

    this.render();
  }

  /**
   * Render the React component into the panel
   */
  private render(): void {
    ReactDOM.render(<MetadataDisplay metadata={this._metadata} />, this.node);
  }

  /**
   * Update the displayed metadata
   */
  public updateMetadata(metadata: IFerretMetadata | null): void {
    console.log('[MetadataPanel] updateMetadata called with:', metadata);
    this._metadata = metadata;
    this.render();
  }

  /**
   * Clear the displayed metadata
   */
  public clear(): void {
    console.log('[MetadataPanel] clear called');
    this._metadata = null;
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
