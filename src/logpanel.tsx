import { Widget } from '@lumino/widgets';
import { ServerConnection } from '@jupyterlab/services';
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import {
  MessageType,
  IMessage,
  IMessageSegment,
  MessageDisplay
} from './messagecomponents';

/**
 * Panel widget for displaying messages from the server's message broadcaster
 */
export class MessagePanel extends Widget {
  private _eventSource: EventSource | null = null;
  private _segments: IMessageSegment[] = [];
  private _currentSegment: IMessageSegment | null = null;
  private _isConnected: boolean = false;
  private _reconnectTimer: number | null = null;

  constructor() {
    super();
    this.id = 'flowbook-message-panel';
    this.addClass('flowbook-message-panel');
    this.title.label = 'FlowBook Output';
    this.title.closable = true;
    this.title.caption = 'FlowBook command output';

    this.render();
    this.connectToStream();
  }

  /**
   * Render the React component into the panel
   */
  private render(): void {
    ReactDOM.render(<MessageDisplay segments={this._segments} />, this.node);
  }

  /**
   * Connect to the server's SSE stream
   */
  private connectToStream(): void {
    try {
      const settings = ServerConnection.makeSettings();
      const streamUrl = `${settings.baseUrl}flowbook/stream`;

      this._eventSource = new EventSource(streamUrl);

      this._eventSource.addEventListener('open', () => {
        console.log('[MessagePanel] Connected to message stream');
        this._isConnected = true;
        this.clearReconnectTimer();
      });

      this._eventSource.addEventListener('message', (event: MessageEvent) => {
        try {
          const message: IMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('[MessagePanel] Error parsing message:', error);
        }
      });

      this._eventSource.addEventListener('error', (error: Event) => {
        console.error('[MessagePanel] EventSource error:', error);
        this._isConnected = false;
        this.scheduleReconnect();
      });
    } catch (error) {
      console.error('[MessagePanel] Error connecting to stream:', error);
      this.scheduleReconnect();
    }
  }

  /**
   * Handle incoming messages from the server
   */
  private handleMessage(message: IMessage): void {
    switch (message.type) {
      case MessageType.APPEND:
        // Create a new segment with the content and metadata
        const segment: IMessageSegment = {
          content: message.content,
          color: message.metadata?.color,
          bold: message.metadata?.bold
        };

        // If there's a current segment being built, append to it
        if (
          this._currentSegment &&
          this._currentSegment.color === segment.color &&
          this._currentSegment.bold === segment.bold
        ) {
          // Same styling, append to current segment
          this._currentSegment.content += segment.content;
        } else {
          // Different styling or first segment, create new
          if (this._currentSegment) {
            this._segments.push(this._currentSegment);
          }
          this._currentSegment = segment;
        }
        this.updateDisplay();
        break;

      case MessageType.NEWLINE:
        // Finalize current segment and add newline
        if (this._currentSegment) {
          this._segments.push(this._currentSegment);
          this._currentSegment = null;
        }
        this._segments.push({ content: '\n' });
        this.updateDisplay();
        break;

      case MessageType.END:
        // Finalize current segment
        if (this._currentSegment) {
          this._segments.push(this._currentSegment);
          this._currentSegment = null;
        }
        // Add completion marker
        this._segments.push({ content: '\n--- Complete ---\n' });
        this.updateDisplay();
        break;

      case MessageType.CLEAR:
        this._segments = [];
        this._currentSegment = null;
        this.updateDisplay();
        break;

      default:
        console.warn('[MessagePanel] Unknown message type:', message.type);
    }
  }

  /**
   * Update the display with current segments
   */
  private updateDisplay(): void {
    const displaySegments = [...this._segments];
    if (this._currentSegment) {
      displaySegments.push(this._currentSegment);
    }
    ReactDOM.render(<MessageDisplay segments={displaySegments} />, this.node);
  }

  /**
   * Schedule a reconnection attempt
   */
  private scheduleReconnect(): void {
    if (this._reconnectTimer === null) {
      console.log('[MessagePanel] Scheduling reconnect in 5 seconds...');
      this._reconnectTimer = window.setTimeout(() => {
        console.log('[MessagePanel] Attempting to reconnect...');
        this.closeStream();
        this.connectToStream();
      }, 5000);
    }
  }

  /**
   * Clear the reconnect timer
   */
  private clearReconnectTimer(): void {
    if (this._reconnectTimer !== null) {
      window.clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
  }

  /**
   * Close the SSE connection
   */
  private closeStream(): void {
    if (this._eventSource) {
      this._eventSource.close();
      this._eventSource = null;
    }
    this._isConnected = false;
  }

  /**
   * Dispose of the panel and clean up resources
   */
  dispose(): void {
    if (this.isDisposed) {
      return;
    }

    this.clearReconnectTimer();
    this.closeStream();
    ReactDOM.unmountComponentAtNode(this.node);
    super.dispose();
  }

  /**
   * Handle when the panel is shown
   */
  protected onActivateRequest(msg: any): void {
    super.onActivateRequest(msg);
    if (!this._isConnected && !this._eventSource) {
      this.connectToStream();
    }
  }

  /**
   * Public method to manually clear the panel
   */
  public clear(): void {
    this._segments = [];
    this._currentSegment = null;
    this.updateDisplay();
  }

  /**
   * Get connection status
   */
  public get isConnected(): boolean {
    return this._isConnected;
  }
}
