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
 * Options for creating a CommandExecutionDialog
 */
export interface ICommandExecutionDialogOptions {
  commandLabel: string;
  onClose?: () => void;
}

/**
 * State of the dialog
 */
interface IDialogState {
  isRunning: boolean;
  isComplete: boolean;
  isSuccess: boolean;
  errorMessage?: string;
}

/**
 * Props for the dialog React component
 */
interface IDialogContentProps {
  commandLabel: string;
  segments: IMessageSegment[];
  state: IDialogState;
  onOkClick: () => void;
}

/**
 * React component for the dialog content
 */
const DialogContent: React.FC<IDialogContentProps> = ({
  commandLabel,
  segments,
  state,
  onOkClick
}) => {
  return (
    <div className='ferret-execution-dialog-content'>
      <div className='ferret-execution-dialog-header'>
        <span className='ferret-execution-dialog-title'>
          Executing: {commandLabel}
        </span>
      </div>
      <div className='ferret-execution-dialog-body'>
        <MessageDisplay segments={segments} />
      </div>
      <div className='ferret-execution-dialog-footer'>
        {state.isComplete && (
          <button
            className='jp-Dialog-button jp-mod-accept'
            onClick={onOkClick}
          >
            OK
          </button>
        )}
      </div>
    </div>
  );
};

/**
 * Custom modal dialog widget that displays real-time command execution messages
 */
export class CommandExecutionDialog extends Widget {
  private _eventSource: EventSource | null = null;
  private _segments: IMessageSegment[] = [];
  private _currentSegment: IMessageSegment | null = null;
  private _state: IDialogState;
  private _commandLabel: string;
  private _onClose?: () => void;
  private _overlay: HTMLDivElement | null = null;
  private _autoCloseTimer: number | null = null;
  private _connectionResolve?: () => void;

  constructor(options: ICommandExecutionDialogOptions) {
    super();
    this._commandLabel = options.commandLabel;
    this._onClose = options.onClose;
    this._state = {
      isRunning: true,
      isComplete: false,
      isSuccess: false
    };

    this.addClass('ferret-execution-dialog');
    this.render();
  }

  /**
   * Show the dialog as a modal and wait for SSE connection
   * Returns a promise that resolves when the SSE connection is established
   */
  public show(): Promise<void> {
    // Create modal overlay
    this._overlay = document.createElement('div');
    this._overlay.className = 'ferret-execution-dialog-overlay';

    // Add overlay to body first
    document.body.appendChild(this._overlay);

    // Attach the widget to the overlay
    Widget.attach(this, this._overlay);

    // Clear any existing messages and connect to stream
    this._segments = [];
    this._currentSegment = null;

    // Return a promise that resolves when SSE connection is open
    return new Promise<void>((resolve) => {
      this._connectionResolve = resolve;
      this.connectToStream();
    });
  }

  /**
   * Close the dialog and cleanup
   */
  public close(): void {
    // Clear auto-close timer
    if (this._autoCloseTimer !== null) {
      window.clearTimeout(this._autoCloseTimer);
      this._autoCloseTimer = null;
    }

    // Disconnect from stream
    this.disconnectFromStream();

    // Detach widget from parent
    if (this.parent) {
      Widget.detach(this);
    }

    // Remove overlay from DOM
    if (this._overlay && this._overlay.parentNode) {
      this._overlay.parentNode.removeChild(this._overlay);
    }

    // Dispose the widget
    this.dispose();

    // Notify close callback
    if (this._onClose) {
      this._onClose();
    }
  }

  /**
   * Render the React component into the widget
   */
  private render(): void {
    const displaySegments = [...this._segments];
    if (this._currentSegment) {
      displaySegments.push(this._currentSegment);
    }

    ReactDOM.render(
      <DialogContent
        commandLabel={this._commandLabel}
        segments={displaySegments}
        state={this._state}
        onOkClick={() => this.close()}
      />,
      this.node
    );
  }

  /**
   * Connect to the server's SSE stream
   */
  private connectToStream(): void {
    try {
      const settings = ServerConnection.makeSettings();
      const streamUrl = `${settings.baseUrl}ferret/stream`;

      this._eventSource = new EventSource(streamUrl);

      this._eventSource.addEventListener('open', () => {
        console.log('[CommandExecutionDialog] Connected to message stream');
        // Resolve the promise to signal connection is ready
        if (this._connectionResolve) {
          this._connectionResolve();
          this._connectionResolve = undefined;
        }
      });

      this._eventSource.addEventListener('message', (event: MessageEvent) => {
        try {
          const message: IMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('[CommandExecutionDialog] Error parsing message:', error);
        }
      });

      this._eventSource.addEventListener('error', (error: Event) => {
        console.error('[CommandExecutionDialog] EventSource error:', error);
        // Don't reconnect - dialog is transient
      });
    } catch (error) {
      console.error('[CommandExecutionDialog] Error connecting to stream:', error);
    }
  }

  /**
   * Disconnect from the SSE stream
   */
  private disconnectFromStream(): void {
    if (this._eventSource) {
      this._eventSource.close();
      this._eventSource = null;
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
        this.render();
        break;

      case MessageType.NEWLINE:
        // Finalize current segment and add newline
        if (this._currentSegment) {
          this._segments.push(this._currentSegment);
          this._currentSegment = null;
        }
        this._segments.push({ content: '\n' });
        this.render();
        break;

      case MessageType.END:
        // Finalize current segment
        if (this._currentSegment) {
          this._segments.push(this._currentSegment);
          this._currentSegment = null;
        }
        // Add completion marker
        this._segments.push({ content: '\n--- Complete ---\n' });

        // Mark as complete
        this._state.isComplete = true;
        this._state.isRunning = false;
        this._state.isSuccess = true;

        this.render();

        // Auto-close after 500ms on success (COMMENTED OUT FOR DEBUGGING)
        // this._autoCloseTimer = window.setTimeout(() => {
        //   this.close();
        // }, 500);
        break;

      case MessageType.CLEAR:
        this._segments = [];
        this._currentSegment = null;
        this.render();
        break;

      default:
        console.warn('[CommandExecutionDialog] Unknown message type:', message.type);
    }
  }

  /**
   * Mark the command as having an error
   */
  public setError(errorMessage: string): void {
    // Finalize current segment
    if (this._currentSegment) {
      this._segments.push(this._currentSegment);
      this._currentSegment = null;
    }

    // Add error message
    this._segments.push({
      content: `\nError: ${errorMessage}\n`,
      color: 'red',
      bold: true
    });

    // Update state
    this._state.isComplete = true;
    this._state.isRunning = false;
    this._state.isSuccess = false;
    this._state.errorMessage = errorMessage;

    this.render();
  }

  /**
   * Dispose of the dialog and clean up resources
   */
  dispose(): void {
    if (this.isDisposed) {
      return;
    }

    if (this._autoCloseTimer !== null) {
      window.clearTimeout(this._autoCloseTimer);
      this._autoCloseTimer = null;
    }

    this.disconnectFromStream();
    ReactDOM.unmountComponentAtNode(this.node);
    super.dispose();
  }
}
