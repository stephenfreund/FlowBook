import * as React from 'react';

/**
 * Shared message handling components and utilities for FlowBook
 * Used by both MessagePanel and CommandExecutionDialog
 */

/**
 * Message types from the server broadcaster
 */
export enum MessageType {
  APPEND = 'append',
  NEWLINE = 'newline',
  END = 'end',
  CLEAR = 'clear'
}

/**
 * Message structure from the server
 */
export interface IMessage {
  type: MessageType;
  content: string;
  metadata?: {
    color?: string;
    bold?: boolean;
  };
}

/**
 * Message segment with optional styling
 */
export interface IMessageSegment {
  content: string;
  color?: string;
  bold?: boolean;
}

/**
 * React component props for displaying messages
 */
export interface IMessageDisplayProps {
  segments: IMessageSegment[];
}

/**
 * React component for displaying message segments with ANSI color styling
 */
export const MessageDisplay: React.FC<IMessageDisplayProps> = ({
  segments
}) => {
  const contentRef = React.useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  React.useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [segments]);

  return (
    <div className="flowbook-message-display" ref={contentRef}>
      <pre className="flowbook-message-content">
        {segments.map((segment, index) => {
          const style: React.CSSProperties = {};

          if (segment.color) {
            style.color = `var(--flowbook-color-${segment.color}, inherit)`;
          }
          if (segment.bold) {
            style.fontWeight = 'bold';
          }

          return (
            <span key={index} style={style}>
              {segment.content}
            </span>
          );
        })}
      </pre>
    </div>
  );
};
