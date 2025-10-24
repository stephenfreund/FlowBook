"""
Message broadcasting system for streaming messages from server to client.

Provides a singleton broadcaster that commands can use to send messages
to connected clients via Server-Sent Events (SSE).
"""

import asyncio
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class MessageType(Enum):
    """Types of messages that can be sent to the client."""
    APPEND = "append"  # Append text to current line
    NEWLINE = "newline"  # Start a new line
    END = "end"  # Signal message/command completion
    CLEAR = "clear"  # Clear the panel


@dataclass
class Message:
    """A message to be sent to the client."""
    type: MessageType
    content: str = ""
    metadata: Optional[Dict] = None

    def to_json(self) -> str:
        """Convert message to JSON string."""
        data = {
            "type": self.type.value,
            "content": self.content,
        }
        if self.metadata:
            data["metadata"] = self.metadata
        return json.dumps(data)


class MessageBroadcaster:
    """
    Singleton broadcaster for streaming messages to clients.

    Commands can send messages via send_message(), and connected
    clients receive them through their SSE connections.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._queues: Dict[str, asyncio.Queue] = {}
        self._initialized = True

    def register_client(self, client_id: str) -> asyncio.Queue:
        """
        Register a new client and return their message queue.

        Args:
            client_id: Unique identifier for the client

        Returns:
            asyncio.Queue for the client's messages
        """
        if client_id not in self._queues:
            self._queues[client_id] = asyncio.Queue()
        return self._queues[client_id]

    def unregister_client(self, client_id: str):
        """
        Unregister a client and clean up their queue.

        Args:
            client_id: Unique identifier for the client
        """
        if client_id in self._queues:
            del self._queues[client_id]

    def send_message(self, message: Message, client_id: Optional[str] = None):
        """
        Send a message to one or all clients.

        Args:
            message: The message to send
            client_id: If provided, send only to this client. Otherwise broadcast to all.
        """
        if client_id:
            # Send to specific client
            if client_id in self._queues:
                try:
                    self._queues[client_id].put_nowait(message)
                except asyncio.QueueFull:
                    pass  # Skip if queue is full
        else:
            # Broadcast to all clients
            for queue in self._queues.values():
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    pass  # Skip if queue is full

    def append(self, text: str, client_id: Optional[str] = None):
        """
        Convenience method to append text to current line.

        Args:
            text: Text to append
            client_id: Optional client ID to send to specific client
        """
        self.send_message(Message(MessageType.APPEND, text), client_id)

    def newline(self, client_id: Optional[str] = None):
        """
        Convenience method to start a new line.

        Args:
            client_id: Optional client ID to send to specific client
        """
        self.send_message(Message(MessageType.NEWLINE), client_id)

    def end(self, client_id: Optional[str] = None):
        """
        Convenience method to signal message/command completion.

        Args:
            client_id: Optional client ID to send to specific client
        """
        self.send_message(Message(MessageType.END), client_id)

    def clear(self, client_id: Optional[str] = None):
        """
        Convenience method to clear the panel.

        Args:
            client_id: Optional client ID to send to specific client
        """
        self.send_message(Message(MessageType.CLEAR), client_id)

    def get_client_count(self) -> int:
        """Return the number of connected clients."""
        return len(self._queues)


# Global broadcaster instance
_broadcaster = MessageBroadcaster()


def get_broadcaster() -> MessageBroadcaster:
    """Get the global message broadcaster instance."""
    return _broadcaster
