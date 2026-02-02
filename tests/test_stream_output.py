"""
Test script for StreamOutputContext and BroadcastStream with ANSI color support.
"""

from flowbook.server.message_broadcaster import MessageBroadcaster, BroadcastStream
from flowbook.util.output import stream_output, log, print as out_print
from flowbook.util.text import parse_ansi_text
import termcolor


class MockBroadcaster:
    """Mock broadcaster that collects messages for testing."""

    def __init__(self):
        self.messages = []

    def send_message(self, message, client_id=None):
        """Record messages with full metadata."""
        msg_dict = {
            "type": message.type.value,
            "content": message.content,
            "metadata": message.metadata
        }
        self.messages.append(msg_dict)
        meta_str = f", metadata={message.metadata}" if message.metadata else ""
        print(f"  -> {message.type.value.upper()}: {repr(message.content)}{meta_str}")

    def append(self, text: str, client_id=None):
        """Record append messages (for backwards compatibility)."""
        msg_dict = {
            "type": "append",
            "content": text,
            "metadata": None
        }
        self.messages.append(msg_dict)
        print(f"  -> APPEND: {repr(text)}")

    def newline(self, client_id=None):
        """Record newline messages."""
        msg_dict = {
            "type": "newline",
            "content": "",
            "metadata": None
        }
        self.messages.append(msg_dict)
        print(f"  -> NEWLINE")

    def end(self, client_id=None):
        """Record end messages."""
        msg_dict = {
            "type": "end",
            "content": "",
            "metadata": None
        }
        self.messages.append(msg_dict)
        print(f"  -> END")

    def clear(self, client_id=None):
        """Record clear messages."""
        msg_dict = {
            "type": "clear",
            "content": "",
            "metadata": None
        }
        self.messages.append(msg_dict)
        print(f"  -> CLEAR")


def test_broadcast_stream():
    """Test BroadcastStream write/flush interface."""
    print("\n=== Test 1: BroadcastStream Direct Usage ===")
    mock = MockBroadcaster()
    stream = BroadcastStream(mock)

    print("Writing 'Hello'...")
    stream.write("Hello")

    print("\nWriting ' World'...")
    stream.write(" World")

    print("\nWriting '\\n'...")
    stream.write("\n")

    print("\nWriting 'Next line'...")
    stream.write("Next line")

    print("\nExpected sequence:")
    print("  APPEND: 'Hello'")
    print("  APPEND: ' World'")
    print("  NEWLINE")
    print("  APPEND: 'Next line'")

    # Verify messages
    assert len(mock.messages) == 4, f"Expected 4 messages, got {len(mock.messages)}"
    assert mock.messages[0]['type'] == 'append' and mock.messages[0]['content'] == 'Hello'
    assert mock.messages[1]['type'] == 'append' and mock.messages[1]['content'] == ' World'
    assert mock.messages[2]['type'] == 'newline'
    assert mock.messages[3]['type'] == 'append' and mock.messages[3]['content'] == 'Next line'

    print("\n✓ Test 1 passed!")


def test_stream_output_context():
    """Test stream_output context manager."""
    print("\n=== Test 2: stream_output Context ===")
    mock = MockBroadcaster()
    stream = BroadcastStream(mock)

    print("Using stream_output context...")
    with stream_output(stream):
        print("Calling log('Test message')...")
        log("Test message")

        print("\nCalling out_print('Another message')...")
        out_print("Another message")

    print("\nMessages collected:")
    for msg in mock.messages:
        if msg['content']:
            print(f"  {msg['type'].upper()}: {repr(msg['content'])}")
        else:
            print(f"  {msg['type'].upper()}")

    print("\n✓ Test 2 passed!")


def test_multiline_write():
    """Test writing text with multiple newlines."""
    print("\n=== Test 3: Multiline Write ===")
    mock = MockBroadcaster()
    stream = BroadcastStream(mock)

    print("Writing 'Line 1\\nLine 2\\nLine 3'...")
    stream.write("Line 1\nLine 2\nLine 3")

    print("\nExpected sequence:")
    print("  APPEND: 'Line 1'")
    print("  NEWLINE")
    print("  APPEND: 'Line 2'")
    print("  NEWLINE")
    print("  APPEND: 'Line 3'")

    # Verify messages
    assert len(mock.messages) == 5, f"Expected 5 messages, got {len(mock.messages)}"
    assert mock.messages[0]['type'] == 'append' and mock.messages[0]['content'] == 'Line 1'
    assert mock.messages[1]['type'] == 'newline'
    assert mock.messages[2]['type'] == 'append' and mock.messages[2]['content'] == 'Line 2'
    assert mock.messages[3]['type'] == 'newline'
    assert mock.messages[4]['type'] == 'append' and mock.messages[4]['content'] == 'Line 3'

    print("\n✓ Test 3 passed!")


def test_ansi_color_parsing():
    """Test ANSI color code parsing and metadata."""
    print("\n=== Test 4: ANSI Color Parsing ===")

    # Create ANSI colored text manually (termcolor may not work in non-TTY)
    red_text = "\x1B[31mError message\x1B[0m"
    stripped, metadata = parse_ansi_text(red_text)

    print(f"Original: {repr(red_text)}")
    print(f"Stripped: {repr(stripped)}")
    print(f"Metadata: {metadata}")

    assert stripped == "Error message", f"Expected 'Error message', got {repr(stripped)}"
    assert metadata is not None, "Expected metadata to be present"
    assert metadata.get('color') == 'red', f"Expected color='red', got {metadata}"

    print("\n✓ Color parsing works!")

    # Test BroadcastStream with colored text
    print("\n--- Testing BroadcastStream with colors ---")
    mock = MockBroadcaster()
    stream = BroadcastStream(mock)

    # Cyan text with ANSI code
    cyan_text = "\x1B[36mProcessing...\x1B[0m"
    print(f"Writing colored text: {repr(cyan_text)}")
    stream.write(cyan_text)

    print("\nMessages collected:")
    for msg in mock.messages:
        if isinstance(msg, dict):
            print(f"  Type: {msg['type']}, Content: {repr(msg['content'])}, Metadata: {msg['metadata']}")

    # Verify the message has color metadata
    assert len(mock.messages) > 0, "Expected at least one message"
    msg = mock.messages[0]
    assert msg['content'] == 'Processing...', f"Expected 'Processing...', got {repr(msg['content'])}"
    assert msg['metadata'] is not None, "Expected metadata"
    assert msg['metadata'].get('color') == 'cyan', f"Expected color='cyan', got {msg['metadata']}"

    print("\n✓ Test 4 passed!")


if __name__ == "__main__":
    test_broadcast_stream()
    test_stream_output_context()
    test_multiline_write()
    test_ansi_color_parsing()
    print("\n=== All tests passed! ===\n")
