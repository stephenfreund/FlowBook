"""
Real kernel message capture utility.

This test runs against an actual Jupyter kernel to capture and log
the exact message format when errors occur, helping diagnose the
parent_header mismatch issue.

Run with: python test_real_kernel_message_capture.py

This is NOT a pytest test - it's a diagnostic utility.
"""

import sys
import json
import time
from pathlib import Path


def capture_messages_with_logging(kc, code, description):
    """
    Execute code and capture all IOPub messages with detailed logging.

    Args:
        kc: Kernel client
        code: Code to execute
        description: Description for logging

    Returns:
        List of captured messages
    """
    print(f"\n{'=' * 70}")
    print(f"Test: {description}")
    print(f"{'=' * 70}")
    print(f"Code:\n{code}")
    print(f"{'=' * 70}")

    # Execute code
    msg_id = kc.execute(code, store_history=False)
    print(f"Execute msg_id: {msg_id}")
    print(f"{'=' * 70}")

    captured_messages = []
    message_num = 0

    # Collect messages for up to 5 seconds
    start_time = time.time()
    timeout = 5.0

    while time.time() - start_time < timeout:
        try:
            msg = kc.get_iopub_msg(timeout=1.0)

            # Only process messages for our execution
            parent_msg_id = msg.get('parent_header', {}).get('msg_id')
            msg_type = msg['header']['msg_type']

            message_num += 1

            # Log message details
            print(f"\n--- Message {message_num} ---")
            print(f"msg_type: {msg_type}")
            print(f"msg_id: {msg['header']['msg_id']}")
            print(f"parent_header.msg_id: {parent_msg_id}")
            print(f"parent_id matches: {parent_msg_id == msg_id}")

            # Log content for important message types
            if msg_type == 'error':
                print(f"ERROR CONTENT:")
                print(f"  ename: {msg['content']['ename']}")
                print(f"  evalue: {msg['content']['evalue']}")
                print(f"  traceback length: {len(msg['content'].get('traceback', []))} lines")
                print(f"  First traceback line: {msg['content'].get('traceback', [''])[0][:100]}")

            elif msg_type == 'stream':
                text = msg['content'].get('text', '')
                print(f"STREAM: {text[:100]}...")

            elif msg_type == 'status':
                state = msg['content'].get('execution_state')
                print(f"STATUS: {state}")
                if state == 'idle' and parent_msg_id == msg_id:
                    print("Execution complete!")
                    captured_messages.append(msg)
                    break

            captured_messages.append(msg)

        except Exception as e:
            if "Timeout waiting for output" in str(e):
                break
            else:
                print(f"Exception: {e}")
                break

    print(f"\n{'=' * 70}")
    print(f"Total messages captured: {len(captured_messages)}")
    print(f"{'=' * 70}\n")

    return captured_messages, msg_id


def analyze_parent_header_consistency(messages, expected_msg_id):
    """
    Analyze parent_header consistency across messages.

    Args:
        messages: List of captured messages
        expected_msg_id: The msg_id from execute request

    Returns:
        Analysis results
    """
    print(f"\n{'=' * 70}")
    print(f"PARENT_HEADER ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Expected msg_id: {expected_msg_id}")
    print(f"{'=' * 70}")

    matching = 0
    mismatched = 0
    missing = 0
    error_messages = []

    for i, msg in enumerate(messages):
        msg_type = msg['header']['msg_type']
        parent_msg_id = msg.get('parent_header', {}).get('msg_id')

        status = "✓" if parent_msg_id == expected_msg_id else "✗"

        if parent_msg_id == expected_msg_id:
            matching += 1
        elif parent_msg_id is None:
            missing += 1
            status = "?"
        else:
            mismatched += 1

        if msg_type == 'error':
            error_messages.append((i, msg, status))

        print(f"{status} Msg {i+1:2d}: {msg_type:15s} parent_id={parent_msg_id}")

    print(f"\n{'=' * 70}")
    print(f"Summary:")
    print(f"  Matching parent_header:   {matching}")
    print(f"  Mismatched parent_header: {mismatched}")
    print(f"  Missing parent_header:    {missing}")
    print(f"  Total error messages:     {len(error_messages)}")

    if error_messages:
        print(f"\n{'=' * 70}")
        print(f"ERROR MESSAGE ANALYSIS:")
        print(f"{'=' * 70}")
        for idx, msg, status in error_messages:
            parent_id = msg.get('parent_header', {}).get('msg_id')
            print(f"\n{status} Error message {idx+1}:")
            print(f"  parent_header.msg_id: {parent_id}")
            print(f"  Matches expected: {parent_id == expected_msg_id}")
            print(f"  Error type: {msg['content']['ename']}")
            print(f"  Error message: {msg['content']['evalue']}")

            if parent_id != expected_msg_id:
                print(f"  ⚠️  POTENTIAL BUG: Error message has mismatched parent_header!")

    print(f"{'=' * 70}\n")

    return {
        'matching': matching,
        'mismatched': mismatched,
        'missing': missing,
        'error_messages': len(error_messages),
        'error_parent_mismatch': any(
            msg.get('parent_header', {}).get('msg_id') != expected_msg_id
            for _, msg, _ in error_messages
        )
    }


def save_messages_to_file(messages, filename):
    """Save captured messages to JSON file for inspection."""
    # Convert messages to serializable format
    serializable = []
    for msg in messages:
        serializable.append({
            'header': msg.get('header', {}),
            'parent_header': msg.get('parent_header', {}),
            'content': msg.get('content', {}),
            'metadata': msg.get('metadata', {})
        })

    output_path = Path(filename)
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"Messages saved to: {output_path.absolute()}")


def main():
    """Main diagnostic function."""
    try:
        from jupyter_client import BlockingKernelClient
        from jupyter_client.manager import start_new_kernel
    except ImportError:
        print("ERROR: jupyter_client not installed")
        print("Install with: pip install jupyter_client")
        sys.exit(1)

    print("Starting Jupyter kernel...")
    km, kc = start_new_kernel(kernel_name='python3')

    try:
        # Wait for kernel to be ready
        kc.wait_for_ready(timeout=10)
        print("Kernel ready!\n")

        # Test 1: Simple error
        messages1, msg_id1 = capture_messages_with_logging(
            kc,
            "x = undefined_variable",
            "Simple NameError"
        )
        analysis1 = analyze_parent_header_consistency(messages1, msg_id1)
        save_messages_to_file(messages1, "messages_simple_error.json")

        # Test 2: Partial execution before error (the bug scenario)
        messages2, msg_id2 = capture_messages_with_logging(
            kc,
            """# First part executes successfully
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
df['b'] = df['a'] * 2

# Then error occurs
df['c'] = undefined_model.predict(df)
""",
            "Partial execution then NameError (bug scenario)"
        )
        analysis2 = analyze_parent_header_consistency(messages2, msg_id2)
        save_messages_to_file(messages2, "messages_partial_execution_error.json")

        # Test 3: Error with output
        messages3, msg_id3 = capture_messages_with_logging(
            kc,
            """print("Starting computation...")
import time
time.sleep(0.1)
print("About to error...")
raise ValueError("Test error")
""",
            "Error with preceding output"
        )
        analysis3 = analyze_parent_header_consistency(messages3, msg_id3)
        save_messages_to_file(messages3, "messages_error_with_output.json")

        # Final summary
        print(f"\n{'=' * 70}")
        print(f"FINAL SUMMARY")
        print(f"{'=' * 70}")

        print(f"\nTest 1 - Simple NameError:")
        print(f"  Error messages: {analysis1['error_messages']}")
        print(f"  Parent mismatch: {analysis1['error_parent_mismatch']}")

        print(f"\nTest 2 - Partial execution (bug scenario):")
        print(f"  Error messages: {analysis2['error_messages']}")
        print(f"  Parent mismatch: {analysis2['error_parent_mismatch']}")

        print(f"\nTest 3 - Error with output:")
        print(f"  Error messages: {analysis3['error_messages']}")
        print(f"  Parent mismatch: {analysis3['error_parent_mismatch']}")

        if any([analysis1['error_parent_mismatch'],
                analysis2['error_parent_mismatch'],
                analysis3['error_parent_mismatch']]):
            print(f"\n⚠️  WARNING: Found error messages with mismatched parent_headers!")
            print(f"    This confirms the bug in _wait_for_execution.")
        else:
            print(f"\n✓ All error messages have matching parent_headers.")
            print(f"  The bug may be in a different part of the message handling.")

        print(f"\n{'=' * 70}\n")

    finally:
        print("Shutting down kernel...")
        kc.stop_channels()
        km.shutdown_kernel()
        print("Done!")


if __name__ == "__main__":
    main()
