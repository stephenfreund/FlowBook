#!/usr/bin/env python3
"""
Simple test to verify comm messages work.
"""

import time
from jupyter_client import KernelManager

# Start a ferret_kernel
km = KernelManager(kernel_name='ferret_kernel')
km.start_kernel()
kc = km.client()
kc.start_channels()
kc.wait_for_ready(timeout=30)

print("Kernel started and ready")

# Send a comm_open message
import uuid
comm_id = uuid.uuid4().hex

msg = kc.session.msg(
    "comm_open",
    content={
        "comm_id": comm_id,
        "target_name": "test_code",
        "data": {
            "original_code": "x = 1",
            "modified_code": "x = 1",
            "output_variables": ["x"]
        }
    }
)

print(f"Sending comm_open with id {comm_id}")
kc.shell_channel.send(msg)

# Wait for messages
print("Waiting for messages...")
timeout = 60
start_time = time.time()

while time.time() - start_time < timeout:
    try:
        reply = kc.iopub_channel.get_msg(timeout=1)
        msg_type = reply['header']['msg_type']
        print(f"Received message: {msg_type}")

        if msg_type == 'comm_msg':
            content = reply['content']
            if content.get('comm_id') == comm_id:
                data = content['data']
                print(f"  Data: {data}")

                if data.get('type') == 'final':
                    print("Received final message!")
                    break
    except Exception as e:
        pass

print("\nStopping kernel...")
kc.stop_channels()
km.shutdown_kernel()
print("Done")
