"""Load SLEAP .slp files under either SLEAP 1.5+ (sleap_io) or the SLEAP 1.4 conda env."""

import os
import sys

# TensorFlow in SLEAP 1.4 ships protobuf code that fails under protobuf >= 3.21.
# The pure-Python parser avoids that without pinning protobuf. It must be set
# before sleap (and so tensorflow) is imported.
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

# sleap_io comes with SLEAP 1.5+ (`uv tool install "sleap[nn]"`). The 1.4.x
# conda env may not have it, and sleap.load_file was removed in 1.5.
try:
    import sleap_io as sio

    load_slp = sio.load_file
except ImportError:
    try:
        import sleap

        load_slp = sleap.load_file
    except (ImportError, AttributeError):
        sys.stderr.write(
            'ERROR: neither sleap_io nor sleap.load_file is available. '
            'Activate the sleap env first.\n'
        )
        sys.exit(1)
