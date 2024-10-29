import json
import os

import numpy as np

from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder
from cerebras.sdk.client import SdkRuntime

# Matrix dimensions
M = 4
N = 6

# Construct A, x, b
A = np.arange(M*N, dtype=np.float32).reshape(M, N)
x = np.full(shape=N, fill_value=1.0, dtype=np.float32)
b = np.full(shape=M, fill_value=2.0, dtype=np.float32)

# Calculate expected y
y_expected = A@x + b

# Read the artifact_path from the JSON file
with open("artifact_path.json", "r", encoding="utf8") as f:
    data = json.load(f)
    artifact_path = data["artifact_path"]

# Instantiate a runner object using a context manager.
# Set simulator=False if running on CS system within appliance.
with SdkRuntime(artifact_path, simulator=True) as runner:
    # Launch the init_and_compute function on device
    runner.launch('init_and_compute', nonblock=False)

    # Copy y back from device
    y_symbol = runner.get_id('y')
    y_result = np.zeros([1*1*M], dtype=np.float32)
    runner.memcpy_d2h(y_result, y_symbol, 0, 0, 1, 1, M, streaming=False,
    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Ensure that the result matches our expectation
np.testing.assert_allclose(y_result, y_expected, atol=0.01, rtol=0)
print("SUCCESS!")