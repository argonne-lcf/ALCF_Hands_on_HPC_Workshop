import json
from cerebras.sdk.client import SdkCompiler

# Instantiate copmiler
compiler = SdkCompiler()

# Launch compile job
artifact_path = compiler.compile(
    ".",
    "layout.csl",
    "--fabric-dims=8,3 --fabric-offsets=4,1 --memcpy --channels=1 -o out",
    "."
)

# Write the artifact_path to a JSON file
with open("artifact_path.json", "w", encoding="utf8") as f:
    json.dump({"artifact_path": artifact_path,}, f)