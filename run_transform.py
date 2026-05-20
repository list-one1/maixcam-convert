import subprocess
import sys
from tpu_mlir import tools_path

# Build the command
cmd = [
    sys.executable,
    tools_path + "/model_transform.py",
    "--model_name", "yolov8m_card",
    "--model_def", "best.onnx",
    "--input_shapes", "[[1,3,640,640]]",
    "--mean", "0,0,0",
    "--scale", "0.00392156862745098,0.00392156862745098,0.00392156862745098",
    "--pixel_format", "rgb",
    "--channel_format", "nchw",
    "--output_names", "/model.22/dfl/conv/Conv_output_0,/model.22/Sigmoid_output_0",
    "--tolerance", "0.99,0.99",
    "--mlir", "workspace/yolov8m_card.mlir",
]

print("Running:", " ".join(cmd))
result = subprocess.run(cmd, cwd="C:/Users/1/Desktop/weights")
sys.exit(result.returncode)
