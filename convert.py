#!/usr/bin/env python3
"""
YOLO .pt → MaixCam (.cvimodel + .mud) 一条龙转换工具
=====================================================
用法: python convert.py [model.pt] [选项]

示例:
  python convert.py best.pt                          # 默认转换
  python convert.py best.pt --quantize INT8          # INT8 量化
  python convert.py best.pt --quantize BF16          # BF16 量化(无需校准图)
  python convert.py best.pt --no-docker              # 只做 pt→onnx+mud, 不跑Docker
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check_python():
    """Check Python availability for ONNX export."""
    try:
        import torch  # noqa: F401
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False


def check_docker():
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "json"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_model_info(pt_path):
    """Extract model metadata from .pt file."""
    from ultralytics import YOLO
    model = YOLO(str(pt_path))

    # Get input size from model
    if hasattr(model.model, 'args'):
        imgsz = model.model.args.get('imgsz', 640)
    else:
        imgsz = 640

    if isinstance(imgsz, (list, tuple)):
        imgsz = imgsz[0] if len(imgsz) > 0 else 640

    return {
        "task": model.task,
        "names": model.names,
        "num_classes": len(model.names),
        "input_size": int(imgsz),
        "model_name": Path(pt_path).stem,
    }


def export_onnx(pt_path, onnx_path, imgsz=640, opset=12):
    """Export YOLO .pt to ONNX."""
    print(f"[1/4] 导出 ONNX: {pt_path} → {onnx_path}")
    from ultralytics import YOLO
    model = YOLO(str(pt_path))
    model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True)
    # ultralytics exports to pt_path.stem + '.onnx' by default
    default_onnx = Path(pt_path).with_suffix(".onnx")
    if default_onnx != Path(onnx_path) and default_onnx.exists():
        default_onnx.rename(onnx_path)
    print(f"  ✓ ONNX 导出完成: {onnx_path}")
    return onnx_path


def generate_mud(model_info, mud_path, quantize="INT8"):
    """Generate .mud config file for MaixCam."""
    print(f"[2/4] 生成 .mud 配置文件: {mud_path}")

    scale = "0.00392156862745098,0.00392156862745098,0.00392156862745098"
    labels = ",".join(model_info["names"].values())

    cvi_model = f"{model_info['model_name']}_{quantize.lower()}.cvimodel"

    content = f"""[basic]
type = cvimodel
model = {cvi_model}

[extra]
model_type = yolov8
input_type = rgb
mean = 0, 0, 0
scale = {scale}
labels = {labels}
"""
    Path(mud_path).write_text(content, encoding="utf-8")
    print(f"  ✓ .mud 文件已生成: {mud_path}")
    print(f"  ✓ 模型类型: {model_info['task']}")
    print(f"  ✓ 类别数: {model_info['num_classes']}")
    print(f"  ✓ 输入尺寸: {model_info['input_size']}x{model_info['input_size']}")
    return mud_path


def get_output_names():
    """Get YOLOv8 output node names (same for all YOLOv8 models)."""
    return "/model.22/dfl/conv/Conv_output_0,/model.22/Sigmoid_output_0"


def run_docker_convert(onnx_path, model_name, input_size, quantize="BF16",
                       calibration_dir=None, processor="cv181x"):
    """Run TPU MLIR conversion inside Docker."""
    print(f"[3/4] Docker 容器内 TPU MLIR 转换 ({quantize}量化)...")

    onnx_abs = Path(onnx_path).resolve()
    workspace_abs = ROOT / "workspace"
    workspace_abs.mkdir(exist_ok=True)

    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{onnx_abs.parent}:/workspace",
        "sophgo/tpuc_dev:latest",
        "bash", "-c", f"""
set -e
cd /workspace
mkdir -p workspace

echo '=== Step 1: ONNX → MLIR ==='
model_transform.py \\
  --model_name {model_name} \\
  --model_def {onnx_abs.name} \\
  --input_shapes "[[1,3,{input_size},{input_size}]]" \\
  --mean "0,0,0" \\
  --scale "0.00392156862745098,0.00392156862745098,0.00392156862745098" \\
  --pixel_format rgb \\
  --channel_format nchw \\
  --output_names "{get_output_names()}" \\
  --tolerance 0.99,0.99 \\
  --mlir workspace/{model_name}.mlir
"""
    ]

    if quantize.upper() == "INT8" and calibration_dir:
        # INT8 calibration path
        cal_dir = Path(calibration_dir).resolve() if calibration_dir else None
        if cal_dir and cal_dir.exists():
            docker_cmd[4] += f" -v {cal_dir}:/calibration_images"
            docker_cmd[-1] += f"""
echo '=== Step 2: Calibration (INT8) ==='
IMG_COUNT=$(ls /calibration_images/*.jpg /calibration_images/*.jpeg /calibration_images/*.png 2>/dev/null | wc -l)
run_calibration.py workspace/{model_name}.mlir \\
  --dataset /calibration_images \\
  --input_num ${{IMG_COUNT}} \\
  -o workspace/{model_name}_cali_table

echo '=== Step 3: MLIR → cvimodel (INT8) ==='
model_deploy.py \\
  --mlir workspace/{model_name}.mlir \\
  --quantize INT8 \\
  --quant_input \\
  --calibration_table workspace/{model_name}_cali_table \\
  --processor {processor} \\
  --tolerance 0.9,0.6 \\
  --model workspace/{model_name}_int8.cvimodel
"""
    else:
        docker_cmd[-1] += f"""
echo '=== Step 2: MLIR → cvimodel (BF16) ==='
model_deploy.py \\
  --mlir workspace/{model_name}.mlir \\
  --quantize BF16 \\
  --quant_input \\
  --processor {processor} \\
  --tolerance 0.99,0.99 \\
  --model workspace/{model_name}_bf16.cvimodel
"""

    docker_cmd[-1] += """
echo '=== Done ==='
ls -la workspace/*.cvimodel
"""

    print(f"  运行: {' '.join(docker_cmd[:5])} ...")
    result = subprocess.run(docker_cmd, cwd=str(onnx_abs.parent))
    return result.returncode == 0


def print_manual_steps(onnx_path, model_name, input_size, quantize="BF16"):
    """Print manual conversion steps when Docker is unavailable."""
    onnx_dir = Path(onnx_path).resolve().parent

    print(f"""
[3/4] ⚠ TPU MLIR 工具需要 Linux 环境

方案A: GitHub Actions 云端转换 (推荐, 无需安装)
──────────────────────────────────────────────────
已配置好 .github/workflows/convert.yml
推送代码到 GitHub 即可自动转换:
  git add . && git commit -m "convert model"
  git remote add origin <你的仓库地址>
  git push -u origin main
然后在 Actions 页面下载生成的 .cvimodel 文件。

方案B: Docker Desktop (需管理员安装一次)
──────────────────────────────────────────────────
docker pull sophgo/tpuc_dev:latest
cd "{onnx_dir}"
docker run --rm -v "%cd%":/workspace sophgo/tpuc_dev:latest \\
  bash -c "cd /workspace && bash convert_model.sh"

方案C: Linux 机器直接运行
──────────────────────────────────────────────────
pip install tpu_mlir
cd "{onnx_dir}"
bash convert_model.sh

[4/4] .mud 文件已生成: {model_name}.mud
──────────────────────────────────────────────────
转换完成后，将 .cvimodel 和 .mud 文件一起拷贝到 MaixCam 设备即可。
""")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO .pt → MaixCam .cvimodel + .mud 一条龙转换"
    )
    parser.add_argument("pt_path", nargs="?", default="best.pt",
                        help=".pt 模型文件路径 (默认: best.pt)")
    parser.add_argument("--model-name", "-n", default=None,
                        help="模型名称 (默认从文件名推断)")
    parser.add_argument("--quantize", "-q", choices=["BF16", "INT8"],
                        default="BF16",
                        help="量化方式: BF16(无需校准图) 或 INT8(需要校准图)")
    parser.add_argument("--input-size", "-s", type=int, default=640,
                        help="输入尺寸 (默认: 640)")
    parser.add_argument("--calibration-dir", "-c", default="calibration_images",
                        help="INT8 校准图片目录 (默认: calibration_images)")
    parser.add_argument("--processor", "-p", default="cv181x",
                        help="目标处理器 (默认: cv181x/MaixCam)")
    parser.add_argument("--no-docker", action="store_true",
                        help="跳过 Docker 转换步骤")
    parser.add_argument("--no-onnx", action="store_true",
                        help="跳过 ONNX 导出 (使用已有的 .onnx)")
    parser.add_argument("--onnx-path", default=None,
                        help="已有 ONNX 文件路径")
    parser.add_argument("--opset", type=int, default=12,
                        help="ONNX opset 版本 (默认: 12)")

    args = parser.parse_args()

    pt_path = Path(args.pt_path)
    if not pt_path.exists():
        print(f"错误: 找不到模型文件 {pt_path}")
        sys.exit(1)

    print("=" * 60)
    print("  YOLO .pt → MaixCam .cvimodel + .mud 转换工具")
    print("=" * 60)

    # Step 1: Get model info
    if check_python():
        info = get_model_info(pt_path)
    else:
        print("警告: 未安装 ultralytics, 使用默认参数")
        info = {
            "task": "detect",
            "names": {},
            "num_classes": 52,
            "input_size": args.input_size,
            "model_name": args.model_name or pt_path.stem,
        }

    model_name = args.model_name or info["model_name"]
    input_size = info.get("input_size", args.input_size)
    quantize = args.quantize

    print(f"\n模型: {model_name}")
    print(f"类别: {info['num_classes']} 类")
    print(f"输入: {input_size}x{input_size}")
    print(f"量化: {quantize}")

    # Step 2: ONNX export
    if args.no_onnx:
        onnx_path = args.onnx_path or str(pt_path.with_suffix(".onnx"))
        if not Path(onnx_path).exists():
            print(f"错误: ONNX 文件不存在: {onnx_path}")
            sys.exit(1)
        print(f"\n[1/4] 使用已有 ONNX: {onnx_path}")
    else:
        onnx_path = args.onnx_path or str(pt_path.with_suffix(".onnx"))
        if not check_python():
            print("错误: 需要 PyTorch + ultralytics 来导出 ONNX")
            print("  安装: pip install torch ultralytics onnx")
            sys.exit(1)
        export_onnx(pt_path, onnx_path, input_size, args.opset)

    # Step 3: Generate .mud
    mud_path = ROOT / f"{model_name}.mud"
    generate_mud(info, mud_path, quantize)

    # Step 4: Docker conversion
    if args.no_docker:
        print("\n[3/4] 跳过 Docker 转换 (--no-docker)")
        print_manual_steps(onnx_path, model_name, input_size, quantize)
    elif check_docker():
        success = run_docker_convert(
            onnx_path, model_name, input_size, quantize,
            args.calibration_dir, args.processor
        )
        if success:
            print(f"\n[4/4] ✓ 转换完成!")
            cvi = f"{model_name}_{quantize.lower()}.cvimodel"
            print(f"  输出文件: workspace/{cvi}")
            print(f"  配置文件: {mud_path.name}")
            print(f"\n将这两个文件拷贝到 MaixCam 设备即可使用。")
        else:
            print("\n✗ Docker 转换失败，请检查错误信息")
            sys.exit(1)
    else:
        print_manual_steps(onnx_path, model_name, input_size, quantize)

    print("\n" + "=" * 60)
    print("  完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
