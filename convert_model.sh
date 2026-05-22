#!/bin/bash
set -e
# ============================================================
# YOLO .pt → MaixCam (.cvimodel) 一条龙转换脚本
# ============================================================
# 用法:
#   bash convert_model.sh [MODEL_NAME] [QUANTIZE]
#
# 示例:
#   bash convert_model.sh yolo11n_card BF16
#   bash convert_model.sh yolo11n_card INT8
#
# 运行环境:
#   方案1: Sophgo Docker 容器内直接运行
#     docker run --rm -v "$(pwd)":/workspace -it sophgo/tpuc_dev:latest
#     cd /workspace && bash convert_model.sh
#
#   方案2: 使用 docker-entrypoint
#     docker run --rm -v "$(pwd)":/workspace \
#       -e MODEL_NAME=yolo11n_card -e QUANTIZE=BF16 \
#       converter
# ============================================================

NET_NAME="${1:-yolo11n_card}"
QUANTIZE="${2:-INT8}"
INPUT_W=640
INPUT_H=640

echo "=========================================="
echo " YOLO → MaixCam 模型转换"
echo "=========================================="
echo "Model:      ${NET_NAME}"
echo "Input:      ${INPUT_W}x${INPUT_H}"
echo "Quantize:   ${QUANTIZE}"
echo "=========================================="

# Ensure workspace directory exists
mkdir -p workspace

# Step 1: ONNX → MLIR
echo ""
echo "=========================================="
echo "Step 1: ONNX → MLIR (model_transform)"
echo "=========================================="

if [ ! -f "${NET_NAME}.onnx" ] && [ ! -f "best.onnx" ]; then
    echo "ERROR: No .onnx file found."
    echo "Run: python convert.py best.pt --no-docker  to export ONNX first."
    exit 1
fi

ONNX_FILE="${NET_NAME}.onnx"
[ -f "$ONNX_FILE" ] || ONNX_FILE="best.onnx"
echo "Using ONNX: $ONNX_FILE"

model_transform.py \
    --model_name "${NET_NAME}" \
    --model_def "${ONNX_FILE}" \
    --input_shapes "[[1,3,${INPUT_H},${INPUT_W}]]" \
    --mean "0,0,0" \
    --scale "0.00392156862745098,0.00392156862745098,0.00392156862745098" \
    --pixel_format rgb \
    --channel_format nchw \
    --output_names "output0" \
    --tolerance 0.99,0.99 \
    --mlir "workspace/${NET_NAME}.mlir"

# Step 2: Calibration (INT8 only) or skip (BF16)
if [ "$QUANTIZE" = "INT8" ]; then
    echo ""
    echo "=========================================="
    echo "Step 2: Generate calibration table (run_calibration)"
    echo "=========================================="

    if [ -d "calibration_images" ] && [ "$(ls -A calibration_images/*.{jpg,jpeg,png} 2>/dev/null)" ]; then
        IMG_COUNT=$(ls calibration_images/*.jpg calibration_images/*.jpeg calibration_images/*.png 2>/dev/null | wc -l)
        echo "Using ${IMG_COUNT} calibration images from calibration_images/"
        run_calibration.py "workspace/${NET_NAME}.mlir" \
            --dataset calibration_images \
            --input_num "${IMG_COUNT}" \
            -o "workspace/${NET_NAME}_cali_table"
    else
        echo "WARNING: No calibration images found in calibration_images/"
        echo "Falling back to BF16 quantization."
        QUANTIZE="BF16"
    fi
fi

# Step 3: MLIR → cvimodel
echo ""
echo "=========================================="
echo "Step 3: MLIR → cvimodel (model_deploy)"
echo "=========================================="

MODEL_SUFFIX=$(echo "$QUANTIZE" | tr '[:upper:]' '[:lower:]')

if [ "$QUANTIZE" = "INT8" ]; then
    echo "Deploying with INT8 quantization..."
    model_deploy.py \
        --mlir "workspace/${NET_NAME}.mlir" \
        --quantize INT8 \
        --quant_input \
        --calibration_table "workspace/${NET_NAME}_cali_table" \
        --processor cv181x \
        --tolerance 0.9,0.6 \
        --model "workspace/${NET_NAME}_${MODEL_SUFFIX}.cvimodel"
else
    echo "Deploying with BF16 quantization..."
    model_deploy.py \
        --mlir "workspace/${NET_NAME}.mlir" \
        --quantize BF16 \
        --quant_input \
        --processor cv181x \
        --tolerance 0.99,0.99 \
        --model "workspace/${NET_NAME}_${MODEL_SUFFIX}.cvimodel"
fi

# Done
echo ""
echo "=========================================="
echo " Conversion complete!"
echo "=========================================="
ls -la workspace/*.cvimodel 2>/dev/null || echo "No cvimodel found"
echo ""
echo "Deploy to MaixCam:"
echo "  1. Copy workspace/*.cvimodel to the device"
echo "  2. Copy ${NET_NAME}.mud to the device"
echo "  3. Load model with:"
echo "     from maix import camera, display, nn"
echo "     model = nn.NN('/path/to/${NET_NAME}.mud')"
