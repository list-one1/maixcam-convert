#!/bin/bash
# Docker entrypoint for YOLO ONNX → cvimodel conversion
# Usage: docker run -v $(pwd):/workspace converter [OPTIONS]

set -e

MODEL_NAME="${MODEL_NAME:-yolov8m_card}"
INPUT_SIZE="${INPUT_SIZE:-640}"
QUANTIZE="${QUANTIZE:-BF16}"
PROCESSOR="${PROCESSOR:-cv181x}"

WORKSPACE=/workspace/workspace
mkdir -p "$WORKSPACE"

ONNX_FILE=$(ls /workspace/*.onnx 2>/dev/null | head -1)
if [ -z "$ONNX_FILE" ]; then
    echo "ERROR: No .onnx file found in /workspace"
    echo "Place your ONNX model in the mounted directory."
    exit 1
fi

echo "=========================================="
echo " YOLO ONNX → MaixCam cvimodel Converter"
echo "=========================================="
echo "Model:      $MODEL_NAME"
echo "Input:      ${INPUT_SIZE}x${INPUT_SIZE}"
echo "Quantize:   $QUANTIZE"
echo "Processor:  $PROCESSOR"
echo "ONNX:       $(basename $ONNX_FILE)"
echo "=========================================="

# Step 1: ONNX → MLIR
echo ""
echo "=== Step 1: ONNX → MLIR ==="
model_transform.py \
    --model_name "$MODEL_NAME" \
    --model_def "$ONNX_FILE" \
    --input_shapes "[[1,3,${INPUT_SIZE},${INPUT_SIZE}]]" \
    --mean "0,0,0" \
    --scale "0.00392156862745098,0.00392156862745098,0.00392156862745098" \
    --pixel_format rgb \
    --channel_format nchw \
    --output_names "/model.22/dfl/conv/Conv_output_0,/model.22/Sigmoid_output_0" \
    --tolerance 0.99,0.99 \
    --mlir "$WORKSPACE/${MODEL_NAME}.mlir"

# Step 2: Calibration (INT8 only)
if [ "$QUANTIZE" = "INT8" ]; then
    echo ""
    echo "=== Step 2: Calibration (INT8) ==="
    CALIB_DIR=""
    if [ -d "/workspace/calibration_images" ] && [ "$(ls -A /workspace/calibration_images 2>/dev/null)" ]; then
        CALIB_DIR="/workspace/calibration_images"
    elif [ -d "/data/calibration_images" ] && [ "$(ls -A /data/calibration_images 2>/dev/null)" ]; then
        CALIB_DIR="/data/calibration_images"
    fi

    if [ -n "$CALIB_DIR" ]; then
        IMG_COUNT=$(ls "$CALIB_DIR"/*.{jpg,jpeg,png} 2>/dev/null | wc -l)
        echo "Using $IMG_COUNT calibration images from $CALIB_DIR"
        run_calibration.py "${MODEL_NAME}.mlir" \
            --dataset "$CALIB_DIR" \
            --input_num "$IMG_COUNT" \
            -o "$WORKSPACE/${MODEL_NAME}_cali_table"
    else
        echo "WARNING: No calibration images found, falling back to BF16"
        QUANTIZE="BF16"
    fi
fi

# Step 3: MLIR → cvimodel
echo ""
echo "=== Step 3: MLIR → cvimodel ==="
MODEL_SUFFIX=$(echo "$QUANTIZE" | tr '[:upper:]' '[:lower:]')

if [ "$QUANTIZE" = "INT8" ]; then
    model_deploy.py \
        --mlir "$WORKSPACE/${MODEL_NAME}.mlir" \
        --quantize INT8 \
        --quant_input \
        --calibration_table "$WORKSPACE/${MODEL_NAME}_cali_table" \
        --processor "$PROCESSOR" \
        --tolerance 0.9,0.6 \
        --model "$WORKSPACE/${MODEL_NAME}_${MODEL_SUFFIX}.cvimodel"
else
    model_deploy.py \
        --mlir "$WORKSPACE/${MODEL_NAME}.mlir" \
        --quantize BF16 \
        --quant_input \
        --processor "$PROCESSOR" \
        --tolerance 0.99,0.99 \
        --model "$WORKSPACE/${MODEL_NAME}_${MODEL_SUFFIX}.cvimodel"
fi

echo ""
echo "=========================================="
echo " Conversion Complete!"
echo "=========================================="
ls -la "$WORKSPACE"/*.cvimodel 2>/dev/null || echo "No cvimodel files found"
echo ""
echo "Output files in workspace/:"
echo "  *.cvimodel  - Model file for MaixCam"
echo "  Copy the .cvimodel and .mud file to your device."
