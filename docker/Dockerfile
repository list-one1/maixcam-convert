FROM sophgo/tpuc_dev:latest

# Pre-install any additional dependencies
RUN pip install --upgrade pip && \
    pip install onnx onnxruntime numpy pillow

# Create workspace
RUN mkdir -p /workspace /data/calibration_images /data/output

WORKDIR /workspace

# Copy conversion scripts
COPY scripts/docker_entrypoint.sh /usr/local/bin/convert
RUN chmod +x /usr/local/bin/convert

ENTRYPOINT ["/usr/local/bin/convert"]
