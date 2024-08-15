import onnx
from onnxruntime.quantization import quantize_qat, QuantType

model_fp32 = "weights/yolov5s.onnx"
model_quant = "weights/yolov5s.quant.onnx"

# 加载FP32模型
onnx_model = onnx.load(model_fp32)

# 进行量化
quantized_model = quantize_qat(
    model=onnx_model,
    quantization_type=QuantType.QInt8,
    force_fusions=True
)

onnx.save_model(quantized_model, model_quant)
