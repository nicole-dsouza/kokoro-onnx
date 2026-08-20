# kokoro-onnx

Kokoro ONNX - Offline TTS for Intel Arc GPU

# Setup Instructions

## Set up Python virtual environment

py -3.12 -m venv .venv

## Install Dependencies

.venv\Scripts\activate

pip install kokoro-onnx soundfile
pip install huggingface_hub
pip install onnx

pip install numpy==2.5.2
pip install onnxruntime-openvino==1.24.1
pip install openvino==2025.4.1

## Verify Setup

.\.venv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.get_available_providers())"

python -c "import openvino; from openvino import Core; print(Core().available_devices)"

.\.venv\Scripts\python.exe -c "import onnxruntime as ort; s=ort.InferenceSession('onnx/model.onnx', providers=['OpenVINOExecutionProvider','CPUExecutionProvider']); print(s.get_providers())"

## Run Project

.venv\Scripts\activate
python speak.py

## Debugging

Check OpenVIVO Execution Provider x OpenVIVO versions compatible versions at https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#requirements
