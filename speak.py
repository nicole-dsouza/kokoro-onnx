import os;
import onnxruntime as ort;

# set up statements (i.e. provider) that should come before kokoro import statement
os.environ["ONNX_PROVIDER"] = "CPUExecutionProvider";

# os.environ["ONNX_PROVIDER"] = "OpenVINOExecutionProvider";

# OPENVINO_LIBS = r"C:\Users\nicol\Development\kokoro-onnx\.venv\Lib\site-packages\openvino\libs";

# os.add_dll_directory(OPENVINO_LIBS);
# os.environ["PATH"] = OPENVINO_LIBS + os.pathsep + os.environ["PATH"];

# can't use DirectML as Microsoft's DirectML driver has a tensor shape handling bug inside its ConvTranspose operation node. when Kokoro converts text embeddings into soundwaves, DirectML lib passes a data structure that causes Intel driver to throw an explicit 80070057 'The parameter is incorrect'

# verify provider
# _original_InferenceSession = ort.InferenceSession

# def _debug_InferenceSession(*args, **kwargs):
#     session = _original_InferenceSession(*args, **kwargs)
#     print("ACTUAL PROVIDERS:", session.get_providers())
#     return session

# ort.InferenceSession = _debug_InferenceSession

_original_InferenceSession = ort.InferenceSession

def _patched_InferenceSession(*args, **kwargs):
    so = kwargs.pop("sess_options", None)

    if so is None:
        so = ort.SessionOptions()

        # how many CPU threads can work on it simultaneously
        # intra=16 median is 12.16s vs. default 12.93s
        so.intra_op_num_threads = 16;

        # multiple ops can run concurrently. how many ops can I execute at once
        # 0.17s average improvement tho median is essentially identical
        # so.inter_op_num_threads = 10;

    return _original_InferenceSession(
        *args,
        sess_options=so,
        **kwargs
    )

ort.InferenceSession = _patched_InferenceSession

# original + cpu: 11 to 13
# patch v1 + cpu: 11 to 13

# original + ovi: error
# patch v1 + ovi: 09 to 29

import numpy as np;
import soundfile as sf;
import sys;
import time;

from debug_scripts.batch_test import benchmark_kokoro;
from huggingface_hub import hf_hub_download;
from kokoro_onnx import Kokoro;

# kokoro model and voices file
MODEL_FILE  = "onnx/model.onnx";
VOICES_FILE = "model_files/voices.bin";

# MODEL_FILE  = "onnx-patched/model_resize4d.onnx";

print("📝 Step 1 - Validating Kokoro model and voice files exist ...");

if MODEL_FILE == "onnx/model.onnx" and not os.path.exists(MODEL_FILE):
    print("⏳ Downloading genuine Kokoro ONNX model weights (approx. 330MB) ...")
    MODEL_FILE = hf_hub_download(
        repo_id="onnx-community/Kokoro-82M-v1.0-ONNX", 
        filename="onnx/model.onnx", 
        local_dir="."
    );
    print("✅ Model download complete.")

if not os.path.exists(VOICES_FILE):
    print("⏳ Downloading genuine voice profiles package ...")
    VOICES_FILE = hf_hub_download(
        repo_id="speaches-ai/Kokoro-82M-v1.0-ONNX", 
        filename="voices.bin", 
        local_dir="."
    );
    print("✅ Voices download complete.")

print("⏳ Step 2 - Loading Kokoro ONNX Engine ...");

onnx_engine = Kokoro(MODEL_FILE, VOICES_FILE); # onnx model

voice = onnx_engine.get_voice_style("af_bella");

text = (
    "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since 1966, when designers at Letraset and James Mosley, the librarian at St Bride Printing Library in London, took a 1914 Cicero translation and scrambled it to make dummy text for Letraset's Body Type sheets. It has survived not only many decades, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised thanks to these sheets and more recently with desktop publishing software like Aldus PageMaker and Microsoft Word including versions of Lorem Ipsum."
);

print("🚀 Step 3 - Synthesizing audio on Intel Arc GPU ...");

start_time = time.time();

# generate raw audio and sample rate
# samples, sample_rate = onnx_engine.create(text, voice, speed=1.0, lang="en-us");
samples, sample_rate, results = benchmark_kokoro(onnx_engine, text, voice);

end_time = time.time();
print(f"✅ Success! Generated speech in {end_time - start_time:.2f} seconds.");

# save resulting audio data
output_filename = "output.wav";

sf.write(output_filename, samples, sample_rate);
print(f"💾 Audio saved successfully to: {output_filename}");