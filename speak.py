import os;

# set up statements (i.e. provider) that should come before kokoro import statement
os.environ["ONNX_PROVIDER"]   = "CPUExecutionProvider";
os.environ["OMP_NUM_THREADS"] = "14"

import numpy as np;
import onnxruntime as ort;
import soundfile as sf;
import sys;
import time;

from huggingface_hub import hf_hub_download;
from kokoro_onnx import Kokoro;

# kokoro model and voices file
MODEL_FILE  = "onnx/model.onnx";
VOICES_FILE = "voices.bin";

print("📝 Step 1 - Validating Kokoro model and voice files exist ...");

if not os.path.exists(MODEL_FILE):
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
samples, sample_rate = onnx_engine.create(text, voice, speed=1.0, lang="en-us");

end_time = time.time();
print(f"✅ Success! Generated speech in {end_time - start_time:.2f} seconds.");

# save resulting audio data
output_filename = "output.wav";

sf.write(output_filename, samples, sample_rate);
print(f"💾 Audio saved successfully to: {output_filename}");