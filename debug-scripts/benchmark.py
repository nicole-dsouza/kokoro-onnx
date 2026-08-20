import os
import time
import numpy as np

os.environ["ONNX_PROVIDER"] = "CPUExecutionProvider"

from kokoro_onnx import Kokoro

k = Kokoro("onnx/model.onnx", "voices.bin")

v = k.get_voice_style("af_bella")

text = (
    "Lorem Ipsum is simply dummy text of the printing and typesetting industry. "
    "Lorem Ipsum has been the industry standard dummy text ever since 1966, "
    "when designers at Letraset and James Mosley, the librarian at St Bride Printing Library "
    "in London, took a 1914 Cicero translation and scrambled it to make dummy text for Letraset."
)

phonemes = k.tokenizer.phonemize(
    k.tokenizer.normalize_text(text),
    "en-us",
)

base = k.tokenizer.tokenize(phonemes)

print("base tokens:", len(base))

for n in [100, 200, 300, 320, 350, 400, 406]:

    tokens = base[:n]

    inputs = {
        "input_ids": np.array([[0, *tokens, 0]], dtype=np.int64),
        "style": np.asarray(
            k._style_for(v, len(tokens)),
            dtype=np.float32,
        ),
        "speed": np.array([1.0], dtype=np.float32),
    }

    times = []

    for i in range(5):
        start = time.perf_counter()
        k.sess.run(None, inputs)
        times.append(time.perf_counter() - start)

        print(
            f"{n:3} tokens -> "
            + ", ".join(f"{x:.3f}s" for x in times)
            + f" | avg(last 4): {np.mean(times[1:]):.3f}s"
        )