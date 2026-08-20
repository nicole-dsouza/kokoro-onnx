import time


def benchmark_kokoro(onnx_engine, text, voice, runs=3):

    results = []
    samples = None
    sample_rate = None

    for i in range(runs):

        start_time = time.perf_counter()

        samples, sample_rate = onnx_engine.create(
            text,
            voice,
            speed=1.0,
            lang="en-us",
        )

        elapsed = time.perf_counter() - start_time
        results.append(elapsed)

        print(f"RUN {i + 1}: {elapsed:.2f}s")

    return samples, sample_rate, results