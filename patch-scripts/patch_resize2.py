import onnx
from onnx import helper, numpy_helper
import numpy as np

src = "onnx/model_resize4d.onnx"
dst = "onnx/model_resize4d_v2.onnx"

m = onnx.load(src)

target = "/decoder/decoder/generator/m_source/l_sin_gen/Resize_1"

nodes = m.graph.node
idx = next(i for i, n in enumerate(nodes) if n.name == target)
resize = nodes[idx]

old_input = resize.input[0]
old_output = resize.output[0]

# [1, 9, L] -> [1, 9, 1, L]
unsq_out = old_input + "_4d"
axes_name = target + "_axes"

m.graph.initializer.append(
    numpy_helper.from_array(
        np.array([2], dtype=np.int64),
        name=axes_name,
    )
)

unsqueeze = helper.make_node(
    "Unsqueeze",
    [old_input, axes_name],
    [unsq_out],
    name=target + "_Unsqueeze4D",
)

# Resize scales:
# [1, 1, scale] -> [1, 1, 1, scale]
old_scales = next(
    x for x in m.graph.initializer
    if x.name == resize.input[2]
)

scales = numpy_helper.to_array(old_scales).astype(np.float32)

new_scales = np.array(
    [scales[0], scales[1], 1.0, scales[2]],
    dtype=np.float32,
)

scales_name = target + "_scales_4d"

m.graph.initializer.append(
    numpy_helper.from_array(new_scales, name=scales_name)
)

resize_out = old_output + "_4d"

resize4d = helper.make_node(
    "Resize",
    [unsq_out, "", scales_name],
    [resize_out],
    name=target + "_4D",
)

resize4d.attribute.extend(resize.attribute)

# [1, 9, 1, L'] -> [1, 9, L']
squeeze_axes = target + "_squeeze_axes"

m.graph.initializer.append(
    numpy_helper.from_array(
        np.array([2], dtype=np.int64),
        name=squeeze_axes,
    )
)

squeeze_out = old_output + "_squeezed"

squeeze = helper.make_node(
    "Squeeze",
    [resize_out, squeeze_axes],
    [squeeze_out],
    name=target + "_Squeeze4D",
)

# Rewire consumers of the original Resize output.
for node in nodes:
    for i, inp in enumerate(node.input):
        if inp == old_output:
            node.input[i] = squeeze_out

nodes.remove(resize)
nodes.insert(idx, unsqueeze)
nodes.insert(idx + 1, resize4d)
nodes.insert(idx + 2, squeeze)

onnx.checker.check_model(m)
onnx.save(m, dst)

print("saved:", dst)
print("patched:", target)
print("scales:", new_scales)