import onnx
from onnx import helper, numpy_helper
import numpy as np

src = "onnx/model_stft_rank4.onnx"
dst = "onnx/model_resize4d.onnx"

m = onnx.load(src)
nodes = m.graph.node

target = "/decoder/decoder/generator/m_source/l_sin_gen/Resize"

idx = next(i for i, n in enumerate(nodes) if n.name == target)
resize = nodes[idx]

old_input = resize.input[0]
old_output = resize.output[0]

# New intermediate tensors
unsq_out = old_input + "_4d"
resize_out = old_output + "_4d"
squeeze_out = old_output + "_squeezed"

# Unsqueeze axis = 2:
# [1, 9, L] -> [1, 9, 1, L]
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

# Replace Resize scales [1, 1, 0.00333333]
# with [1, 1, 1, 0.00333333]
old_scales_name = resize.input[2]
old_scales = next(
    x for x in m.graph.initializer
    if x.name == old_scales_name
)

scales = numpy_helper.to_array(old_scales).astype(np.float32)

new_scales = np.array(
    [scales[0], scales[1], 1.0, scales[2]],
    dtype=np.float32,
)

new_scales_name = target + "_scales_4d"

m.graph.initializer.append(
    numpy_helper.from_array(new_scales, name=new_scales_name)
)

resize4d = helper.make_node(
    "Resize",
    [unsq_out, "", new_scales_name],
    [resize_out],
    name=target + "_4D",
)

# Preserve all Resize attributes.
resize4d.attribute.extend(resize.attribute)

# [1, 9, 1, L'] -> [1, 9, L']
squeeze_axes_name = target + "_squeeze_axes"

m.graph.initializer.append(
    numpy_helper.from_array(
        np.array([2], dtype=np.int64),
        name=squeeze_axes_name,
    )
)

squeeze = helper.make_node(
    "Squeeze",
    [resize_out, squeeze_axes_name],
    [squeeze_out],
    name=target + "_Squeeze4D",
)

# Rewire the following Transpose to consume the squeezed output.
next_node = next(
    n for n in nodes
    if old_output in n.input
)

for i, inp in enumerate(next_node.input):
    if inp == old_output:
        next_node.input[i] = squeeze_out

# Replace original Resize with our 3-node sequence.
nodes.remove(resize)
nodes.insert(idx, unsqueeze)
nodes.insert(idx + 1, resize4d)
nodes.insert(idx + 2, squeeze)

onnx.checker.check_model(m)
onnx.save(m, dst)

print("saved:", dst)
print("patched:", target)
print("scales:", new_scales)