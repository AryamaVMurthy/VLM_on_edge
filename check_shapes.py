import onnx

model = onnx.load("fastvlm_full_fp16_embedded.onnx", load_external_data=False)
print("Input Shapes:")
for input in model.graph.input:
    shape = [d.dim_value for d in input.type.tensor_type.shape.dim]
    print(f"{input.name}: {shape}")
