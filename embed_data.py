import onnx
import os

input_path = "fastvlm_full_fp16.onnx"
output_path = "fastvlm_full_fp16_embedded.onnx"

print(f"Loading {input_path} (with external data)...")
model = onnx.load(input_path, load_external_data=True)

print(f"Saving to {output_path} (embedded)...")
# save_model by default does NOT separate data unless size > 2GB (proto limit)
onnx.save_model(model, output_path)

size = os.path.getsize(output_path) / (1024*1024)
print(f"✓ Saved! Size: {size:.2f} MB")
