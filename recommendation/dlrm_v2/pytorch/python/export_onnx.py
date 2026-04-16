import torch
from backend_pytorch_native import build_dlrm_v2_native  # 官方模型构造函数

# 构建模型实例（随机初始化）
model = build_dlrm_v2_native()
model.eval()

# ===== 构造 dummy 输入 =====
batch_size = 1

# dense 特征
dense = torch.randn(batch_size, model.config.num_dense_features)

# sparse 特征数量
num_sparse = len(model.config.sparse_arch.embedding_bag_collection.embedding_bags)

# EmbeddingBag 需要 indices + offsets
indices = torch.randint(0, 100000, (batch_size * num_sparse,), dtype=torch.long)
offsets = torch.arange(0, batch_size * num_sparse, step=num_sparse, dtype=torch.long)

# ===== 导出 ONNX =====
torch.onnx.export(
    model,
    (dense, indices, offsets),
    "dlrm_v2_official.onnx",
    opset_version=17,
    input_names=["dense", "indices", "offsets"],
    output_names=["output"],
    dynamic_axes={
        "dense": {0: "batch"},
        "output": {0: "batch"},
    },
)

print("ONNX 模型导出完成：dlrm_v2_official.onnx")