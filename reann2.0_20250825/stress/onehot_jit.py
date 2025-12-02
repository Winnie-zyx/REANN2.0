import torch

@torch.jit.script
def to_one_hot(tensor: torch.Tensor, num_classes: int = 118) -> torch.Tensor:
    """
    仅支持一维输入的 TorchScript 兼容独热编码实现
    参数:
        tensor: 1D张量 [Natom]，存储每个原子的序数（0~num_classes-1）
        num_classes: 类别数，默认118（对应元素周期表118种元素）
    返回:
        2D独热编码张量 [Natom, num_classes]
    """
    # 1. 强制转为long类型（索引必须是整数，TorchScript要求）
    tensor_long = tensor.long()
    
    # 2. 严格校验：仅允许1D输入
    if tensor_long.dim() != 1:
        raise ValueError(f"Input must be 1D tensor, got {tensor_long.dim()}D")
    
    # 3. 获取原子数量（1D张量的长度）
    Natom: int = tensor_long.size(0)
    
    # 4. 初始化全0独热编码张量
    one_hot = torch.zeros(
        Natom, num_classes,  # 2D形状 [Natom, num_classes]
        dtype=torch.float32,
        device=tensor_long.device
    )
    
    # 5. 用scatter_实现独热编码（TorchScript完全兼容）
    # 扩展为 [Natom, 1]，匹配scatter_的索引维度要求
    tensor_expanded = tensor_long.unsqueeze(-1)
    one_hot.scatter_(dim=-1, index=tensor_expanded, value=1.0)
    
    return one_hot
