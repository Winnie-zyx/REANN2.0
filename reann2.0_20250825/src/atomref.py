import torch

#def to_one_hot(tensor, num_classes=119):
def to_one_hot(tensor: torch.Tensor, num_classes: int = 119):
    """
    将[Nconf, Natom]张量转换为[Nconf, Natom, 118]的独热编码张量
    
    参数:
        tensor: 输入张量，形状为[Nconf, Natom]，元素值为原子序数（0-117）
        num_classes: 类别数，默认为118（对应元素周期表118种元素）
    
    返回:
        独热编码张量，形状为[Nconf, Natom, 118]
    """
    # 获取输入张量形状
    Nconf, Natom = tensor.shape
    # 初始化输出张量（全0）
    one_hot = torch.zeros(Nconf, Natom, num_classes, device=tensor.device)
    # 将对应位置设为1（使用高级索引）
    one_hot[torch.arange(Nconf)[:, None], torch.arange(Natom)[None, :], tensor] = 1
    return one_hot
def read_eref_to_tensor(eref_file="Eref"):
    """
    从Eref文件读取最后一列能量数据，转换为[118, 1]的PyTorch张量
    
    参数:
        eref_file: Eref文件路径，默认为"Eref"
    
    返回:
        torch.Tensor: 形状为[118, 1]的张量，包含118种元素的参考能量
    """
    energies = []
    with open(eref_file, 'r') as f:
        # 跳过表头（前2行，根据实际文件格式调整）
        # 如果文件没有表头，可删除这两行
        next(f)  # 跳过第一行表头
        next(f)  # 跳过第二行分隔线
        
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            # 分割行内容，取最后一列（能量值）
            parts = line.split()
            energy = float(parts[-1])  # 最后一列是能量
            energies.append([energy])  # 用列表包裹，确保维度为[118, 1]
    
    # 转换为PyTorch张量
    energy_tensor = torch.tensor(energies, dtype=torch.float32)
    
    # 验证形状是否正确
    if energy_tensor.shape != (119, 1):
        raise ValueError(f"读取的能量数据形状错误，应为(118, 1)，实际为{energy_tensor.shape}")
    
    return energy_tensor
