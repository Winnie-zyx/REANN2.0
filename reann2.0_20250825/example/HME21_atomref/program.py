import torch

def to_one_hot(tensor, num_classes=118):
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

# 示例测试
if __name__ == "__main__":
    # 输入张量：[[1,1,8],[1,1,8]]
    input_tensor = torch.tensor([[1, 1, 8], [1, 1, 8]])
    # 转换为独热编码
    output_tensor = to_one_hot(input_tensor)
    # 验证结果（查看第0个构型的第0个原子）
    print("第0个构型的独热编码：")
    print(output_tensor[0, 0, :10])  # 打印前10个元素，应在索引1处为1
    print(output_tensor[0, 1, :10])
    #print("\n第0个构型的第2个原子独热编码：")
    print(output_tensor[0, 2, :10])  # 打印前10个元素，应在索引8处为1
    print(output_tensor)
