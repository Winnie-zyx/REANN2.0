import re

def calculate_bracket_values(file_path):
    """
    读取文件，提取所有中括号中的值，计算每个中括号内值的乘积，然后将所有乘积相加
    
    参数:
        file_path: 输入文件路径（通常是.out文件）
        
    返回:
        所有中括号内值的乘积之和
    """
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 使用正则表达式匹配所有中括号及其内容
        # 假设中括号中的值是数字，用逗号或空格分隔
        pattern = r'\[(.*?)\]'
        matches = re.findall(pattern, content)
        
        total_sum = 0
        
        # 处理每个匹配到的中括号内容
        for i, match in enumerate(matches, 1):
            # 提取数字，支持整数、小数和负数
            numbers = re.findall(r'-?\d+\.?\d*', match)
            
            if not numbers:
                print(f"警告：第{i}个中括号中未找到有效数字")
                continue
            
            # 将字符串转换为浮点数
            try:
                numbers = [float(num) for num in numbers]
            except ValueError as e:
                print(f"警告：第{i}个中括号中的数字转换失败: {e}")
                continue
            
            # 计算乘积
            product = 1
            for num in numbers:
                product *= num
            
            print(f"第{i}个中括号: {numbers}，乘积 = {product}")
            total_sum += product
        
        print(f"\n所有中括号乘积的总和: {total_sum}")
        return total_sum
    
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'")
        return None
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return None

if __name__ == "__main__":
    # 替换为你的.out文件路径
    file_path = "out"
    calculate_bracket_values(file_path)

