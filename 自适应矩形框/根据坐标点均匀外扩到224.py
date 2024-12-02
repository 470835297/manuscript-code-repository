# 定义原始坐标
global_min_x, global_min_y = 197, 142
global_max_x, global_max_y = 357, 366

# 目标尺寸
target_width = 224
target_height = 224

# 计算原始矩形的宽度和高度
original_width = global_max_x - global_min_x
original_height = global_max_y - global_min_y

# 计算需要扩展的宽度和高度
expand_width = target_width - original_width
print('expand_width=',expand_width)
expand_height = target_height - original_height
print('expand_height=',expand_height)
# 均匀扩展坐标
new_min_x = global_min_x - expand_width // 2
print('new_min_x=',new_min_x)
new_max_x = global_max_x + expand_width // 2
new_min_y = global_min_y - expand_height // 2
new_max_y = global_max_y + expand_height // 2


# 打印新坐标
print(new_min_x, new_min_y, new_max_x, new_max_y)

