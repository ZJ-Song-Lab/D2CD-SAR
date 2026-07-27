import os
import math
import xml.etree.ElementTree as ET

# 定义XML文件所在的目录
xml_dir = 'C:/Users/song/Desktop/trans/RBox_SSDD/voc_style/Annotations_test_offshore'
# 定义输出TXT文件的目录
output_dir = 'C:/Users/song/Desktop/trans/output'

# 如果输出目录不存在，则创建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历目录中的所有XML文件
for xml_file in sorted(os.listdir(xml_dir)):
    if xml_file.endswith('.xml'):
        # 解析XML文件
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()

        # 获取图片的宽度和高度
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)

        # 设置输出TXT文件的名称（与XML文件同名，但扩展名为.txt）
        txt_file = os.path.splitext(xml_file)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_file)

        # 遍历文件中的每个对象
        for obj in root.iter('object'):
            # 提取旋转边界框参数
            rotated_bndbox = obj.find('rotated_bndbox')

            if rotated_bndbox is not None:
                # 提取旋转边界框参数
                rotated_bbox_cx = float(rotated_bndbox.find('rotated_bbox_cx').text)
                rotated_bbox_cy = float(rotated_bndbox.find('rotated_bbox_cy').text)
                rotated_bbox_w = float(rotated_bndbox.find('rotated_bbox_w').text)
                rotated_bbox_h = float(rotated_bndbox.find('rotated_bbox_h').text)
                rotated_bbox_theta = math.radians(float(rotated_bndbox.find('rotated_bbox_theta').text))

            # 计算中心点偏移量
            half_w = rotated_bbox_w / 2
            half_h = rotated_bbox_h / 2

            # 计算旋转后的四个顶点坐标
            cos_theta = math.cos(rotated_bbox_theta)
            sin_theta = math.sin(rotated_bbox_theta)

            x1 = rotated_bbox_cx - half_w * cos_theta + half_h * sin_theta
            y1 = rotated_bbox_cy - half_w * sin_theta - half_h * cos_theta

            x2 = rotated_bbox_cx + half_w * cos_theta + half_h * sin_theta
            y2 = rotated_bbox_cy + half_w * sin_theta - half_h * cos_theta

            x3 = rotated_bbox_cx + half_w * cos_theta - half_h * sin_theta
            y3 = rotated_bbox_cy + half_w * sin_theta + half_h * cos_theta

            x4 = rotated_bbox_cx - half_w * cos_theta - half_h * sin_theta
            y4 = rotated_bbox_cy - half_w * sin_theta + half_h * cos_theta

            # 获取类名并转换为类索引（这里假设有一个类名到索引的映射）
            class_name = obj.find('name').text
            class_index = 0  # 这里应该根据你的实际类别映射来设置

            # 归一化坐标
            x1, y1, x2, y2, x3, y3, x4, y4 = (
                x1 / width, y1 / height,
                x2 / width, y2 / height,
                x3 / width, y3 / height,
                x4 / width, y4 / height
            )

            # 将坐标格式化为YOLO的OBB格式并写入文件
            with open(txt_path, 'a') as f:
                f.write(f"{class_index} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")

print("转换完成")
