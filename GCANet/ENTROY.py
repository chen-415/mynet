import os
import cv2
import numpy as np
from skimage.measure import shannon_entropy

def calculate_image_entropy(image):
    """
    计算单张图像的熵值。
    参数：
        image (numpy.ndarray): 输入的图像，灰度图。
    返回：
        float: 图像的熵值。
    """
    if len(image.shape) == 3:  # 转换为灰度图
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return shannon_entropy(image)

def calculate_folder_entropy_average(folder_path):
    """
    计算文件夹中所有图像的平均熵值。
    参数：
        folder_path (str): 文件夹路径。
    返回：
        float: 文件夹中图像的平均熵值。
    """
    entropy_values = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 检查是否为图像文件
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                # 读取图像
                image = cv2.imread(file_path)
                if image is not None:
                    entropy = calculate_image_entropy(image)
                    entropy_values.append(entropy)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    # 计算平均熵值
    if entropy_values:
        return np.mean(entropy_values)
    else:
        raise ValueError("No valid images found in the specified folder.")

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing images: ")
    try:
        average_entropy = calculate_folder_entropy_average(folder_path)
        print(f"Average Entropy of images in folder: {average_entropy:.4f}")
    except Exception as e:
        print(f"Error: {e}")
