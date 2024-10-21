import matplotlib.pyplot as plt
import numpy as np
from configClass import config

def plot_two_arrays(MSEtrain, MSEtest):
    # Check if the input arrays are of the same length
    if len(MSEtrain) != len(MSEtest):
        raise ValueError("Input arrays must be of the same length")

    # Generate the index for the x-axis
    index = np.arange(len(MSEtrain))

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(index, MSEtrain, label='MSE_train', marker='o')
    plt.plot(index, MSEtest, label='MSE_test', marker='s')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Line Plot of Two Arrays')

    # Add legend
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.show()


def write_metrics_to_txt(file_path, mse, mae, config):
    # 将时间和字符串组合起来
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 获取 config 对象的所有属性
    config_attributes = "\n".join([f"{attr}: {value}" for attr, value in config.__dict__.items()])

    # 组合输出字符串
    line = f"{timestamp} Test MSE: {mse:.6f}, Test MAE: {mae:.6f}\n{config_attributes}\n\n"

    # 将字符串写入到txt文件的新一行
    with open(file_path, 'a') as file:
        file.write(line)


# 使用示例
# write_metrics_to_txt('metrics.txt', 0.123456, 0.234567)


if __name__ == "__main__":
# Example usage
#     array1 = [1, 3, 5, 7, 9]
#     array2 = [2, 4, 6, 8, 10]
#     plot_two_arrays(array1, array2)
    CONFIG = config()
    write_metrics_to_txt('../Experimental_Logger.txt', 0.123456, 0.234567, CONFIG)