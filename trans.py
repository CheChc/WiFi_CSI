import os
import subprocess
import shutil

# 定义数据路径
data_path = 'E:\WiFi'
new_data_path = 'E:\WiFi_png'

# 动作标签
actions = ['clap', 'kick', 'pickup', 'run', 'sitdown', 'standup', 'walk', 'wavehand']

# 展开用户目录
data_path = os.path.expanduser(data_path)
new_data_path = os.path.expanduser(new_data_path)

# 确保新的数据路径存在
os.makedirs(new_data_path, exist_ok=True)

# 打开日志文件
log_file_path = os.path.join(new_data_path, 'conversion_log.txt')
with open(log_file_path, 'w') as log_file:
    # 遍历每个动作文件夹
    for action in actions:
        action_path = os.path.join(data_path, action)
        new_action_path = os.path.join(new_data_path, action)

        # 确保新的动作文件夹路径存在
        os.makedirs(new_action_path, exist_ok=True)

        # 确保原始路径存在
        if not os.path.isdir(action_path):
            log_file.write(f"路径 {action_path} 不存在，跳过...\n")
            continue

        # 遍历每个.dat文件
        for filename in os.listdir(action_path):
            if filename.endswith('.dat'):
                dat_file = os.path.join(action_path, filename)
                temp_image_file = os.path.join(action_path, filename.replace('.dat', '.png'))
                final_image_file = os.path.join(new_action_path, filename.replace('.dat', '.png'))

                # 切换到dat文件所在目录
                os.chdir(action_path)

                # 使用csikit生成图像
                command = f'csikit -g {filename}'
                log_file.write(f"正在处理文件: {dat_file}\n")
                try:
                    subprocess.run(command, shell=True, check=True)
                    # 移动并重命名生成的图像文件
                    if os.path.exists(temp_image_file):
                        shutil.move(temp_image_file, final_image_file)
                        log_file.write(f'图像生成并移动完成: {dat_file} -> {final_image_file}\n')
                    else:
                        log_file.write(f'图像生成失败: {temp_image_file} 不存在\n')
                except subprocess.CalledProcessError as e:
                    log_file.write(f'图像生成失败: {dat_file}\n')
                    log_file.write(f'错误信息: {e}\n')
                except FileNotFoundError as e:
                    log_file.write(f'移动图像失败: {temp_image_file} 不存在\n')
                    log_file.write(f'错误信息: {e}\n')

print(f'转换完成，日志文件保存在: {log_file_path}')
