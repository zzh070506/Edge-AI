# Edge-AI
## 机器学习可视化：监督学习与无监督学习对比（监督学习和无监督学习）
本项目通过Python代码实现了监督学习与无监督学习的可视化对比，使用逻辑回归和K均值算法分别展示了两类机器学习方法的基本原理和应用场景。

项目概述
本代码使用scikit-learn库生成模拟数据集，并分别应用监督学习（分类）和无监督学习（聚类）算法对数据进行处理，最后通过matplotlib将结果可视化，形成对比图。该项目非常适合机器学习初学者理解两类基本学习方法的区别和实际应用。

功能特性
✅ 生成模拟分类和聚类数据集

✅ 实现逻辑回归分类器训练和可视化

✅ 实现K均值聚类算法和结果可视化

✅ 绘制决策边界和聚类中心

✅ 中英文双语支持显示

✅ 清晰的图表标注和说明
## 循环
使用说明
​准备数据​：按照COCO格式组织数据集
​配置参数​：根据需要修改 args.yaml和 coco8.yaml
​开始训练​：运行 python train.py
​监控训练​：查看生成的曲线图和指标文件
​评估模型​：使用训练好的权重进行推理和评估

扩展应用
训练完成的模型可以用于：
实时目标检测
视频分析
图像识别应用
自动驾驶视觉系统
安防监控系统

注意事项：
确保训练数据标注准确且多样
根据硬件条件调整批次大小和图像尺寸
定期保存训练检查点防止意外中断
使用验证集监控模型是否过拟合

##开发环境版本检查工具（）
这是一个简单的 Python 脚本，用于检查当前开发环境中关键数据科学和机器学习库的版本信息。
功能描述
该脚本通过导入各主要库并输出其版本号，帮助开发者快速确认环境配置情况。这对于环境调试、项目依赖管理和协作开发非常有用。
检查的库版本
​Python: 3.13.5
​Jupyter Notebook: 7.3.2
​NumPy: 2.1.3
​Pandas: 2.2.3
​Matplotlib: 3.10.0
​PyTorch: 2.8.0+cpu
使用方法
确保已安装所需的库：
pip install notebook numpy pandas matplotlib torch
在 Jupyter Notebook 或 Python 环境中运行提供的代码：
# 输出Python版本
import sys
print(sys.version)

# 输出Jupyter Notebook版本
import notebook
print(f"Jupyter Notebook 版本：{notebook.__version__}")

# 输出NumPy版本
import numpy as np
print(f"NumPy 版本：{np.__version__}")

# 输出Pandas版本
import pandas as pd
print(f"Pandas 版本：{pd.__version__}")

# 输出Matplotlib版本
import matplotlib as mpl
print(f"Matplotlib 版本：{mpl.__version__}")

# 输出PyTorch版本
import torch
print(f"PyTorch 版本：{torch.__version__}")
环境要求
建议使用 Python 3.8+ 版本以获得最佳兼容性。该脚本适用于：

Jupyter Notebook/Lab

Google Colab

本地 Python 环境

虚拟环境（venv, conda等）

应用场景
项目环境配置验证

教学和培训环境检查

协作开发时的环境一致性确认

故障排除和调试

扩展建议
您可以扩展此脚本以检查更多库的版本，如：

Scikit-learn

TensorFlow

OpenCV

Seaborn

等其他数据科学常用库

注意事项
运行前请确保已安装所有需要检查的库

不同版本的库可能会有API差异，请根据实际需求选择合适的版本

对于生产环境，建议使用 requirements.txt 或 environment.yml 文件来管理依赖
