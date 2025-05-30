# 微调在ImageNet上预训练的卷积神经网络实现Caltech-101分类

任务描述：

本项目微调在ImageNet上预训练的卷积神经网络实现Caltech-101分类，模型权重下载地址：https://pan.baidu.com/s/1z6buig0mTdihUPoAlsbgrw?pwd=8phn

基本要求：
* 训练集测试集按照 [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) 标准；
* 修改现有的 CNN 架构（如 AlexNet、ResNet-18）用于 Caltech-101 识别，通过将其输出层大小设置为 101 以适应数据集中的类别数量，其余层使用在 ImageNet 上预训练得到的网络参数进行初始化；
* 在 Caltech-101 数据集上从零开始训练新的输出层，并对其余参数使用较小的学习率进行微调；
* 观察不同的超参数，如训练步数、学习率、及其不同组合带来的影响，并尽可能提升模型性能；
* 与仅使用 Caltech-101 数据集从随机初始化的网络参数开始训练得到的结果**进行对比**，观察预训练带来的提升。

## 1️⃣ 数据准备
Caltech-101 数据集已经提前下载并放在了本项目的`data/101_ObjectCategories`目录下，可以使用`utils.py`中的`get_dataloaders`直接进行加载

## 2️⃣ 超参数搜索
自动遍历一系列超参数组合，记录验证集准确率，搜索结果自动保存在 [`gridsearch_results_finetune.json`](gridsearch_results_finetune.json) 和  [`gridsearch_results_random.json`](gridsearch_results_random.json)中. 训练过程中的 loss 曲线和 accuracy 曲线会被自动放在 `training_plots_finetune` 和 `training_plots_random` 目录下.

``` bash
python search.py --exp finetune --epochs 10 
``` 

``` bash
python search.py --exp random --epochs 10 --no-pretrain 
``` 
你可以在 [`search.py`](search.py) 文件中的 hyper_param_opts 字典自定义搜索空间。

  ```python
hyper_param_opts = {
    "batch_size": [256, 128, 64],
    "lr": [0.01, 0.001],
    "step_size": [10, 30],
    "gamma": [0.5, 0.1],
    "weight_decay": [1e-4, 1e-5, 0.0],
}
  ```

## 3️⃣ 模型训练

* 你可以直接运行以下命令开始训练：

  ```bash
  # 在 Caltech-101 数据集上从零开始训练新的输出层，并对其余参数使用较小的学习率进行微调
  python train.py --exp best_finetune --epochs 20 --batch_size 64 --lr 0.01 --step_size 10 --gamma 0.5 --weight_decay 1e-5
  ```

  ```bash
   # 使用 Caltech-101 数据集从随机初始化的网络参数开始训练
   python train.py --exp best_random --no-pretrain --epochs 30 --batch_size 64 --lr 0.01 --step_size 30 --gamma 0.1 --weight_decay 0.0
  ```
  
模型训练结束后会分别生成 `best_finetune_best.pth` 和 `best_random_best.pth` 两个文件，分别是两种训练策略下保存的模型参数。同时在 `training_plots_finetune` 和 `training_plots_random` 文件夹下会生成相应的训练过程中在训练集和验证集上的 loss 曲线和 accuracy 曲线, 在 `run` 文件夹下会生成对应的 TensorBoard logs.

## 4️⃣ 模型测试
* 将模型权重文件下载后放于项目目录下，例如 `best_finetune_best.pth` 和 `best_random_best.pth`.

* 运行：

  ```python
  # 测试预训练微调的模型
  python test.py --exp best_finetune --batch_size --64
  ```
   ```python
  # 测试从头开始训练的模型
  python test.py --exp best_random --batch_size --64
  ```
  程序运行结束后会打印模型在测试集上的损失和正确率
  
## 5️⃣ Tensorboard可视化

 


  ```bash
  tensorboard --logdir runs --port 6006
  ```

## ✅ 实验结果
最终预训练微调模型在 Caltech-101 测试集上达到了**96.33**%的准确率，而从头开始训练的模型达到了**70.23**%的准确率.