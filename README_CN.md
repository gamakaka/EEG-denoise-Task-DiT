# TaskEEGDiT 中文说明

TaskEEGDiT 是一个基于扩散模型的脑电信号去噪项目。主模型采用 DDPM 框架，并使用基于 TimeMixer 的噪声预测网络，在反向扩散过程中预测高斯噪声。

项目支持无条件去噪和任务条件去噪。在条件去噪模式下，可以使用任务频谱原型作为紧凑的任务级条件信息，用于区分 sleep EEG 和 motor imagery EEG。

## 主要功能

- 基于 DDPM 的 EEG 去噪。
- 基于 TimeMixer 的噪声预测网络。
- 可选的任务频谱原型条件输入。
- 可选的 content loss 和频谱 style loss。
- 使用统一的 YAML 配置文件管理训练和测试。
- 内置 CC、SNR、时域 RRMSE 和频域 RRMSE 等评估指标。

## 项目结构

```text
config/
  config.yaml                  主配置文件。

src/
  data/                        数据集加载器。
  models/                      DDPM、模型构建函数和噪声预测网络。
  models/noise_predictors/     基于 TimeMixer 的噪声预测网络。
  utils/                       配置、日志、checkpoint 和随机种子工具。
  evaluate.py                  测试指标和测试阶段工具函数。
  train_loop.py                训练和验证循环。

timemixer/
  config.py                    TimeMixer 配置。
  exp/, layers/, models/, utils/
                               TimeMixer 的具体实现。

Data_Preparation/
  数据集准备脚本，保留用于后续处理其他数据集。

train.py                       训练入口。
test.py                        测试入口。
requirements.txt               Python 依赖。
```

## 环境安装

创建并激活 Python 环境后，安装依赖：

```bash
pip install -r requirements.txt
```

如果需要使用 GPU 训练或推理，请安装支持 CUDA 的 PyTorch。

## 配置文件

主配置文件为：

```text
config/config.yaml
```

主要配置部分包括：

- `project`：随机种子和运行设备。
- `data`：训练集、验证集和测试集路径。
- `model`：TimeMixer 参数和任务条件设置。
- `train`：优化器、batch size、epoch 数和辅助损失权重。
- `eval`：采样方式和 DDIM 参数。
- `diffusion`：DDPM 噪声调度参数。
- `output`：checkpoint、日志和测试结果输出目录。

任务条件输入由下面的配置控制：

```yaml
model:
  task_condition: true
  spectral_prototype_path: data/spectral_prototypes.pt
```

如果要训练或测试无条件模型，将 `task_condition` 设置为 `false`。

辅助损失由下面的配置控制：

```yaml
train:
  lambda_content: 0.005
  lambda_style: 0.001
  encoder_ckpt_path: path/to/encoder_checkpoint.pt
```

当某个辅助损失权重为 `0` 时，该损失不会被计算。当两个权重都为 `0` 时，训练退化为原始 DDPM 损失。

## 训练

使用默认配置训练：

```bash
python train.py
```

常用命令行覆盖参数：

```bash
python train.py --device cuda:0
python train.py --conditional
python train.py --no-conditional
python train.py --lambda-content 3e-5 --lambda-style 350
```

如果希望训练完成后立即测试：

```bash
python train.py --run-test-after-train
```

## 测试

使用默认配置测试：

```bash
python test.py
```

使用 DDIM 采样：

```bash
python test.py --sampler ddim --ddim-num-steps 20 --ddim-eta 0.0
```

测试指定辅助损失权重训练得到的条件模型：

```bash
python test.py \
  --conditional \
  --lambda-content 3e-5 \
  --lambda-style 350 \
  --sampler ddim \
  --ddim-num-steps 20
```

测试脚本会按照配置连续测试两个测试集。

## 任务频谱原型条件

TaskEEGDiT 可以使用任务频谱原型作为条件向量。prototype 文件应包含 shape 为 `[2, F]` 的两个类别原型，每一行是一个归一化后的频谱分布。

类别含义为：

- class `0`：sleep EEG prototype；
- class `1`：motor imagery EEG prototype。

在条件推理时，模型根据 label 选择对应的 prototype，将其编码后通过残差条件路径融入 TimeMixer 噪声预测网络。

## 数据格式

数据集加载器默认读取 NumPy 字典文件，至少包含：

```text
clean_eeg
noisy_eeg
label
```

EEG 信号形状应为 `[N, 1, 512]` 或 `[N, 512]`。标签应为整数类别 ID。

## 使用注意

- 数据路径、checkpoint 路径和输出路径应与 `config/config.yaml` 保持一致。
- 测试条件模型时，传入 `test.py` 的辅助损失权重应与训练该 checkpoint 时使用的权重一致。
- 如果已经准备好了预处理后的数据文件，则不需要运行数据准备脚本；数据准备脚本主要保留给后续新数据集处理使用。
