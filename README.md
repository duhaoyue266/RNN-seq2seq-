英法翻译 Seq2Seq（GRU + 注意力）

一个用 PyTorch 实现的极简英→法翻译示例：编码器 GRU + 解码器 GRU + 加性注意力。
包含数据预处理、训练、推理与注意力热力图可视化。

功能特点

GRU 编码器 / 解码器 + 注意力机制

训练/验证一体脚本（见 src/eng2fre.py）

安全加载权重（weights_only=True，避免反序列化代码风险）

按固定间隔记录并绘制训练损失

推理与注意力热力图保存（../data/attention.png）

明确的 SOS/EOS 处理与文本清洗流程

目录结构
.
├─ src/
│  └─ eng2fre.py              # 主脚本（训练 / 推理 / 可视化）
├─ data/
│  ├─ eng-fra-v2.txt          # 并行语料：每行 英文<TAB>法文
│  └─ attention.png           # 注意力图（运行后生成）
└─ save_model/
   ├─ encoder_gru.pth         # 训练后保存的编码器权重
   └─ atten_decoder_gru.pth   # 训练后保存的解码器权重


路径可在脚本中调整。

环境依赖

Python 3.9+

PyTorch 1.12+（可选 GPU）

numpy、pandas、tqdm、matplotlib

Conda 快速安装示例：

conda create -n seq2seq python=3.10 -y
conda activate seq2seq
# 选择你的平台与 CUDA/CPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas tqdm matplotlib

数据准备

将并行语料放到 data/eng-fra-v2.txt，每行格式：

<english sentence>\t<french sentence>


脚本会做统一清洗：

小写 + 去空白

标点前加空格

仅保留 [A-Za-z.!?]

训练与推理须使用同一清洗规则。

训练

在 src/ 目录下运行：

python eng2fre.py


默认会：

从语料构建词表

训练 epochs=2（可在脚本内修改）

保存权重到：

../save_model/encoder_gru.pth

../save_model/atten_decoder_gru.pth

按间隔打印/记录损失并保存曲线 ./loss.png

想按轮次分别保存，把保存文件名改为含 epoch 的形式，例如：

torch.save(encoder.state_dict(), f'../save_model/encoder_gru_epoch{epoch_idx:02d}.pth')
torch.save(decoder.state_dict(), f'../save_model/atten_decoder_gru_epoch{epoch_idx:02d}.pth')

推理 / 简单评估

脚本内提供 test_seq2seqEvaluate() 与 seq2seq_Ecaluate(...)。
它们会加载 ../save_model/*.pth，对若干示例句子做贪心解码并输出预测。

你可以修改 my_samplepairs 测试自己的句子（尽量保证词汇在训练词表里，否则会被当作 OOV 跳过）。

注意力可视化

plot_attention() 会：

加载已保存权重

对一条示例句前向，获取注意力权重

保存热力图到 ../data/attention.png

在脚本末尾 if __name__ == '__main__': 已调用 plot_attention()（你也可以改成调用训练或评估函数）。

关键实现细节

特殊符号：SOS=0，EOS=1

MAX_LENGTH：默认 10（决定注意力可视化的宽度；需要更长句子请相应调大）

批大小：训练/推理默认 batch_size=1

损失函数：NLLLoss（解码器输出 log-prob）

教师强迫：teacher_forcing_ratio=0.5

OOV 处理：当前无 UNK，推理时跳过 OOV 词（可按需添加 UNK）

安全加载（PyTorch）

使用：

state = torch.load(path, map_location=device, weights_only=True)
model.load_state_dict(state)


避免执行不受信的 pickle 代码

若你的 Torch 版本没有 weights_only，脚本会自动回退到常规 torch.load

常见问题排查
1) Windows 上 OMP 冲突崩溃（退出码 3）

报错类似：

OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.


临时解决（放在 导入 torch/numpy 前）：

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


长期建议：不要在同一环境混装 pip/conda 版的 MKL/OpenMP；尽量用 conda 统一安装。

2) 图片不显示

用非交互式后端确保保存成功：

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
...
plt.savefig('../data/attention.png', dpi=200, bbox_inches='tight')

3) 损失曲线不更新

确认训练循环里使用的是：

if item % plot_interval_num == 0:
    avg_plot_loss = plot_loss_total / plot_interval_num
    plot_loss_list.append(avg_plot_loss)
    plot_loss_total = 0.0

4) 变量名不一致导致的错误

english_word2index / french_index2word 等名称要与 get_data() 返回保持一致。

加载解码器权重时不要覆盖模型实例：

dec_state = torch.load('...pth', map_location=device, weights_only=True)
mydecoder.load_state_dict(dec_state)

个性化扩展

支持更长句子：调大 MAX_LENGTH

批训练：引入 padding + pack/pad 序列

UNK 机制：为未知词分配 ID，训练时适配

断点恢复：保存包含优化器与 epoch 的 checkpoint

torch.save({
    'epoch': epoch_idx,
    'encoder': encoder.state_dict(),
    'decoder': decoder.state_dict(),
    'enc_opt': enc_opt.state_dict(),
    'dec_opt': dec_opt.state_dict(),
}, f'../save_model/checkpoint_epoch{epoch_idx:02d}.pt')

许可证

建议使用 MIT（或你偏好的开源协议），在仓库根目录添加 LICENSE。

致谢

参考了 PyTorch 社区中的经典 Seq2Seq + Attention 教学示例。
本项目仅作教学/演示用，语料较小，指标（如 BLEU）不代表实际生产效果。
