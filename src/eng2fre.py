# coding:utf-8
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import random
import matplotlib.pyplot as plt
import numpy
import pandas
from tqdm import tqdm
# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义开始结束字符：特殊token
SOS_token = 0
EOS_token = 1
# 定义句子最大长度（包含标点）
MAX_LENGTH = 10

# 定义数据集路径
data_path = '../data/eng-fra-v2.txt'

# 定义文本清洗函数
def normalString(s):
    """
    这是一个清洗函数
    第一步将字符串小写，去除前后空白
    第二步对字符串里的标点前面加上空格，和原始单词分开
    第三步除了正常字符，其他字符用空格替代
    :param s: 要输入的字符串
    :return: 清洗后的字符串
    """
    s1 = s.lower().strip()
    s2 = re.sub(r"([.!?])", r" \1", s1)
    s3 = re.sub(r"[^a-zA-Z.!?]+", r" ", s2)
    return s3

def get_data():
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    mypairs = [[normalString(s) for s in line.split('\t')] for line in lines]
    # 构建英文和法文字典
    english_word2index = {"SOS":0, "EOS":1}
    english_word_n = 2
    french_word2index = {"SOS":0, "EOS":1}
    french_word_n = 2
    for pair in mypairs:
        for word in pair[0].split(' '):
            if word not in english_word2index:
                english_word2index[word] = english_word_n
                english_word_n += 1
        for word in pair[1].split(' '):
            if word not in french_word2index:
                french_word2index[word] = french_word_n
                french_word_n += 1
    # 获取字典：
    english_index2word = {v:k for k,v in english_word2index.items()}
    french_index2word = {v:k for k,v in french_word2index.items()}
    return (english_word2index, english_index2word, english_word_n,
            french_word2index,french_index2word,french_word_n, mypairs)
(english_word2index, english_index2word, english_word_n,
 french_word2index,french_index2word,french_word_n, mypairs) = get_data()

class MyPairsDataset(Dataset):
    def __init__(self, mypairs):
        self.mypairs = mypairs
        self.sample_len = len(mypairs)

    def __len__(self):
        return self.sample_len
    def __getitem__(self, index):
        index = min(max(index, 0) , self.sample_len - 1)
        x = self.mypairs[index][0]
        y = self.mypairs[index][1]
        # 样本x数值化
        x = [english_word2index[word] for word in x.split(' ')]
        x.append(EOS_token)
        tensor_x = torch.tensor(x, dtype=torch.long, device=device)
        # 样本y数值化
        y = [french_word2index[word] for word in y.split(' ')]
        y.append(EOS_token)
        tensor_y = torch.tensor(y, dtype=torch.long, device=device)
        return tensor_x, tensor_y

# 构建数据迭代器
def get_dataloader():
    my_dataset = MyPairsDataset(mypairs)
    my_dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True)

class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(EncoderGRU, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embd = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)  # ★

    def forward(self, input, hidden):
        # input: (batch, seq_len) 或 (seq_len,)（单条样本）
        if input.dim() == 1:           # 兼容 batch_size=1 时可能出现的一维情况
            input = input.unsqueeze(0) # -> (1, seq_len)
        input = self.embd(input)       # -> (batch, seq_len, hidden)
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self, batch_size=1):  # ★ 接受 batch_size
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

def test_encoder():
    mydataset = MyPairsDataset(mypairs)
    mydataloader = DataLoader(mydataset, batch_size=1, shuffle=True)
    vocab_size = english_word_n
    hidden_size = 256
    my_encoder = EncoderGRU(vocab_size, hidden_size).to(device)

    for x, y in mydataloader:
        # DataLoader 默认在 CPU，上面 Dataset 已经返回在 device 上；若此处是 CPU，就搬到 device：
        x = x.to(device)
        hidden = my_encoder.initHidden(batch_size=x.size(0))  # ★ 用 x 的 batch 维
        output, hidden = my_encoder(x, hidden)
        print(output.shape)  # (batch, seq_len, hidden)
        print(hidden.shape)  # (num_layers, batch, hidden)
        break
class DecoderGRU(nn.Module):
    def __init__(self, french_vocab_size, hidden_size):
        super().__init__()
        self.vocab_size = french_vocab_size
        self.hidden_size = hidden_size
        self.embd = nn.Embedding(self.vocab_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        # 允许 input 是标量 / (batch,) / (batch,1) / (batch, seq_len)
        if input.dim() == 0:                  # 标量 -> (1,1)
            input = input.unsqueeze(0).unsqueeze(0)
        elif input.dim() == 1:                # (batch,) -> (batch,1)
            input = input.unsqueeze(1)

        emb = self.embd(input)                # (batch, seq_len, hidden)
        emb = F.relu(emb)
        output, hidden = self.gru(emb, hidden)
        logits = self.out(output[:, -1, :])   # 取最后一步 (batch, vocab)
        logprob = self.softmax(logits)
        return logprob, hidden

    def initHidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


def test_decoder():
    mydataset = MyPairsDataset(mypairs)
    mydataloader = DataLoader(mydataset, batch_size=1, shuffle=True)

    vocab_size = english_word_n
    hidden_size = 256
    my_encoder = EncoderGRU(vocab_size, hidden_size).to(device)
    my_decoder = DecoderGRU(french_word_n, hidden_size).to(device)

    for i, (x, y) in enumerate(mydataloader):  # ✅ 修正 enumerate 的解包
        x = x.to(device)
        y = y.to(device)

        hidden = my_encoder.initHidden(batch_size=x.size(0))
        enc_out, hidden = my_encoder(x, hidden)

        # 给解码器一个合法形状的输入：(batch, 1)
        # 这里用目标序列的第一个 token 做测试；实际训练通常用 SOS_token
        decoder_input = y[:, 0].unsqueeze(1)

        out, hidden = my_decoder(decoder_input, hidden)
        print(out.shape)   # 期望: (batch, vocab_size)
        break

# 定义带attention的解码器
class AttenDecoder(nn.Module):
    def __init__(self,vocab_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttenDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embd = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        # 定义第一个全连接层，得到注意力的权重分数
        self.attn = nn.Linear(hidden_size * 2, self.max_length)
        # 定义第二个全连接层，让注意力按照指定维度输出
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        # 定义gru
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    def forward(self, input, hidden, encoder_outputs):
        # 允许 input 是标量 / (batch,) / (batch,1) / (batch, seq_len)
        # input ->query [1,1] ->[[8]]
        # hidden ->key [1,1,256]
        # encoder_outputs ->value [max_len,256] ->[10,256]
        embedded = self.embd(input)
        # 经过dropout
        embedded = self.dropout(embedded)
        # 按照注意力计算步骤第一步，将qkv,按照第一计算规则计算
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=-1)
        # 将注意力权重和value进行矩阵乘法
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(dim=0))
        # 因为第一步是拼接，所以第二步要将Q和attn_applied再次拼接,在经过线性层
        atten = self.attn_combine(torch.cat((embedded[0], attn_applied[0]), 1).unsqueeze(0))
        # 激活函数relu
        atten = F.relu(atten)
        # 将atten和hidden送入gru
        gru_output, hidden = self.gru(atten, hidden)
        # 输出层
        result = self.softmax(self.out(gru_output[0]))
        return result, hidden, attn_weights

# 测试
def test_atten_decoder():
    mydataset = MyPairsDataset(mypairs)
    mydataloader = DataLoader(mydataset, batch_size=1, shuffle=True)
    vocab_size = english_word_n
    hidden_size = 256
    my_encoder = EncoderGRU(vocab_size, hidden_size).to(device)
    my_decoder = AttenDecoder(french_word_n, hidden_size).to(device)
    for i, (x, y) in enumerate(mydataloader):  # ✅ 修正 enumerate 的解包
        x = x.to(device)
        y = y.to(device)
        hidden = my_encoder.initHidden(batch_size=x.size(0))
        output, hidden = my_encoder(x, hidden)
        encoder_c = torch.zeros(MAX_LENGTH,hidden_size,device= device)
        for idx in range(output.shape[1]):
            encoder_c[idx] = output[0][idx]

        for idx in range(y.shape[1]):
            temp = y[0][idx].view(1,-1)
            decoder_output , hidden, attention_weights =my_decoder(temp, hidden, encoder_c)
            print(decoder_output.shape)
        break

my_lr = 1e-4
epochs = 2
teacher_forcing_ratio = 0.5
print_interval_num = 1000
plot_interval_num = 100


# 定义内部迭代函数
def train_iter(x, y, encoder, decoder, encoder_adam, decoder_adam, cross_entropy):
    # 将x也就是英文文本送入编码器得到编码结果
    encoder_output, encoder_hidden = encoder(x, encoder.initHidden())
    # 定义解码器输入，统一句子长度，得到value
    encoder_output_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    for i in range(x.shape[1]):
        encoder_output_c[i] = encoder_output[0, i]
    # 得到k值
    decoder_hidden = encoder_hidden
    # 得到Q值
    input_y = torch.tensor([[SOS_token]],device= device)

    my_loss = 0.0
    y_len = y.shape[1]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        for idx in range(y_len):
            output_y, decoder_hidden, decoder_attention = decoder(input_y, decoder_hidden, encoder_output_c)
            target_y = y[0][idx].view(1)
            my_loss += cross_entropy(output_y, target_y)
            input_y = y [0][idx].view(1, -1)
    else:
        for idx in range(y_len):
            output_y, decoder_hidden, decoder_attention = decoder(input_y, decoder_hidden, encoder_output_c)
            target_y = y[0][idx].view(1)
            my_loss += cross_entropy(output_y, target_y)# 定义模型训练函数
            # 获取下一个input_y
            topv, topi = torch.topk(output_y,k=1)
            if topi.item() == EOS_token:
                break
            input_y = topi.detach()
    # 梯度清零，反向传播，梯度更新
    encoder_adam.zero_grad()
    decoder_adam.zero_grad()
    my_loss.backward()
    encoder_adam.step()
    decoder_adam.step()
    return my_loss.item() / y_len
def model2train():
    # 获取数据
    my_dataset = MyPairsDataset(mypairs)
    my_dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True)
    # 实例化模型(编码器）
    english_vocab_size = english_word_n
    hidden_size = 256
    encoder_gru = EncoderGRU(english_vocab_size, hidden_size).to(device)
    # 实例化解码器
    french_vocab_size = french_word_n
    atten_decoder_gru = AttenDecoder(french_vocab_size, hidden_size).to(device)
    # 实例化优化器
    encoder_adam = optim.Adam(encoder_gru.parameters(), lr=my_lr)
    decoder_adam = optim.Adam(atten_decoder_gru.parameters(), lr=my_lr)
    # 实例化损失函数对象
    cross_entropy = nn.NLLLoss()
    # 定义一个训练日志的参数
    plot_loss_list = []
    # 开始外部循环
    for epoch_idx in range(1, 1 +epochs):
        # 初始化变量
        print_loss_total, plot_loss_total = 0, 0
        start_time = time.time()
        for item, (x, y) in enumerate(tqdm(my_dataloader), start=1):
            my_loss = train_iter(x, y, encoder_gru, atten_decoder_gru, encoder_adam, decoder_adam, cross_entropy)
            print_loss_total += my_loss
            plot_loss_total += my_loss
            if item % print_interval_num == 0:
                avg_print_loss = print_loss_total / print_interval_num
                print_loss_total = 0
                use_time = time.time() - start_time
                print('当前训练的轮次：%d, 损失值是：%.2f，时间是：%d' % (epoch_idx, avg_print_loss, use_time))
            if item % plot_interval_num == 0:
                avg_plot_loss = plot_loss_total / plot_interval_num
                plot_loss_list.append(avg_plot_loss)
                plot_loss_total = 0.0
        torch.save(encoder_gru.state_dict(), '../save_model/encoder_gru.pth')
        torch.save(atten_decoder_gru.state_dict(), '../save_model/atten_decoder_gru.pth')
    plt .figure()
    plt.plot(plot_loss_list)
    plt.savefig('./loss.png')
    plt.show()
    return plot_loss_list

# 测试模型预测函数
def safe_load_state_dict(path, device):
    try:
        # 新版 PyTorch：安全加载，仅反序列化张量
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # 旧版 PyTorch：没有 weights_only 参数，回退
        return torch.load(path, map_location=device)

# 测试模型预测函数
def seq2seq_Ecaluate(x, myencoder, mydecoder):
    with torch.no_grad():
        # 编码
        h0 = myencoder.initHidden(batch_size=x.size(0))
        encoder_output, encoder_hidden = myencoder(x, h0)

        # 准备注意力的 value（按你的实现：MAX_LENGTH x hidden）
        encoder_output_c = torch.zeros(MAX_LENGTH, myencoder.hidden_size, device=device)
        seq_len = x.shape[1]
        for idx in range(seq_len):
            encoder_output_c[idx] = encoder_output[0, idx]

        # 解码
        decoder_hidden = encoder_hidden
        input_y = torch.tensor([[SOS_token]], device=device)
        decoder_list = []
        decoder_attention = torch.zeros(MAX_LENGTH, MAX_LENGTH, device=device)

        for t in range(MAX_LENGTH):
            output_y, decoder_hidden, atten_weight = mydecoder(input_y, decoder_hidden, encoder_output_c)
            topv, topi = torch.topk(output_y, k=1)  # topi: (1,1)
            decoder_attention[t] = atten_weight[0]

            token_id = topi.item()
            if token_id == EOS_token:
                decoder_list.append('<EOS>')
                break
            else:
                # ✅ 正确的词典名
                decoder_list.append(french_index2word[token_id])

            # ✅ 别忘了调用 .detach()；topi 形状已是 (1,1)，可直接喂下一步
            input_y = topi.detach()

        return decoder_list, decoder_attention[:t+1]

def to_en_ids(x_str: str):
    # 与词表构建时一致的清洗
    x_str = normalString(x_str)
    ids = []
    for w in x_str.split(' '):
        if not w:
            continue
        if w in english_word2index:   # 无 UNK，跳过 OOV
            ids.append(english_word2index[w])
    ids.append(EOS_token)
    return torch.tensor([ids], dtype=torch.long, device=device)

def test_seq2seqEvaluate():
    myencoder = EncoderGRU(english_word_n, hidden_size=256).to(device)
    enc_state = safe_load_state_dict('../save_model/encoder_gru.pth', device)
    myencoder.load_state_dict(enc_state)
    myencoder.eval()

    mydecoder = AttenDecoder(french_word_n, hidden_size=256).to(device)
    dec_state = safe_load_state_dict('../save_model/atten_decoder_gru.pth', device)
    mydecoder.load_state_dict(dec_state)
    mydecoder.eval()

    my_samplepairs = [
        ['i am a boy', 'je suis un garcon'],
        ['i m impressed with your french .', 'je suis impressionné par votre français .'],
        ['i m more than a friend .', 'je suis plus qu un ami .'],
        ['she is beautiful like her mother .', 'elle est belle comme sa mère .']
    ]

    for x, y in my_samplepairs:
        tensor_x = to_en_ids(x)  # ✅ 使用清洗和词典
        decoder_list, decoder_attention = seq2seq_Ecaluate(tensor_x, myencoder, mydecoder)
        print('src:', x)
        print('pred:', ' '.join(decoder_list))
        print('tgt:', y)
        print('-' * 40)

# 定义函数：实现注意力权重的绘图
def plot_attention():
    # 1) 加载模型
    myencoder = EncoderGRU(english_word_n, hidden_size=256).to(device)
    enc_state = safe_load_state_dict('../save_model/encoder_gru.pth', device)
    myencoder.load_state_dict(enc_state)
    myencoder.eval()

    mydecoder = AttenDecoder(french_word_n, hidden_size=256).to(device)
    dec_state = safe_load_state_dict('../save_model/atten_decoder_gru.pth', device)
    mydecoder.load_state_dict(dec_state)
    mydecoder.eval()

    # 2) 准备输入（清洗 + 映射 + 加 batch 维）
    sentence = 'we are both teachers .'
    tensor_x = to_en_ids(sentence)  # -> (1, seq_len)

    # 3) 前向+获取注意力
    decoder_list, decoder_attention = seq2seq_Ecaluate(tensor_x, myencoder, mydecoder)
    predict = ' '.join(decoder_list)
    print('predict:', predict)

    # 4) 可视化注意力
    attn = decoder_attention.detach().cpu().numpy()  # (tgt_len, MAX_LENGTH)
    plt.figure()
    plt.imshow(attn, aspect='auto')     # 行：目标序列时间步；列：源序列对齐位置（最多 MAX_LENGTH）
    plt.colorbar()
    plt.title('Attention')
    plt.xlabel('Encoder time steps')
    plt.ylabel('Decoder time steps')
    plt.tight_layout()
    plt.savefig('../data/attention.png')
if __name__ == '__main__':
    plot_attention()
