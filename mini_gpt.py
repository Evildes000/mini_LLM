import re
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pylab as plt


decoder_config = {
    "context_length": 512,   # 位置编码的最大长度（需 >= 训练用的 window_size）
    "d_model": 512,
    "num_heads": 8,
    "d_k": 64,               # d_model / num_heads = 512 / 8 = 64
    "num_layers": 6
}

training_config = {
    "num_epochs": 100,
    "lr": 1e-4,
    "batch_size": 16,
    "window_size": 256,      # 训练序列长度（seq_len）
    "sliding_step": 64,      # 滑窗步长；step=1 会切出约 21 万个高度重叠样本，占近 1GB 内存
}


def tokenize(text):
    # 按单词和标点符号切分，忽略空白字符
    return re.findall(r"\w+|[^\w\s]", text)


def generate_vocab():
    """
    generate token-Id pair dict
    """
    with open('corpus_clean.txt', 'r', encoding='utf-8') as file:
        content = file.read()

    tokens = tokenize(content)

    # 用来保存单词-序号对
    unique_vocab_dict = {}

    for token in tokens:
        if token not in unique_vocab_dict:
            unique_vocab_dict[token] = len(unique_vocab_dict)

    print(f"length of vocab_size is: {len(unique_vocab_dict)}")

    return unique_vocab_dict


def generate_ids(vocab_dict: dict):
    """
    generate ID-token pairs
    """
    id_token = {id: word for word, id in vocab_dict.items()}

    return id_token


def word_to_id(content, vocab_dict):
    """
    Transfer each word to its corresponding id
    """
    ids = []

    for word in tokenize(content):
        if word in vocab_dict:
            ids.append(vocab_dict[word])

    return ids


def build_pairs(ids, window_size=256, step=1):
    """
    ids: all token ids
    window_size: how many token will be given to model (aka seq_len)
    step: sliding step

    build input-output pairs
    """
    inputs = []
    outputs = []

    # 起点应滑动到 len(ids) - window_size，而不是 int(len(ids)/window_size)
    for i in range(0, len(ids) - window_size, step):
        inputs.append(ids[i:i + window_size])
        outputs.append(ids[i + 1:i + window_size + 1])

    return inputs, outputs


class TextDataset(Dataset):
    def __init__(self, input, output):
        super().__init__()
        self.xs = torch.tensor(input, dtype=torch.long)
        self.ys = torch.tensor(output, dtype=torch.long)

    # __len__ and __getitem__ are necessary
    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


class Fnn(nn.Module):
    def __init__(self, hidden_layer=4 * decoder_config["d_model"]):
        super().__init__()
        # 这里fc1接受的维度d_model只要和输入x的最后一个维度匹配即可
        self.fc1 = nn.Linear(decoder_config["d_model"], hidden_layer)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer, decoder_config["d_model"])

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class LinearProj(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.proj = nn.Linear(decoder_config["d_model"], vocab_size)
        # softmax is included in CrossEntropy
        # self.softmax = torch.softmax()

    def forward(self, x):
        x = self.proj(x)
        return x


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(vocab_size, d_model))

    def forward(self, idx):
        return self.weights[idx]


class positional_encoding:
    def __init__(self):
        pass

    def __encoding__(self):
        """
        返回编码后的位置矩阵（torch.Tensor）
        """
        pos_mat = np.zeros((decoder_config["context_length"], decoder_config["d_model"]))
        for row in range(decoder_config["context_length"]):
            pos_mat[row, 0::2] = np.sin(row / 10000 ** (np.arange(0, decoder_config["d_model"], 2) / decoder_config["d_model"]))
            pos_mat[row, 1::2] = np.cos(row / 10000 ** (np.arange(0, decoder_config["d_model"], 2) / decoder_config["d_model"]))

        return torch.tensor(pos_mat, dtype=torch.float32)


def plot_positional_encoding(pos_mat):
    """
    pos_mat: positional encoded matrix
    This function is used to plot positional encoding
    """
    plt.imshow(pos_mat)
    plt.xlabel("d_model")
    plt.ylabel("pos")
    plt.show()


class LayerNorm(nn.Module):
    """
    LayerNorm
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # 必须 keepdim=True，否则 (x - mean) 的广播维度对不上
        mean = x.mean(dim=-1, keepdim=True)          # (batch_size, seq_len, 1)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x + self.beta


class multihead_attention(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = decoder_config["d_model"]
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        
        batch_size, seq_len, _ = x.size()
        d_k = decoder_config["d_k"]
        num_heads = decoder_config["num_heads"]

        q = self.wq(x).view(batch_size, seq_len, num_heads, d_k).transpose(1, 2) # [batch_size, num_heads, seq_len, d_k]
        k = self.wk(x).view(batch_size, seq_len, num_heads, d_k).transpose(1, 2) # [batch_size, num_heads, seq_len, d_k]
        v = self.wv(x).view(batch_size, seq_len, num_heads, d_k).transpose(1, 2) # [batch_size, num_heads, seq_len, d_k]

        # 构建上三角部分为 True 的矩阵（不包含对角线
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        atten = q @ k.transpose(-1, -2) / math.sqrt(d_k) # [batch_size, num_heads, seq_len, seq_len]
        # masked_fill 的广播机制
        atten = atten.masked_fill(mask, float('-inf')) 
        heads = torch.softmax(atten, dim=-1) @ v
        multihead_atten = heads.transpose(1, 2).reshape(batch_size, seq_len, decoder_config["d_model"])
        multihead_atten = self.wo(multihead_atten) # [batch_size, seq_len, d_model]

        return multihead_atten


class DecoderModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.atten = multihead_attention()
        self.ln1 = LayerNorm(decoder_config["d_model"])
        self.fnn = Fnn()
        self.ln2 = LayerNorm(decoder_config["d_model"])

    def forward(self, x):
        # residual connection + LayerNorm
        x = self.ln1(self.atten(x) + x)
        x = self.ln2(self.fnn(x) + x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 一个 DecoderModule 循环 N 次
        self.layers = nn.ModuleList([DecoderModule() for _ in range(decoder_config['num_layers'])])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class MyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = Embedding(vocab_size, decoder_config["d_model"])
        # 位置编码是常量，注册成 buffer 以便随 model.to(device) 一起移动
        self.register_buffer("pe", positional_encoding().__encoding__())
        self.decoder = Decoder()
        self.porj = LinearProj(vocab_size)

    def forward(self, idx):
        """
        idx: [batch_size, seq_len]
        """
        x = self.emb(idx) + self.pe[:idx.size(1)]
        x = self.decoder(x)
        x = self.porj(x)

        return x


def _detokenize(tokens):
    """把词级 token 拼回文本，简单修正标点前多余空格。"""
    text = " ".join(tokens)
    for p in ".,!?;:)]}":
        text = text.replace(" " + p, p)
    for p in "([{":
        text = text.replace(p + " ", p)
    return text


def generate(model, vocab, id_to_word, start_text, max_new_tokens=50):
    """自回归生成文本。"""
    model.eval()
    device = next(model.parameters()).device
    ids = word_to_id(start_text, vocab)
    if not ids:
        ids = [0]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx = torch.tensor([ids[-training_config["window_size"]:]], dtype=torch.long, device=device)
            logits = model(idx)                                   # [1, seq, vocab]
            probs = torch.softmax(logits[0, -1, :], dim=-1)       # 只取最后一个位置的分布
            next_id = torch.multinomial(probs, num_samples=1).item()
            ids.append(next_id)

    return _detokenize([id_to_word[i] for i in ids])


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    # # prepare data
    # with open('corpus_clean.txt', 'r', encoding='utf-8') as file:
    #     content = file.read()

    # vocab = generate_vocab()
    # i = 0
    # for key, value in vocab.items():
    #     if i < 20:
    #         print(f"{key},{value}")
    #         i = i + 1


    # vocab_size = len(vocab)
    # id_to_word = generate_ids(vocab)

    # ids = word_to_id(content, vocab)
    # inputs, outputs = build_pairs(ids, window_size=training_config["window_size"], step=training_config["sliding_step"])
    # print(f"num training samples: {len(inputs)}")

    # my_dataset = TextDataset(input=inputs, output=outputs)
    # my_dataloader = DataLoader(my_dataset, batch_size=training_config["batch_size"], shuffle=True, drop_last=True)

    # # model, optimizer, and loss function
    # mini_gpt = MyGPT(vocab_size).to(device)
    # optimizer = torch.optim.AdamW(mini_gpt.parameters(), lr=training_config["lr"])
    # loss_fn = nn.CrossEntropyLoss()

    # for epoch in range(training_config["num_epochs"]):
    #     mini_gpt.train()
    #     total_loss = 0
    #     for x, y in my_dataloader:
    #         x, y = x.to(device), y.to(device)
    #         logits = mini_gpt(x)                                          # [batch, seq, vocab]
    #         loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1)) # [batch*seq, vocab]

    #         """
    #         这里为什么不用输入one-hot形式的label？
    #         CrossEntropy里集合了NlLloss, NLLoss先用logsoftmax先计算对数概率(vocab_size)，
    #         然后根据label里的数字, 找到对数概率里相应位置的那个对数概率。所以不用one-hot编码
    #         """
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()

    #     print(f"epoch {epoch + 1}/{training_config['num_epochs']}  avg loss: {total_loss / len(my_dataloader):.4f}")

    # # 保存训练好的模型。除了权重，还必须一起保存词表：
    # # 否则加载后只有一堆 id，无法把 id 还原成文字做生成。
    # torch.save({
    #     "model_state_dict": mini_gpt.state_dict(),   # 模型权重
    #     "vocab": vocab,                              # word -> id
    #     "id_to_word": id_to_word,                    # id -> word
    # }, "mini_gpt.pth")
    # print("model saved to mini_gpt.pth")

    # print("=" * 50)
    # print("generated sample:")
    # print(generate(mini_gpt, vocab, id_to_word, start_text="the", max_new_tokens=50))


    # ------------------------------------推理-------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  
    ckpt = torch.load("mini_gpt.pth", map_location=device, weights_only=False)


    vocab = ckpt["vocab"]
    id_to_word = ckpt["id_to_word"]
    vocab_size = len(vocab)

    mini_gpt = MyGPT(vocab_size).to(device)
    mini_gpt.load_state_dict(ckpt["model_state_dict"])
    mini_gpt.eval()

    print(generate(mini_gpt, vocab, id_to_word, start_text="the", max_new_tokens=50))
