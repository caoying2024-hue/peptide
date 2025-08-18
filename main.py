import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, dataset
from torch.nn import functional as F
from tqdm import tqdm
from vit_model import Transformer


def read_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            data.append(list(line.replace('\n', '')))
    return data


def get_position(seq):
    freq_position_aa = torch.mean(seq, dim=1)
    return freq_position_aa


def get_pair(seq):
    freq_pair_aa = torch.einsum("blij, blkg -> bikjg", seq, seq) / seq.shape[1]
    return freq_pair_aa

def calculate_coupling(freq_position_aa, freq_pair_aa, o=9, NA=20):
    device = freq_position_aa.device
    mean_freq = torch.tensor(
        [0.025, 0.023, 0.042, 0.053, 0.089, 0.063, 0.013, 0.033, 0.073, 0.072, 0.056, 0.073, 0.043, 0.04, 0.05, 0.061,
         0.023, 0.052, 0.064, 0.052], device=device)

    sum_freq_coupling = torch.zeros((o, o), device=device)

    phi = torch.where((freq_position_aa == 0) | (freq_position_aa == 1),
                      torch.tensor(0.0, device=device),
                      torch.log((freq_position_aa * (1 - mean_freq)) / ((1 - freq_position_aa) * mean_freq)))

    for m in range(o):
        for r in range(o):
            freq_coupling = freq_pair_aa[m, r] - freq_position_aa[m].unsqueeze(1) * freq_position_aa[r].unsqueeze(0)
            weight_freq_coupling = phi[m].unsqueeze(1) * phi[r].unsqueeze(0) * freq_coupling
            sum_freq_coupling[m, r] = torch.sum(weight_freq_coupling * weight_freq_coupling)
    scale = torch.ones(9, 9, device=device)
    scale[range(9), range(9)] = 1
    sum_freq_coupling = sum_freq_coupling * scale
    sum_freq_coupling=torch.sqrt(sum_freq_coupling)
    return sum_freq_coupling

class SeqDataset(dataset.Dataset):
    def __init__(self, file_paths, mode):
        super(SeqDataset, self).__init__()
        self.file_paths = file_paths
        self.residue_simple = ['C', 'M', 'F', 'I', 'L', 'V', 'W', 'Y', 'A', 'G', 'T', 'S', 'N', 'Q', 'D', 'E', 'H', 'R', 'K','P']
        self.mean_freq = [0.025, 0.023, 0.042, 0.053, 0.089, 0.063, 0.013, 0.033, 0.073, 0.072, 0.056, 0.073, 0.043, 0.04,
                     0.05, 0.061, 0.023, 0.052, 0.064, 0.052]
        self.residue_simple_num = {letter: idx for idx, letter in enumerate(self.residue_simple)}
        self.length = 5000
        self.mode = mode

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = read_file(file_path)
        index_m = np.vectorize(self.residue_simple_num.get)(data)
        final = np.eye(20)[index_m]
        input = torch.from_numpy(final).float()
        if self.mode == 'train':
            return input
        if self.mode == 'test':
            return input, os.path.basename(file_path).split('.')[0]


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse1 = nn.MSELoss()
        self.mse2 = nn.MSELoss()

    def forward(self, output, freq_position_aa, freq_pair_aa):
        pos = get_position(output)
        pair = get_pair(output)
        loss1 = self.mse1(pos, freq_position_aa)
        loss2 = self.mse2(pair, freq_pair_aa)
        return loss1 + 65*loss2
        # return loss2


def save_frequency(frequency, folder):
    frequency = frequency.numpy()
    path = os.path.join(folder, 'frequency.txt')
    with open(path, 'w') as f:
        for seq in frequency:
            for AA in seq:
                for freq in AA:
                    f.write(str(freq) + ',')
                f.write('\n')
            f.write('\n\n')


def save_seqence(frequency, folder):
    path = os.path.join(folder, 'seqence.txt')
    res = torch.zeros(frequency.shape[0], frequency.shape[1], 1, dtype=float)
    for i in range(frequency.shape[0]):
        j = 0
        while j < frequency.shape[1]:
            sam = torch.multinomial(frequency[i, j, :], num_samples=1, replacement=True)
            res[i, j, :] = sam
            j = j + 1

    res = res.squeeze(dim=-1).numpy().tolist()
    residue_simple = ['C', 'M', 'F', 'I', 'L', 'V', 'W', 'Y', 'A', 'G', 'T', 'S', 'N', 'Q', 'D', 'E', 'H', 'R', 'K',
                      'P']
    residue_simple_num = {letter: idx for idx, letter in enumerate(residue_simple)}
    g = {idx: letter for letter, idx in residue_simple_num.items()}
    residue = []
    for seq in res:
        a = []
        for i in seq:
            a.append(g[i])
        residue.append("".join(a))
    with open(path, 'w') as f:
        f.write("\n".join(residue))


if __name__ == '__main__':
    mode = "train"   #训练模块
    # mode = "test"  #测试模块
    train_folder = "./train_data"
    train_path = [os.path.join(train_folder, i) for i in os.listdir(train_folder)]
    test_folder = "./test_data"
    test_path = [os.path.join(test_folder, i) for i in os.listdir(test_folder)]
    model_path = "model.pth17.2"
    output_folder = "output17.2"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    model = Transformer(
        img_size=[90, 180],
        in_c=1,
        patch_size=[10, 10],
        embed_dim=180,
        depth=12,
        num_heads=12,
        representation_size=1800,
        num_classes=4476
    )
    model.cuda()
    if mode == "train":
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, weights_only=True))
        model.train()
        loss_fn = Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        train_dataloader = DataLoader(
            SeqDataset(train_path, mode),
            batch_size=1)
        for epoch in range(2000):
            train_loss = 0
            for input in train_dataloader:
                optimizer.zero_grad()
                freq_position_aa = get_position(input).cuda()
                freq_pair_aa = get_pair(input).cuda()
                output = model(freq_position_aa, freq_pair_aa)
                loss = loss_fn(output, freq_position_aa, freq_pair_aa)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss = train_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1} , Train Loss: {train_loss}")
            # print(train_loss)
        torch.save(model.state_dict(), model_path)
    if mode == "test":
        assert os.path.exists(model_path), f"Model path {model_path} does not exist!"
        model.load_state_dict(torch.load(model_path, weights_only=True))
        test_dataloader = DataLoader(
            SeqDataset(test_path, mode),
            batch_size=1)
        model.eval()
        with torch.no_grad():
            for input, file_path in test_dataloader:
                freq_position_aa = get_position(input).cuda()
                freq_pair_aa = get_pair(input).cuda()
                output = model(freq_position_aa, freq_pair_aa)
                output = output.cpu()
                for i in range(len(output)):
                    frequency = output[i]
                    folder = os.path.join(output_folder, file_path[i])
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    save_frequency(frequency, folder)
                    save_seqence(frequency, folder)
