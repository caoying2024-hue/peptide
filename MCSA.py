import torch
import math
import numpy as np
import random

# 使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NA=20
residue_sample = ['C', 'M', 'F', 'I', 'L', 'V', 'W', 'Y', 'A', 'G', 'T', 'S', 'N', 'Q', 'D', 'E', 'H', 'R', 'K', 'P']
residue_simple_num = {letter: idx for idx, letter in enumerate(residue_sample)}


def get_input1(data, residue_simple_num=residue_simple_num):
    index_m = np.vectorize(residue_simple_num.get)(data)
    final = np.eye(20)[index_m]
    input = torch.from_numpy(final).float()
    return input


def get_input2(data, residue_simple_num=residue_simple_num):
    index_m = np.vectorize(residue_simple_num.get)(data)
    index_m = torch.from_numpy(index_m)
    return index_m


# 从文件读取氨基酸平均频率
# 读取氨基酸平均频率文件
def read_aa_mean_frequency(filename):
    residue = []         # 用于存储氨基酸全称
    residue_simple = []  # 用于存储氨基酸简化符号
    mean_freq = []       # 用于存储氨基酸平均频率

    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        full_name = parts[0]           # 氨基酸全称
        simple_symbol = parts[1]       # 简化符号
        frequency = float(parts[2])    # 平均频率

        residue.append(full_name)
        residue_simple.append(simple_symbol)
        mean_freq.append(frequency)

    return residue, residue_simple, mean_freq


# 读取MSA列数据
def read_msa_columns(msa_file):
    """
    从 MSA 文件中提取每列氨基酸数据。  
    :param msa_file: MSA 文件路径
    :return: 列列表，每列是一个包含氨基酸的列表
    """
    columns = []
    with open(msa_file, 'r') as file:
        msa = [line.strip() for line in file.readlines() if line.strip()]  # 读取 MSA 并移除空行

    o = len(msa[0])  # 假设所有序列长度相同
    for col_idx in range(o):
        column = [seq[col_idx] for seq in msa]  # 提取每列的氨基酸
        columns.append(column)   
    return columns


def tubian(current_family, frequencies, aphla=0.1):
    # current_family shape  2000*9
    current_family = np.array(current_family).T
    idx = int(np.random.random() * 9)
    fam = random.choices(list(frequencies[idx].keys()), weights = list(frequencies[idx].values()), k = 1)[0]
    current_family[idx] = fam
    all = np.array(current_family).T.tolist()
    return all


def tubian2(current_family):
    current_family = np.array(current_family).T
    idx = int(np.random.random() * 9)
    residue_sample = ['C', 'M', 'F', 'I', 'L', 'V', 'W', 'Y', 'A', 'G', 'T', 'S', 'N', 'Q', 'D', 'E', 'H', 'R', 'K', 'P']
    fam = np.random.choice(residue_sample, size=current_family.shape[0])
    current_family[idx] = fam
    all = np.array(current_family).T.tolist()
    return all


def calculate_column_frequencies(columns):
    """
    计算每列中氨基酸的频率分布。
    :param columns: MSA 的列数据
    :return: 每列的频率分布列表
    """
    frequencies = []
    for column in columns:
        unique, counts = np.unique(column, return_counts=True)
        total = len(column)
        freq_dict = {aa: count / total for aa, count in zip(unique, counts)}
        frequencies.append(freq_dict)
    return frequencies


def shuffle_sequence_from_msa_columns(columns, column_frequencies):
    """
    从 MSA 列数据中生成随机序列（基于列内频率分布）。
    :param columns: MSA 的列数据，每列是一个包含氨基酸的列表
    :param column_frequencies: 每列的频率分布列表
    :return: 随机生成的序列
    """
    sequence = []
    for column, freq_dict in zip(columns, column_frequencies):
        amino_acids = list(freq_dict.keys())
        probabilities = list(freq_dict.values())
        aa = np.random.choice(amino_acids, p=probabilities)  # 按概率抽样
        sequence.append(aa)
    return sequence

import numpy as np

def shuffle_sequence_from_msa_columns2(columns):
    """
    从 MSA 列数据中生成随机序列（完全随机选择，不按频率分布）。
    :param columns: MSA 的列数据，每列是一个包含氨基酸的列表
    :return: 随机生成的序列
    """
    sequence = []
    for column in columns:
        column = list(set(column))
        aa = np.random.choice(column)  # 完全随机选择一个氨基酸
        sequence.append(aa)
    return sequence

def perturb_family(family, columns, column_frequencies, perturbation_count=9):
    """
    对多个序列进行局部微调，基于每列的氨基酸频率分布。
    :param family: 要微调的序列家族（一个包含多个序列的列表）
    :param columns: MSA 的列数据，每列包含所有序列中相同位置的氨基酸
    :param column_frequencies: 每列氨基酸频率字典的列表
    :param perturbation_count: 控制微调次数的参数，表示每条序列会微调多少个位置
    :return: 微调后的序列家族
    """
    perturbed_family = []  # 保存微调后的家族序列

    for seq in family:
        # 对每条序列进行微调
        new_seq = seq[:]  # 创建序列的副本，以避免修改原始序列
        num_positions = len(seq)  # 序列的长度
        
        # 随机选择一些位置进行微调
        positions_to_perturb = random.sample(range(num_positions), perturbation_count)
        
        for i in positions_to_perturb:
            # 获取当前列的频率分布
            freq_dict = column_frequencies[i]
            possible_residues = list(freq_dict.keys())  # 获取该列的所有氨基酸
            frequencies = [freq_dict[aa] for aa in possible_residues]  # 获取对应氨基酸的频率

            # 按照频率进行选择，新的氨基酸可能会不同
            new_residue = random.choices(possible_residues, weights=frequencies)[0]
            new_seq[i] = new_residue  # 更新该位置的氨基酸

        perturbed_family.append(new_seq)  # 将微调后的序列添加到家族中

    return perturbed_family





def calculate_coupling(sequence_id, flag=False, o=9, NA=20):
    # sequence_id = torch.tensor(sequence_id)
    sequence_id = get_input1(sequence_id)
    device = sequence_id.device
    residue_simple = torch.arange(NA, device=device)
    mean_freq = torch.tensor(
        [0.025, 0.023, 0.042, 0.053, 0.089, 0.063, 0.013, 0.033, 0.073, 0.072, 0.056, 0.073, 0.043, 0.04, 0.05, 0.061,
         0.023, 0.052, 0.064, 0.052], device=device)

    # Initialize tensors
    hist_position_aa = torch.zeros((o, NA), device=device)
    freq_position_aa = torch.zeros((o, NA), device=device)
    freq_pair_aa = torch.zeros((o, o, NA, NA), device=device)
    hist_pair_aa = torch.zeros((o, o, NA, NA), device=device)
    phi = torch.zeros((o, NA), device=device)
    sum_freq_coupling = torch.zeros((o, o), device=device)
    if flag == True:
        # 统计每个位置上氨基酸的出现次数
        for m in range(o):
            hist_position_aa[m] = torch.bincount(sequence_id[:, m], minlength=NA)

        # 计算每个位置上氨基酸的频率
        freq_position_aa = hist_position_aa / len(sequence_id)

        # 统计每个位置两两氨基酸对的出现次数
        for m in range(o):
            for r in range(o):
                for p in range(NA):
                    for q in range(NA):
                        hist_pair_aa[m, r, p, q] = torch.sum((sequence_id[:, m] == p) & (sequence_id[:, r] == q))

        # 计算氨基酸对的频率
        freq_pair_aa = hist_pair_aa / len(sequence_id)
    else:
        # print(sequence_id)
        freq_position_aa = torch.sum(sequence_id, dim=0) / len(sequence_id)
        freq_pair_aa = torch.einsum('bik,bjl->ijkl', sequence_id, sequence_id) / len(sequence_id)
    # 计算phi值
    phi = torch.where((freq_position_aa == 0) | (freq_position_aa == 1),
                      torch.tensor(0.0, device=device),
                      torch.log((freq_position_aa * (1 - mean_freq)) / ((1 - freq_position_aa) * mean_freq)))

    # 计算保守性耦合分数
    for m in range(o):
        for r in range(o):
            freq_coupling = freq_pair_aa[m, r] - freq_position_aa[m].unsqueeze(1) * freq_position_aa[r].unsqueeze(0)
            weight_freq_coupling = phi[m].unsqueeze(1) * phi[r].unsqueeze(0) * freq_coupling
            sum_freq_coupling[m, r] = torch.sum(weight_freq_coupling * weight_freq_coupling)
    # print(sum_freq_coupling)
    scale = torch.ones(9, 9, device=device)
    scale[range(9), range(9)] = 1
    sum_freq_coupling = sum_freq_coupling * scale
    # print(sum_freq_coupling)
    sum_freq_coupling = torch.sqrt(sum_freq_coupling)
    return sum_freq_coupling


# 读取目标耦合矩阵

def load_target_coupling_matrix(file_path, matrix_size=9):
    """
    从文件中读取目标耦合矩阵并生成指定大小的张量。

    参数:
    - file_path (str): 文件路径，文件格式为 "row column value"。
    - matrix_size (int): 目标矩阵的大小（默认为 9，即 9x9）。

    返回:
    - torch.Tensor: 目标耦合矩阵 (matrix_size x matrix_size)。
    """
    target_coupling_matrix = torch.zeros((matrix_size, matrix_size))  # 初始化矩阵
    with open(file_path, "r") as f:
        for line in f:
            i, j, value = line.strip().split()  # 拆分行数据
            i, j = int(i) - 1, int(j) - 1       # 转为零索引
            target_coupling_matrix[i, j] = float(value)  # 填入对应的值
    return target_coupling_matrix


def perturb_family2(current_family, columns, column_frequencies, perturbation_rate=0.1):
    """
    对蛋白质家族进行局部扰动，随机选择若干列并根据频率分布对其进行打乱。
    :param current_family: 当前家族的序列列表，每个序列是一个氨基酸列表
    :param columns: MSA 的列数据，每列包含所有序列中相同位置的氨基酸
    :param column_frequencies: 每列的氨基酸频率字典
    :param perturbation_rate: 扰动比例（决定扰动的列数，范围为 0-1）
    :return: 新的家族序列
    """
    # 拷贝当前家族，避免直接修改输入
    new_family = [seq[:] for seq in current_family]
    
    num_columns = len(columns)
    num_perturbed_columns = max(1, int(perturbation_rate * num_columns))  # 至少扰动 1 列

    # 随机选择若干列进行扰动
    perturbed_columns = np.random.choice(num_columns, size=num_perturbed_columns, replace=False)
    #perturbed_columns = [int(np.random.random() * 9 / 10)]  #只突变一列

    for column_index in perturbed_columns:
        # 获取该列的频率分布
        freq_dict = column_frequencies[column_index]
        amino_acids = list(freq_dict.keys())
        probabilities = list(freq_dict.values())

        # 根据频率分布生成新列
        shuffled_column = np.random.choice(amino_acids, size=len(current_family), p=probabilities).tolist()

        # 更新每条序列的对应列
        for i in range(len(new_family)):
            new_family[i][column_index] = shuffled_column[i]

    return new_family



def perturb_family3(current_family, columns, perturbation_rate=0.1, all_amino_acids=None):
    """
    对蛋白质家族进行局部扰动，随机选择若干列并对每列的氨基酸进行随机突变。
    :param current_family: 当前家族的序列列表，每个序列是一个氨基酸列表
    :param columns: MSA 的列数据，每列包含所有序列中相同位置的氨基酸
    :param perturbation_rate: 扰动比例（决定扰动的列数，范围为 0-1）
    :param all_amino_acids: 所有可能的氨基酸列表（如果为 None，则仅从当前列中提取氨基酸）
    :return: 新的家族序列
    """
    # 拷贝当前家族，避免直接修改输入
    new_family = [seq[:] for seq in current_family]
    
    num_columns = len(columns)
    num_perturbed_columns = max(1, int(perturbation_rate * num_columns))  # 至少扰动 1 列

    # 随机选择若干列进行扰动
    perturbed_columns = np.random.choice(num_columns, size=num_perturbed_columns, replace=False)#至少突变一列
    #perturbed_columns = int(np.random.random() * 9 / 10)  #只突变一列

    for column_index in perturbed_columns:
        # 获取该列的所有可能氨基酸
        if all_amino_acids is None:
            amino_acids = list(set([seq[column_index] for seq in new_family]))  # 从当前列中提取
        else:
            amino_acids = all_amino_acids  # 使用所有可能的氨基酸

        # 对该列的每个氨基酸进行随机突变
        for i in range(len(new_family)):
            new_amino_acid = np.random.choice(amino_acids)
            new_family[i][column_index] = new_amino_acid

    return new_family

def   perturb_sequence(sequence, columns, column_frequencies, perturbation_strength=0.08):
    """
    对给定的序列进行局部微调，基于每列的氨基酸频率分布。
    :param sequence: 要微调的氨基酸序列
    :param columns: MSA 的列数据，每列包含所有序列中相同位置的氨基酸
    :param column_frequencies: 每列氨基酸频率字典的列表
    :param perturbation_strength: 控制微调强度的参数，值越大，变动越大
    :return: 微调后的序列
    """
    new_sequence = sequence[:]  # 创建序列的副本，以避免修改原始序列
    num_positions = len(sequence)  # 序列的长度

    for i in range(num_positions):
        # 获取当前列的频率分布
        freq_dict = column_frequencies[i]
        possible_residues = list(freq_dict.keys())  # 获取该列的所有氨基酸
        frequencies = [freq_dict[aa] for aa in possible_residues]  # 获取对应氨基酸的频率

        # 选择一个新的氨基酸
        if random.random() < perturbation_strength:  # 根据 perturbation_strength 决定是否进行变动
            # 按照频率进行选择，新的氨基酸可能会不同
            new_residue = random.choices(possible_residues, weights=frequencies)[0]
        else:
            # 若不进行变动，保持原序列的氨基酸
            new_residue = sequence[i]

        new_sequence[i] = new_residue  # 更新该位置的氨基酸

    return new_sequence    


def perturb_sequence2(sequence, columns, perturbation_strength=0.08):
    """
    对给定的序列进行局部微调，完全随机选择新的氨基酸（不按频率分布）。
    :param sequence: 要微调的氨基酸序列
    :param columns: MSA 的列数据，每列包含所有序列中相同位置的氨基酸
    :param perturbation_strength: 控制微调强度的参数，值越大，变动越大
    :return: 微调后的序列
    """
    new_sequence = sequence[:]  # 创建序列的副本，以避免修改原始序列
    num_positions = len(sequence)  # 序列的长度

    for i in range(num_positions):
        # 获取当前列的所有氨基酸
        # possible_residues = columns[i]
        possible_residues = list(set(columns[i]))

        # 选择一个新的氨基酸
        if random.random() < perturbation_strength:  # 根据 perturbation_strength 决定是否进行变动
            # 完全随机选择一个氨基酸
            new_residue = random.choice(possible_residues)
        else:
            # 若不进行变动，保持原序列的氨基酸
            new_residue = sequence[i]

        new_sequence[i] = new_residue  # 更新该位置的氨基酸

    return new_sequence

def read_IC(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            data.append(list(line.replace("\n", "")))
    return data



# 模拟退火算法更新（使用 GPU 加速）
def monte_carlo_family_optimization(target_coupling_matrix, residue_simple, columns, column_frequencies,
                                    mean_freq, o, NA, 
                                    max_iterations,initial_temp, cooling_rate,num_sequences=410, steps_per_temp=1000):
    """
    优化蛋白质序列家族，使其耦合矩阵与目标耦合矩阵的差值最小化。
    每个温度下进行多次搜索，直到找到平衡状态才降温。
    """
    # 初始化序列家族
    # current_family = [   #current_family 是通过调用 shuffle_sequence_from_msa_columns 函数num_sequences次来初始化的，每次调用都会生成一个新的氨基酸序列。
    #     shuffle_sequence_from_msa_columns(columns, column_frequencies) for _ in range(num_sequences)
    # ]
    # 初始化序列家族
    current_family = [shuffle_sequence_from_msa_columns2(columns) for _ in range(num_sequences)]
    # filename = "HLA-DRB5_01_01/ic.txt" #01_03_02_01(alig).txt
    # current_family = read_IC(filename)

    # 计算初始家族的耦合矩阵和目标函数值
    current_coupling = calculate_coupling(current_family)

    target_coupling_matrix = target_coupling_matrix.to(device)

    # # 提取对角线元素的差值的绝对值
    diagonal_diff = torch.sqrt(torch.pow(torch.diagonal(current_coupling - target_coupling_matrix), 2)) #torch.sqrt(torch.pow(x, 2)) 实现，等价于 torch.abs(x)。
    # 计算对角线误差（绝对值）
    diagonal_error = torch.sum(diagonal_diff)

    # 提取非对角线元素的差值的绝对值
    off_diagonal_diff = (current_coupling - target_coupling_matrix) ** 2
    off_diagonal_diff = off_diagonal_diff.fill_diagonal_(0)  # 去掉对角线部分
    # 计算非对角线误差（绝对值）
    off_diagonal_error = torch.sum(off_diagonal_diff)

    # 组合误差
    alpha = 0.1  # 对角线部分的权重
    current_score = alpha * diagonal_error.item() + (1 - alpha) * off_diagonal_error.item()
    # current_score = diagonal_error.item() + off_diagonal_error.item()

     # 设置初始参数
    temp = initial_temp
    best_family = current_family[:]
    best_score = current_score

    # 记录收敛数据
    convergence_data = [] #new score
    terminate = False  # 标志变量

    for iteration in range(max_iterations):
        temp_stable = False
        steps_at_current_temp = 0
         # 初始化计数器
        delta_neg_count = 0
        accepted_count = 0
        rejected_count = 0
        print(f"Iteration {iteration}/{max_iterations}, Current Score: {current_score:.4f}, Best Score: {best_score:.4f}")
    
        while not temp_stable and steps_at_current_temp < steps_per_temp:#steps_per_temp 是控制每个温度下优化次数
            
            # new_family = tubian(current_family, column_frequencies)   
            #new_family=[perturb_sequence2(sequence,columns,perturbation_strength=0.08) for sequence in current_family]     
            #new_family = perturb_family2(current_family, columns, column_frequencies, perturbation_rate=0.1) 

            new_family= perturb_family3(current_family, columns, perturbation_rate=0.1, all_amino_acids=None) #无频率

            # if temp>0.01:
            #     new_family=[perturb_sequence(sequence,columns,column_frequencies,perturbation_strength=0.08) for sequence in current_family]
            # else:
            #     new_family=[perturb_sequence(sequence,columns,column_frequencies,perturbation_strength=0.02) for sequence in current_family]
            #new_family=[perturb_sequence(sequence,columns,column_frequencies,perturbation_strength=0.08) for sequence in current_family]
            new_coupling = calculate_coupling(new_family)
        
            # # 提取对角线元素的差值的绝对值
            diagonal_diff = torch.sqrt(torch.pow(torch.diagonal(new_coupling - target_coupling_matrix), 2)) #torch.sqrt(torch.pow(x, 2)) 实现，等价于 torch.abs(x)。
            # 计算对角线误差（绝对值）
            diagonal_error = torch.sum(diagonal_diff)

            # 提取非对角线元素的差值的绝对值
            off_diagonal_diff = (new_coupling - target_coupling_matrix) ** 2
            off_diagonal_diff = off_diagonal_diff.fill_diagonal_(0)  # 去掉对角线部分
            # 计算非对角线误差（绝对值）
            off_diagonal_error = torch.sum(off_diagonal_diff)

            # 组合误差
            alpha = 0.1  # 对角线部分的权重
            new_score = alpha * diagonal_error.item() + (1 - alpha) * off_diagonal_error.item()
            # 计算接受概率
            delta = new_score - current_score  #若delta<0,此时acceptance_probability=1无条件接收，反之，概率接收
            acceptance_probability = math.exp(-delta / temp) if delta > 0 else 1
            # 计算接受概率
            if delta > 0:
                acceptance_probability = math.exp(-delta / temp)
                print(f"  Delta > 0, Acceptance Probability: {acceptance_probability:.4f}")
            else:
                acceptance_probability = 1
                delta_neg_count += 1  # 计数 delta < 0 的情况,新解优于当前解，满足无条件接受规则,delta_neg_count 是 accepted_count 的子集。因为每次 delta < 0，必然会导致接受，因此这些已经包含在 accepted_count 中。
                            
            # 判断是否接受新家族
            if random.random()  < acceptance_probability:  #如果 delta < 0，此时acceptance_probability=1，random.random() 返回的是一个 [0, 1) 之间的随机浮动值，因为随机数 random.random() 总是小于 1，所以会接受新家族,所以delta<0，random.random()  < acceptance_probability成立。如果 delta >= 0，仍然有一定的机会接受新家族，概率是 math.exp(-delta / temp)。随着温度逐渐降低，接受新家族的概率会逐渐减小。
                accepted_count += 1  # 计数被接收的次数
                current_family = new_family
                current_score = new_score
                
                # 更新最佳解
                if new_score < best_score:
                    best_family = new_family
                    best_score = new_score
                print(f"  Accepted new family (better solution or by chance).")
            else:
            # 未接受新家族
                rejected_count += 1  # 计数被拒绝的次数
                print(f"  Rejected new family (did not pass acceptance probability).")
             # 更新步数和稳定性检测
            steps_at_current_temp += 1
            print(f" Current Score: {current_score}, New Score: {new_score}, Acceptance Probability: {acceptance_probability:.4f},delta:{delta:.4f}")

        # if temp>0.01:
        #    temp *= cooling_rate1
        # else:
        #     temp *= cooling_rate2
        temp *= cooling_rate
    
        
        # 记录当前得分
        convergence_data.append(current_score)
        # 打印迭代信息
        print(f"temp: {temp}")
        print(f"Iteration {iteration + 1}/{max_iterations}, Current Score: {current_score:.4f}, Best Score: {best_score:.4f},New Score: {new_score:.4f}, Delta: {delta:.4f}")
        print(f"Temperature {temp}: Delta < 0 count: {delta_neg_count}, Accepted count: {accepted_count}, Rejected count: {rejected_count}")

    return best_family, best_score, convergence_data

import os
# 主函数
if __name__ == "__main__":

    # Define the folder name
    folder_name = "HLA-DRB5_01_01/output_files(cc)"# 31是0.99，200，700,  30是0.97，120，1500.初始温度都是0.0009，alpha=0.1,  33是0.97,120,1500，初始温度是0.0025
    
    # 初始化参数s
    max_iterations =2000
    initial_temp =15
    cooling_rate= 0.99
    # cooling_rate1= 0.99
    # cooling_rate2=0.95
    num_sequences = 410
    alpha = 0.1
    steps_per_temp=1200
    perturb_family3=perturb_family3
    # perturbation_strength=0.08
    # tubian=tubian
    

    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # msa_file = "11-19(alig).txt"
    msa_file = "HLA-DRB5_01_01/bind.txt"
    # target_coupling_file = "coupling_complex(2).txt"
    freq_file = "AAMeanFrequency.dat"

    # Define the file paths
    parameters_file = os.path.join(folder_name, "parameters5.txt")
    output_file = os.path.join(folder_name, "optimized_family5.txt")
    convergence_file = os.path.join(folder_name, "convergence_data_family5.txt")


    o = 9
    input = read_IC(msa_file)
    columns = read_msa_columns(msa_file)
    # print(len(columns))
    # print(len(columns[0]))
    column_frequencies = calculate_column_frequencies(columns)

    # target_coupling_matrix = load_target_coupling_matrix(target_coupling_file)
    target_coupling_matrix = calculate_coupling(input)
    # print("ok")
    residue, residue_simple, mean_freq = read_aa_mean_frequency(freq_file)
    print(residue_simple)
    #执行蒙特卡罗优化
    optimized_family, best_score, convergence_data = monte_carlo_family_optimization(
        target_coupling_matrix, residue_simple, columns, column_frequencies, mean_freq, o, NA,
        max_iterations=max_iterations, initial_temp=initial_temp, cooling_rate=cooling_rate,num_sequences=num_sequences, steps_per_temp=steps_per_temp
    )

    # 保存优化结果
    with open(output_file, "w") as file:
        file.write("Optimized Family and Best Function Value\n")
        for i, seq in enumerate(optimized_family):
            sequence_str = "".join(seq)
            file.write(f"Sequence {i + 1}: {sequence_str}\n")
        file.write(f"\nBest Function Value: {best_score:.4f}\n")

    # 保存收敛数据
    with open(convergence_file, "w") as file:
        file.write("Iteration\tObjective Function Value\n")
        for iteration, score in enumerate(convergence_data):
            file.write(f"{iteration + 1}\t{score:.4f}\n")

    with open(parameters_file, "w") as file:
        file.write(f"max_iterations = {max_iterations}\n")
        file.write(f"initial_temp = {initial_temp}\n")
        file.write(f"cooling_rate= {cooling_rate}\n")
        # file.write(f"cooling_rate1 = {cooling_rate1}\n")
        # file.write(f"cooling_rate2= {cooling_rate2}\n")
        file.write(f"num_sequences = {num_sequences}\n")
        file.write(f"steps_per_temp= {steps_per_temp}\n")
        file.write(f"alpha = {alpha}\n")
        # file.write(f"perturbation_strength=0.08\n")
        file.write(f"perturb_family3=0.1\n")
        # file.write(f"tubian= {tubian}\n")

    print(f"Files have been saved in the '{folder_name}' folder.")