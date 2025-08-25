import torch
import math
import numpy as np
import random


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


# The amino acid average frequency from the file
def read_aa_mean_frequency(filename):
    residue = []
    residue_simple = []
    mean_freq = []

    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        full_name = parts[0]
        simple_symbol = parts[1]
        frequency = float(parts[2])

        residue.append(full_name)
        residue_simple.append(simple_symbol)
        mean_freq.append(frequency)

    return residue, residue_simple, mean_freq


# MSA data
def read_msa_columns(msa_file):
    columns = []
    with open(msa_file, 'r') as file:
        msa = [line.strip() for line in file.readlines() if line.strip()]  

    o = len(msa[0])  
    for col_idx in range(o):
        column = [seq[col_idx] for seq in msa] 
        columns.append(column)   
    return columns


#Calculate the frequency distribution of amino acids in each column
def calculate_column_frequencies(columns):
    frequencies = []
    for column in columns:
        unique, counts = np.unique(column, return_counts=True)
        total = len(column)
        freq_dict = {aa: count / total for aa, count in zip(unique, counts)}
        frequencies.append(freq_dict)
    return frequencies

#Generate random sequences from MSA data (based on frequency distribution within columns)
def shuffle_columns(columns, column_frequencies):
    sequence = []
    for column, freq_dict in zip(columns, column_frequencies):
        amino_acids = list(freq_dict.keys())
        probabilities = list(freq_dict.values())
        aa = np.random.choice(amino_acids, p=probabilities)  
        sequence.append(aa)
    return sequence

#Generate random sequences from MSA data (completely randomly selected, not distributed by frequency)
def shuffle_columns2(columns):
    sequence = []
    for column in columns:
        column = list(set(column))
        aa = np.random.choice(column)  
        sequence.append(aa)
    return sequence


def calculate_coupling(sequence_id, flag=False, o=9, NA=20):
    sequence_id = get_input1(sequence_id)
    device = sequence_id.device
    residue_simple = torch.arange(NA, device=device)
    mean_freq = torch.tensor(
        [0.025, 0.023, 0.042, 0.053, 0.089, 0.063, 0.013, 0.033, 0.073, 0.072, 0.056, 0.073, 0.043, 0.04, 0.05, 0.061,
         0.023, 0.052, 0.064, 0.052], device=device)

    hist_position_aa = torch.zeros((o, NA), device=device)
    freq_position_aa = torch.zeros((o, NA), device=device)
    freq_pair_aa = torch.zeros((o, o, NA, NA), device=device)
    hist_pair_aa = torch.zeros((o, o, NA, NA), device=device)
    phi = torch.zeros((o, NA), device=device)
    sum_freq_coupling = torch.zeros((o, o), device=device)
    if flag == True:
        for m in range(o):
            hist_position_aa[m] = torch.bincount(sequence_id[:, m], minlength=NA)
        freq_position_aa = hist_position_aa / len(sequence_id)

        for m in range(o):
            for r in range(o):
                for p in range(NA):
                    for q in range(NA):
                        hist_pair_aa[m, r, p, q] = torch.sum((sequence_id[:, m] == p) & (sequence_id[:, r] == q))
        freq_pair_aa = hist_pair_aa / len(sequence_id)
    else:
        freq_position_aa = torch.sum(sequence_id, dim=0) / len(sequence_id)
        freq_pair_aa = torch.einsum('bik,bjl->ijkl', sequence_id, sequence_id) / len(sequence_id)
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
    sum_freq_coupling = torch.sqrt(sum_freq_coupling)
    return sum_freq_coupling


#Mutations are made to a given sequence, based on the amino acid frequency distribution of each column.
def   perturb_sequence(sequence, columns, column_frequencies, perturbation_strength=0.08):
    new_sequence = sequence[:]  
    num_positions = len(sequence) 

    for i in range(num_positions):
        freq_dict = column_frequencies[i]
        possible_residues = list(freq_dict.keys())
        frequencies = [freq_dict[aa] for aa in possible_residues]

        if random.random() < perturbation_strength:
            new_residue = random.choices(possible_residues, weights=frequencies)[0]
        else:
            new_residue = sequence[i]
            
        new_sequence[i] = new_residue

    return new_sequence    

#Mutation of a given sequence to completely randomly select new amino acids (not distributed by frequency)
def perturb_sequence2(sequence, columns, perturbation_strength=0.08):
    new_sequence = sequence[:]  
    num_positions = len(sequence)  

    for i in range(num_positions):
        possible_residues = list(set(columns[i]))
        if random.random() < perturbation_strength: 
            new_residue = random.choice(possible_residues)
        else:
            new_residue = sequence[i]

        new_sequence[i] = new_residue 

    return new_sequence

def read_IC(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            data.append(list(line.replace("\n", "")))
    return data



def monte_carlo_family_optimization(target_coupling_matrix, residue_simple, columns, column_frequencies,
                                    mean_freq, o, NA, 
                                    max_iterations,initial_temp, cooling_rate,num_sequences=410, steps_per_temp=1000):


    current_family = [ shuffle_columns(columns, column_frequencies) for _ in range(num_sequences)]

    # current_family = [shuffle_columns2(columns) for _ in range(num_sequences)]
                                        
    # filename = "HLA-DRB5_01_01/ic.txt"
    # current_family = read_IC(filename)

    current_coupling = calculate_coupling(current_family)
    target_coupling_matrix = target_coupling_matrix.to(device)


    diagonal_diff = torch.sqrt(torch.pow(torch.diagonal(current_coupling - target_coupling_matrix), 2))
    diagonal_error = torch.sum(diagonal_diff)
    off_diagonal_diff = (current_coupling - target_coupling_matrix) ** 2
    off_diagonal_diff = off_diagonal_diff.fill_diagonal_(0) 
    off_diagonal_error = torch.sum(off_diagonal_diff)

    #The combination error of diagonal and non-diagonal lines
    alpha = 0.1 
    current_score = alpha * diagonal_error.item() + (1 - alpha) * off_diagonal_error.item()
    temp = initial_temp
    best_family = current_family[:]
    best_score = current_score


    convergence_data = []

    for iteration in range(max_iterations):
        temp_stable = False
        steps_at_current_temp = 0
        delta_neg_count = 0
        accepted_count = 0
        rejected_count = 0
        print(f"Iteration {iteration}/{max_iterations}, Current Score: {current_score:.4f}, Best Score: {best_score:.4f}")
    
        while not temp_stable and steps_at_current_temp < steps_per_temp:

            # new_family=[perturb_sequence2(sequence,columns,perturbation_strength=0.08) for sequence in current_family]
            
            new_family=[perturb_sequence(sequence,columns,column_frequencies,perturbation_strength=0.08) for sequence in current_family]
            new_coupling = calculate_coupling(new_family)
        
            diagonal_diff = torch.sqrt(torch.pow(torch.diagonal(new_coupling - target_coupling_matrix), 2))
            diagonal_error = torch.sum(diagonal_diff)
            off_diagonal_diff = (new_coupling - target_coupling_matrix) ** 2
            off_diagonal_diff = off_diagonal_diff.fill_diagonal_(0)
            off_diagonal_error = torch.sum(off_diagonal_diff)
            alpha = 0.1  
            new_score = alpha * diagonal_error.item() + (1 - alpha) * off_diagonal_error.item()

            delta = new_score - current_score
            acceptance_probability = math.exp(-delta / temp) if delta > 0 else 1
   
            if delta > 0:
                acceptance_probability = math.exp(-delta / temp)
                print(f"  Delta > 0, Acceptance Probability: {acceptance_probability:.4f}")
            else:
                acceptance_probability = 1
                delta_neg_count += 1
                            

            if random.random()  < acceptance_probability:
                accepted_count += 1
                current_family = new_family
                current_score = new_score
                

                if new_score < best_score:
                    best_family = new_family
                    best_score = new_score
                print(f"  Accepted new family (better solution or by chance).")
            else:

                rejected_count += 1
                print(f"  Rejected new family (did not pass acceptance probability).")
                
            steps_at_current_temp += 1
            print(f" Current Score: {current_score}, New Score: {new_score}, Acceptance Probability: {acceptance_probability:.4f},delta:{delta:.4f}")
        temp *= cooling_rate
        

        convergence_data.append(current_score)
        print(f"temp: {temp}")
        print(f"Iteration {iteration + 1}/{max_iterations}, Current Score: {current_score:.4f}, Best Score: {best_score:.4f},New Score: {new_score:.4f}, Delta: {delta:.4f}")
        print(f"Temperature {temp}: Delta < 0 count: {delta_neg_count}, Accepted count: {accepted_count}, Rejected count: {rejected_count}")

    return best_family, best_score, convergence_data

import os

if __name__ == "__main__":

    folder_name = "HLA-DRB5_01_01/output_files(L2)"
    max_iterations =2000
    initial_temp =15
    cooling_rate= 0.99
    num_sequences = 410
    alpha = 0.1
    steps_per_temp=1200

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    msa_file = "HLA-DRB5_01_01"
    freq_file = "AAMeanFrequency.dat"
    parameters_file = os.path.join(folder_name, "parameters.txt")
    output_file = os.path.join(folder_name, "optimized_family.txt")
    convergence_file = os.path.join(folder_name, "convergence_data_family.txt")


    o = 9
    input = read_IC(msa_file)
    columns = read_msa_columns(msa_file)
    column_frequencies = calculate_column_frequencies(columns)
    target_coupling_matrix = calculate_coupling(input)
    residue, residue_simple, mean_freq = read_aa_mean_frequency(freq_file)
    optimized_family, best_score, convergence_data = monte_carlo_family_optimization(
        target_coupling_matrix, residue_simple, columns, column_frequencies, mean_freq, o, NA,
        max_iterations=max_iterations, initial_temp=initial_temp, cooling_rate=cooling_rate,num_sequences=num_sequences, steps_per_temp=steps_per_temp
    )


    with open(output_file, "w") as file:
        file.write("Optimized Family and Best Function Value\n")
        for i, seq in enumerate(optimized_family):
            sequence_str = "".join(seq)
            file.write(f"Sequence {i + 1}: {sequence_str}\n")
        file.write(f"\nBest Function Value: {best_score:.4f}\n")


    with open(convergence_file, "w") as file:
        file.write("Iteration\tObjective Function Value\n")
        for iteration, score in enumerate(convergence_data):
            file.write(f"{iteration + 1}\t{score:.4f}\n")

    print(f"Files have been saved in the '{folder_name}' folder.")
