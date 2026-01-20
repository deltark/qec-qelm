import pickle
import numpy as np
import matplotlib.pyplot as plt
import plothist

nqubits = 3

##bitstrings in logical 1 state of Steane code
logical_1_states = ['1111111', '0101010', '1001100', '0011001', '1110000', '0100101', '1000011', '0010110']
##bitstrings that are within distance 1 of logical 1 state
error_1_states = []
for state in logical_1_states:
    error_1_states.append(state)
    for i in range(7):
        flipped_bit = '1' if state[i] == '0' else '0'
        error_state = state[:i] + flipped_bit + state[i+1:]
        error_1_states.append(error_state)
#convert to integers
error_1_states_int = [int(state, 2) for state in error_1_states]
# print(f'Error states (int): {error_1_states_int}')

logical_0_states = ['0000000', '1010101', '0110011', '1100110', '0001111', '1011010', '0111100', '1101001']
##bitstrings that are within distance 1 of logical 0 state
error_0_states = []
for state in logical_0_states:
    error_0_states.append(state)
    for i in range(7):
        flipped_bit = '1' if state[i] == '0' else '0'
        error_state = state[:i] + flipped_bit + state[i+1:]
        error_0_states.append(error_state)
#convert to integers
error_0_states_int = [int(state, 2) for state in error_0_states]
# print(f'Error states (int): {error_0_states_int}')

variance = {'encoded': [], 'raw': []}

for error_prob in [0.0, 0.0001, 0.001, 0.01, 0.1]:

    output_values = {'encoded': [], 'raw': []}

    # for error_prob in [0.0, 0.01, 0.03, 0.07, 0.1]:
    for x in np.arange(0, 1, 0.2):
        # filename = f'results/circuit_runs/steane_code_{nqubits}logqubits_errorprob{error_prob}_x{x:.2f}.pkl'

        # unencoded_counts = {'0': 0, '1': 0}
        # nb_codestates = 0
        # nb_decodable_states = 0
        # with open(filename, 'rb') as f:
        #     counts = pickle.load(f)
        # for bitstring, count in counts.items():
        #     if bitstring in logical_0_states or bitstring in logical_1_states:
        #         nb_codestates += count
        #     if bitstring in error_0_states:
        #         unencoded_counts['0'] += count
        #         nb_decodable_states += count
        #     elif bitstring in error_1_states:
        #         unencoded_counts['1'] += count
        #         nb_decodable_states += count
            
        #     average_result = (unencoded_counts['0']*(-1) + unencoded_counts['1']) / nb_decodable_states if nb_decodable_states > 0 else 0
        #     output_values['encoded'].append(average_result)

            filename = f'results/circuit_runs/raw_{nqubits}qubits_errorprob{error_prob}_x{x:.2f}.pkl'
            with open(filename, 'rb') as f:
                counts = pickle.load(f)
            average_result = (counts.get('0', 0)*(-1) + counts.get('1', 0)) / sum(counts.values())
            output_values['raw'].append(average_result)

            # print(f'For error probability {error_prob} and x={x:.2f}, average count for code states: {average_result}')
            # print(f'Number of code states counted: {nb_codestates}')
            # print('/n')
    for key in output_values.keys():
        variance[key].append(np.var(output_values[key]))

#plot variance vs error probability
plt.figure()
# for key in variance.keys():
#     plt.plot([0.0, 0.0001, 0.001, 0.01, 0.1], variance[key], label=key, marker='o')
plt.plot([0.0, 0.0001, 0.001, 0.01, 0.1], variance['raw'], marker='o')
# plt.xscale('log')
# plt.xlim([0.00001, 0.2])
plt.xlabel('Error Probability')
plt.ylabel('Variance of Output Values')
plt.title('Variance of Output Values vs Error Probability')
# plt.legend()
plt.savefig(f'results/circuit_runs/variance_vs_error_probability_n{nqubits}.png')
        

