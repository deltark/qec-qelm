import pickle
import numpy as np
import matplotlib.pyplot as plt
import plothist

nqubits = 3
pT = 0.15

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

# decodable_states = np.sort(np.array(error_0_states_int + error_1_states_int))
# print(f'Decodable states (int): {decodable_states}')

variance = {'encoded': [], 'raw': []}
mean = {'encoded': [], 'raw': []}

# filename = f'results/circuit_runs/steane_code_{nqubits}logqubits_errorprob0.1_x0.00.pkl'
# with open(filename, 'rb') as f:
#     counts = pickle.load(f)

# # print(counts)
# measured_states = np.sort([int(bitstring, 2) for bitstring in counts.keys()])
# print(f'Measured states (int): {measured_states}')

for error_prob in [0.0, 0.0001, 0.001, 0.01, 0.1]:
# for error_prob in [0.0]:

    output_values = {'encoded': [], 'raw': []}
    output_variance = {'encoded': [], 'raw': []}
    exact_expval = []

    # for error_prob in [0.0, 0.01, 0.03, 0.07, 0.1]:
    # for x in np.arange(0.0, 1.0, 0.2):
    for x in [0.4]:
        # ======= ENCODED =======
        # filename = f'results/circuit_runs/steane_code_{nqubits}logqubits_pT{pT}_errorprob{error_prob}_x{x:.2f}.pkl'

        # unencoded_counts = {'0': 0, '1': 0}
        # nb_codestates = 0
        # nb_decodable_states = 0
        # # nb_discarded_states = 0
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
        #     # else:
        #     #     nb_discarded_states += count

              
        # average_result = (unencoded_counts.get('0', 0)+ unencoded_counts.get('1', 0)*(-1) ) / nb_decodable_states if nb_decodable_states > 0 else 0
        # # sumd2 = (unencoded_counts.get('0', 0)*(-1-average_result)**2 + unencoded_counts.get('1', 0)*(1-average_result)**2)
        # # variance_result = sumd2/nb_decodable_states
        # output_values['encoded'].append(average_result)
        # # output_variance['encoded'].append(variance_result)

        # # print('Unencoded counts:', unencoded_counts)

        # ======= RAW ==========

        filename = f'results/circuit_runs/raw_{nqubits}qubits_pT{pT}_errorprob{error_prob}_x{x:.2f}_500shots.pkl'
        with open(filename, 'rb') as f:
            counts = pickle.load(f)
        average_result = (counts.get('0', 0) + counts.get('1', 0)*(-1)) / sum(counts.values())
        # sumd2 = (counts.get('0', 0)*(-1-average_result)**2 + counts.get('1', 0)*(1-average_result)**2)
        # variance_result = sumd2 /sum(counts.values())
        output_values['raw'].append(average_result)
        # output_variance['raw'].append(variance_result)

        if error_prob == 0.0:
            filename = f'results/circuit_runs/raw_{nqubits}qubits_pT{pT}_errorprob{error_prob}_x{x:.2f}_statevector.pkl'
            with open(filename, 'rb') as f:
                exact_expval.append(pickle.load(f))

        # print('Raw counts', counts)

        # print(f'For error probability {error_prob} and x={x:.2f}, average count for code states: {average_result}')
            # print(f'Number of code states counted: {nb_codestates}')
            # print('/n')
        # print(f'For error probability {error_prob} and x={x:.2f}:'
        #       f'\n  Number of codestates counted: {nb_codestates}'
        #       f'\n  Number of decodable states counted: {nb_decodable_states}'
        #     #   f'\n  Number of discarded states: {nb_discarded_states}'
        #       f'\n  Unencoded counts: {unencoded_counts}')
    
    # print(output_values)
    # print("Shots var", output_variance)

    print("Exact_expectation_values:")
    print(exact_expval)

    for key in output_values.keys():
        variance[key].append(np.var(output_values[key]))
        mean[key].append(np.mean(output_values[key]))
    if error_prob == 0:
        mean_exact = np.mean(exact_expval)
        var_exact = np.var(exact_expval)
        # mean['raw'][0] = mean_exact
        # variance['raw'][0] = np.var(exact_expval)

print(f'Variance at error probability 0: encoded={variance["encoded"][0]}, raw={variance["raw"][0]}')
print(f'Mean at error probability 0: encoded={mean["encoded"][0]}, raw={mean["raw"][0]}')
# plot variance vs error probability

plt.figure()
# plt.hlines(var_exact, 0, 4, colors="black")
plt.hlines(mean_exact, 0, 4, colors="black")
# plt.hlines(mean_exact+var_exact, 0, 4, colors="grey")
# plt.hlines(mean_exact-var_exact, 0, 4, colors="grey")
# for key in variance.keys():
#     # plt.plot(range(5), variance[key], label=key, marker='o')
#     plt.errorbar([0, 1, 2, 3, 4], mean[key], yerr = variance[key], marker = 'x')
plt.plot(range(5), mean['raw'], marker = "x")

    
# plt.plot([0.0, 0.0001, 0.001, 0.01, 0.1], variance['raw'], marker='o')
# plt.xscale('log')
# plt.xlim([0.00001, 0.2])
plt.xlabel('Error Probability')
plt.ylabel('Variance of Output Values')
plt.title('Variance of Output Values vs Error Probability')
plt.legend()
plt.savefig(f'results/circuit_runs/variance_vs_error_probability_n{nqubits}_pT{pT}.png')
        

