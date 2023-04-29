import sys
import matplotlib.pyplot as plt
import statistics
import random
import numpy as np

circuit_inputs = []  # top inputs
output_gates_names = [] 
output_gates = []

def generate_truthtable(n: int):
    truth_table = [[(j >> i) & 1 for i in range(n)] for j in range(2**n)]
    truth_table = [i[::-1] for i in truth_table]
    return truth_table

def NOT(signal_prob: float):
    return 1 - signal_prob


def NOR(signal_probs: list):
    output = 1
    for i in signal_probs:
            output *= 1 - i
            
    return output

def OR(signal_probs: list):
    return 1 - NOR(signal_probs)


def AND(signal_probs: list):
    output = 1
    for i in signal_probs:
            output *= i
            
    return output

def NAND(signal_probs: list):
    return 1 - AND(signal_probs)


def XNOR2inp(two_signal_probs: list):
    return NOR(two_signal_probs) + AND(two_signal_probs)


def XNOR(signal_probs: list):
    output = XNOR2inp([signal_probs[0], signal_probs[1]])
    
    for i in signal_probs[2:]:
        output = XNOR2inp([output, i])
    
    return output
        
def XOR(signal_probs: list):
    return 1 - XNOR(signal_probs)

def calculate_Esw_for_gate(gate_output: float):
    return 2 * gate_output * (1 - gate_output)

def open_file(path: str):
    try:
        fd = open(path, 'r')
        return fd
    
    except IOError:
        sys.exit("Error: File " + path + " does not appear to exist in the given path.")   

class Circuit:
    def __init__(self, elements_table: list, signals_table: list, top_inputs: list):
        self.elements_table = elements_table
        self.signals_table = signals_table
        self.top_inputs = top_inputs
       
class Element(Circuit):
    def __init__(self, gate_type: str, inputs: list, inputs_alph: list,  output: int, output_alph: str ):
        self.gate_type = gate_type
        self.inputs = inputs
        self.inputs_alph = inputs_alph
        self.output = output
        self.output_alph = output_alph
        
    def __str__(self):
        inputs = ' '.join(map(str, self.inputs))
        inputs_alph = ' '.join(map(str, self.inputs_alph))
        return (str(self.gate_type) + ", [" + str(inputs) + "], [" + str(inputs_alph) + "], " + str(self.output) 
                + ", " + self.output_alph)

def check_file_type(first_line):
    first_line = first_line.replace('\n', '')
    inputs = first_line.split(" ")
    
    if inputs[0] == "top_inputs":
        return inputs[1:]
    
    return weed_line(first_line)  

def weed_line(line):    
    line = line.replace('\n', '')
    gate_type = line.partition('(')[0]
    line = line.split(" ")

    return line

def get_common_elements(input_signal_list, outputs_memory):
    common_elements_indices = [i for i in range(len(outputs_memory)) if outputs_memory[i] in input_signal_list]
    return common_elements_indices

def compute_circuit_inputs(all_inputs, outputs_memory):
    # keep order after difference, by sorting by all_inputs index
    set_dif = sorted(set(all_inputs) - set(outputs_memory), key=all_inputs.index)
    return set_dif[:len(set_dif)]

def sort_inputs(input_list: list):
    i_lst = [i for i in input_list if i[0] == 'i']
    t_lst = [t for t in input_list if t[0] == 't']
    
  
    i_lst.sort(key = lambda x: int(x[1:]))
    t_lst.sort(key = lambda x: int(x[1:]))

    return i_lst + t_lst
    
def init_circuit_from_txt(inputs: list, path: str):
    global circuit_inputs
    
    fd = open_file(path)
    
    file_content = fd.read()
    file_content = file_content.split('\n')
    

    # check if top_inputs are explicitly given in .txt file
    check_line = check_file_type(file_content[0])
    
    if check_line[0] not in ('AND', 'NAND', 'OR', 'NOR', 'XOR', 'XNOR', 'NOT'):
        inputs = check_line
        file_content = file_content[1:] # if explicit top_inputs declaration
                                        # ignore first line of text file    
    all_inputs = []
    elements_table = []
    signals_table = []
    top_inputs = []
    
    circuit = Circuit(elements_table, signals_table, top_inputs)
    
    outputs_memory = []  # contains all output signals of current processing level
                       # used in order to understand whether we have moved to the
                       # next processing level.
            
    outputs_memory_indices = []  # cmon mr. tenentes, no dictionaries?? really? 
    
    signal_count = 0
       
    for line in file_content:
        input_signals_list_numerical = []
        
        line = weed_line(line)
        
        gate_type = line[0]
        input_signals_list_alph = line[2:len(line)]
        all_inputs += input_signals_list_alph
        output_signal = line[1]
                        
        common_signals_indices = get_common_elements(input_signals_list_alph, outputs_memory)
               
        if common_signals_indices:
            for i in circuit.elements_table:
                if i.output == 0:
                    i.output = signal_count
                    outputs_memory_indices.append(signal_count)
                    signal_count += 1
            
            for i in common_signals_indices:
                input_signals_list_numerical.append(outputs_memory_indices[i])
                
            # add padding of 0s in case a gate has as input the output of a NOT (1-input) gate. 
            if gate_type != 'NOT':
                input_signals_list_numerical += [0] * (len(input_signals_list_alph) - len(input_signals_list_numerical)) 
                                                  
        else:
            current_gate_inputs_num = len(input_signals_list_alph)
            input_signals_list_numerical = [i for i in range(signal_count, signal_count + current_gate_inputs_num)]
            signal_count += current_gate_inputs_num
            
        outputs_memory.append(output_signal)
              
        # instantiate an element from line and add it to elements_table:
        element = Element(gate_type, input_signals_list_numerical, input_signals_list_alph, 0, output_signal)
        
        circuit.elements_table.append(element)
           
        line = fd.readline()
    
    
    circuit.elements_table[-1].output = signal_count
    
    circuit_inputs = compute_circuit_inputs(all_inputs, outputs_memory)
    
    circuit_inputs = sort_inputs(circuit_inputs)
       
    return circuit
    
def have_at_least_one_common(inputs_alph, top_inputs_current):  
    return len(set(inputs_alph).intersection(set(top_inputs_current))) > 0
    
def rectify_top_element_inputs(element, top_inputs_current_lvl):
    top_element_flag = False
    top_element_counter = 0
    
    for idx, i in enumerate(element.inputs_alph):
        if i in top_inputs_current_lvl:
            element.inputs[idx] = top_inputs_current_lvl.index(i)
            
            top_element_counter += 1
        
    if top_element_counter == len(element.inputs_alph):
        top_element_flag = True

    return top_element_flag

def sort_elements_table(circuit):
    elements_table_sorted = [] 
    
    
    signal_count = len(circuit_inputs)
    
    elements_table_len = len(circuit.elements_table)
        
    for element in circuit.elements_table:
        element.inputs_alph = sort_inputs(element.inputs_alph)  # we sorting the inputs so that the connection order
        
        
        # e.g. (b, a) instead of (a, b) is irrelevant.
        top_element_flag = rectify_top_element_inputs(element, circuit_inputs)
              
        if top_element_flag:
            elements_table_sorted.append(element)
    
    # sort elements_table_sorted by inputs.
    elements_table_sorted.sort(key = lambda x: x.inputs[0]) 
    
    # alphabetic IDs of signals we have seen till now
    everything = [i for i in circuit_inputs]
    # numerical IDs of signals we have seen till now 
    everything_indices = [i for i in range(len(everything))]
    
    # rectify top input elements' output ID 
    for element in elements_table_sorted:
        element.output = signal_count
        everything.append(element.output_alph)
        everything_indices.append(signal_count)
        signal_count += 1
        circuit.elements_table.remove(element)
   
    added = [i for i in elements_table_sorted]
    while len(elements_table_sorted) != elements_table_len:
        current_top_inputs = []
        current_top_inputs_indices = []
        
        for element in circuit.elements_table :
            if element in added:
                continue
                
            # all inputs must obligingly be outputs of previous level of elements. 
            if not(set(element.inputs_alph) <= set(everything)):
                continue
                
            added.append(element)
            
            # assign the right input IDs
            for idx, i in enumerate(element.inputs_alph):
                if i in everything:
                    indx = everything.index(i) 
                    element.inputs[idx] = everything_indices[indx]
            
            # assign the right output ID and append it to already seen outputs list (everything)
            element.output = signal_count
            signal_count += 1
            current_top_inputs.append(element.output_alph)
            current_top_inputs_indices.append(element.output)
            elements_table_sorted.append(element)
                    
        everything += current_top_inputs
        everything_indices += current_top_inputs_indices
    
    circuit.elements_table = elements_table_sorted
            
def process(circuit, inputs):
    global output_gates_names, output_gates
    
    MAX_SIGNAL_ID = circuit.elements_table[-1].output
    tt_input = inputs
    
    circuit.signals_table = [0 for i in range(MAX_SIGNAL_ID + 1)]
    circuit.top_inputs = inputs       
    
    for idx, i in enumerate(inputs):
        circuit.signals_table[idx] = i
                    
    for i in circuit.elements_table:
        
        output_gates_names.append(i.gate_type)
        
        inputs = [circuit.signals_table[ind] for ind in i.inputs] 
        if i.gate_type == 'NOT':
            res = NOT(inputs[0])
            circuit.signals_table[i.output] = res
          
        elif i.gate_type == 'AND':
            res = AND(inputs)
            circuit.signals_table[i.output] = res
        
        elif i.gate_type == 'NAND':
            res = NAND(inputs)
            circuit.signals_table[i.output] = res
                
        elif i.gate_type == 'OR':
            res = OR(inputs)
            circuit.signals_table[i.output] = res        
        
        elif i.gate_type == 'NOR':
            res = NOR(inputs)
            circuit.signals_table[i.output] = res
                        
        elif i.gate_type == 'XOR':
            res = XOR(inputs)
            circuit.signals_table[i.output] = res
                
        elif i.gate_type == 'XNOR':
            res = XNOR(inputs)
            circuit.signals_table[i.output] = res           
        else:
            print("Unsupported type of gate. Please check possible linefeeds at the end of file.")
            sys.exit(-1)
            
        output_gates.append(res)
    
    return (circuit.signals_table[-1], output_gates)

def print_truthtable_results(output_gates_names: list, output_gates: list, 
                             circuit_inputs_num: list, circuit_output: float, Esw: float):
    print('==========================================')
    print("Given inputs:")
    
    for idx, i in enumerate(circuit_inputs):
        print(i + " = " + str(circuit_inputs_num[idx]))
    print('==========================================')
    print("Result for given circuit: " + str(circuit_output))
    print("Esw for given circuit: " + str(Esw))
    print('==========================================')
    print('\n')
    
    # uncoment if signal probabilities as inputs:
    '''
    for idx, i in enumerate(output_gates_names):
        print("Result for " + i + ": " + str(output_gates[idx]))
        print("Esw for " + i + ": " + str(calculate_Esw_for_gate(output_gates[idx])))
        if idx != len(output_gates_names) - 1:
            print('\n')
     '''

def plot_switches_graph(y_axis, num_iter):
    x_axis = [i for i in range(1, num_iter + 1)]
    
    font = {'color':  'k',
            'size': 12 }
    
    plt.figure(figsize=(8, 8))
    
    plt.text(18, 28.2, "Mean: {}".format(round(statistics.mean(y_axis), 2)), fontdict = font)
    plt.text(18, 27.2, "Variance: {}".format(round(statistics.variance(y_axis), 2)), fontdict = font)
        
    plt.title("Number of switches for " + str(num_iter) + " stress-tests.")
    plt.xlabel("individual")
    plt.ylabel("number of switches")
    plt.plot(x_axis, y_axis)
    plt.show()
    
def cloned_individuals(individual1, individual2):
    return (individual1[0] == individual2[0]) and (individual1[1] == individual2[1])


def mutate(offsprings, mutation_rate=0.05):
    '''
    np.random.binomial(n, p, size=offspring_array.shape): Draw (random) samples from a binomial distribution, in the shape
    of offsping_array np array.
    '''
    mutated_offsprings = []
    
    for offspring in offsprings:
        offspring_array = np.array(offspring)
        mask = np.random.binomial(1, mutation_rate, size=offspring_array.shape)
        mutated_offspring = np.abs(offspring_array - mask)
        mutated_offsprings.append(mutated_offspring.tolist())
        
    return mutated_offsprings

def calculate_score(individual: list, L=2):
    # L: number of individuals (workloads) in a single stress-test
    global output_gates_names, output_gates
    
    if len(individual) == 0:
        individual = [[random.randint(0, 1) for i in range(20)] for j in range(L)] 

    gate_outputs = []
    
    for i in individual:
        output_gates_names = []
        output_gates = []
        circuit = init_circuit_from_txt(i, 'test.txt')
        sort_elements_table(circuit)
        (circuit_output, output_gates) = process(circuit, i)
        gate_outputs.append(output_gates)
    
        
    return (sum(a != b for a,b in zip(gate_outputs[0], gate_outputs[1])), individual) 

def find_parents(population, scores):
    max_idx = scores.index(max(scores))
    second_max_idx = scores.index(sorted(scores)[-2])
    count_idx = -2
    
    # make sure that the two parents are not same individual.
    while True:
        if cloned_individuals(population[max_idx], population[second_max_idx]):
            count_idx -= 1
            second_max_idx = scores.index(sorted(scores)[count_idx])
        else:
            break
  
    return (max_idx, second_max_idx)

def find_first_parents(num_individuals):
    scores = []
    population = []
    for i in range(num_individuals):
        (score, individual) = calculate_score([], 2)
        scores.append(score)
        population.append(individual)
   
    
    (max_idx, second_max_idx) = find_parents(population, scores)
    return (population[max_idx], population[second_max_idx], scores[max_idx], scores[second_max_idx])
   
def crossover(num_individuals, parent1, parent2, score1: int, score2: int):
    """
    Crossover two parent stress-tests to reproduce offspring stress-tests.

    :param list parent1: Parent no.1. The parent with the highest score
    :param list parent2: Parent no.2. The parent with the second-highest score
    :param int score1: Score of parent1
    :param int score2: Score of parent2
    :param int R: parameter determining the number of and which rows each parent is going to  
        contribute to the offspring. 
    :param int : C is astochastic parameter determining whether the R first rows are going to be 
        contributed to the offspring from parent1 (if R == 1) or from parent2 (if R == 2).
    """
         
    NUM_OFFSPRINGS = num_individuals - 2
    offsprings = []
   
    for i in range(NUM_OFFSPRINGS):
        C = random.randint(1, 2)
        R = random.randint(0, 2)
        
        if C == 1 and R == 1:
            offsprings.append([[i for i in parent1[0]], [i for i in parent2[1]]])
            
        elif (C == 1 and R == 2) or (R == 0 and C == 2):
            offsprings.append([[i for i in parent1[0]], [i for i in parent1[1]]])
            
        elif C == 2 and R == 1:
            offsprings.append([[i for i in parent2[0]], [i for i in parent1[1]]])
        
        elif (C == 2 and R == 2) or (R == 0 and C == 1):
            offsprings.append([[i for i in parent2[0]], [i for i in parent2[1]]])

    
    offsprings = mutate(offsprings)
    
    offsprings.append(parent1)
    offsprings.append(parent2)
      
    scores = []
    for i in offsprings:
        (score, individual) = calculate_score(i, 2)
        scores.append(score)
    
        
    (max_idx, second_max_idx) = find_parents(offsprings, scores)
    return (offsprings[max_idx], offsprings[second_max_idx], scores[max_idx], scores[second_max_idx])


def main():       
    list_scoreGs = []
    parents1 = []
    
    
    
    for i in range(3):
        (parent1, parent2, score1, score2) = find_first_parents(2000)  # seed
        scoreGs = []
        for j in range(100):
            (parent1, parent2, score1, score2) = crossover(30, parent1, parent2, score1, score2)
            scoreGs.append(score1)
            
            if j == 99:
                parents1.append(parent1)
                
        list_scoreGs.append(scoreGs)
 
    x_axis = [i for i in range(1, 101)]
    plt.title("Best score for each generation of 30 individuals.")
    plt.xlabel("Generation")
    plt.ylabel("Number of switches")
    
    line1, = plt.plot(x_axis, list_scoreGs[0], color='r', label='Run 1')
    line2, = plt.plot(x_axis, list_scoreGs[1], color='g', label='Run 2')
    line3, = plt.plot(x_axis, list_scoreGs[2], color='b', label='Run 3')
    
    plt.legend(handles=[line1, line2, line3])

    plt.show()
    
    for i in range(3):
        print("The two states that produce the greatest switching activity when switching from one to another for run no.{} is:".format(str(i+1)))
        print(parents1[i][0])
        print(parents1[i][1])
        
    
        
if __name__ == "__main__":
    main() 
