# Stress-test-generation-with-Genetic-Algorithm

## About:
The script takes as input a .txt file that describes a circuit consisting of elementary gates (AND, NAND, OR, NOR, NOT, NOR, XNOR). The pipeline of the script consists of sorting the circuit according to the inputs provided for every element, the calculation of two states that produce the greatest switching activity on the circuit using a genetic algorithm for 3 individual runs and plotting the greatest switching activity for each generation, for each individual run. 

## How does the genetic algorithm work?:

**1) Initial population (seed)**: <br>
• Workload: time series of input values of the circuit. 

Algorithm parameters: <br>
• Workload length L: The size of a time series, measured as the number of input vectors (default: L=2).  <br>
• Population size N: the number of distinct workloads that the algorithm will explore at each step.  <br><br>
Initially, we start with a population of N random workloads. Each distinct workload (individual workload) is a time series of length L.  <br><br>
For example, if L=2, the structure of a (random) workload can be: <br>
![image](https://user-images.githubusercontent.com/48795138/235300338-28d8b95f-6cdb-4dc4-85b0-fed76af3a0ad.png) <br><br>


**2) Calculation of the switching activity**: <br>
In this step we calculate the switching activity for every individual workload of the current generation, defined as: <br>
• switching_activity(i) = absolute number of switches on the components of the circuit induced by individual workload i. <br><br>


**3) Selection:** <br> 
At this stage, we select from the population as parents the two individuals who achieve the highest switching activity. <br><br>

**4) Crossover**: <br>
From the 2 selected parents from step 3), we generate a new population. In the newly generated population by default 2 of the members are the 2 parents that produced the population because 1. so far they provide the best result and 2. we are not yet sure that their offspring will have a better result. <br><br>

**5) Mutation**: <br>
With a very small probability, called the mutation rate m, we change the bits of the offspring generated from the crossover in the previous step. We do not mutate the two parents of the generation. The value of m can be very small, in our occassion m is equal to **0.05**, i.e. **5% probability of mutation for every bit**.

## Input file structure:
• The first line of the .txt file describes the input level signals. Consider the example given below describing the .txt input file structure for a simple circuit. <br>
• The file is not necessarily sorted. The script sorts the file considering inputs and outputs provided.<br>
• Every row has necessarily the following structure: gate_type, output_id, input1_id, input2_id, input3_id, ..., inputn_id, <br>
> where:    n=1, if gate_type == "NOT" <br>
          n∈[2, +∞], else. 
          

**Note:** The structure of the circuit **is not sequentially desrcibed from the .txt input file**. This means that an element placed above another one in the .txt file does not necessarily mean that it belongs to a lower level in the circuit. The corresponding level of every element is determined from its input. Below, the list of types of inputs based on their syntax in the .txt file:
1) i1, i2, ..., in: input level inputs. Elements with these kind of inputs are elements contained in the first level of the circuit.
2) t1, t2, ..., tn: intermediate level inputs. Elements with these kind of inputs are elements contained in every level of the circuit that is not the input level.
3) o1, o2, ..., on: the outputs of the circuit. 

### Example circuit:
![Screenshot 2023-04-28 230841](https://user-images.githubusercontent.com/48795138/235244030-37d9ff24-a4c5-4b13-ade0-1a6bf3598aff.png)

### Corresponding input file structure for example circuit:

![Screenshot 2023-04-28 233710](https://user-images.githubusercontent.com/48795138/235249654-aa63e40f-8aa7-42b0-8235-c16ea10becc4.png)

# Example:
![image](https://user-images.githubusercontent.com/48795138/235258860-58642050-72f9-41a2-a4e7-fc504b4365bb.png)

