# Stress-test-generation-with-Genetic-Algorithm

## About:
The script takes as input a .txt file that describes a circuit consisting of elementary gates (AND, NAND, OR, NOR, NOT, NOR, XNOR). 

## Input file structure:
• The first line of the .txt file describes the input level signals. Consider the example given below describing the .txt input file structure for a simple circuit.
• The file is not necessarily sorted. The script sorts the file considering inputs and outputs provided.
• Every row has necessarily the following structure: gate_type, output_id, input1_id, input2_id, input3_id, ..., inputn_id, <br>
> where:    n=1, if gate_type == "NOT" <br>
          n∈[2, +∞], else. 
          

**Note:** The structure of the circuit **is not sequentially desrcibed from the .txt input file**. This means that an element placed above another one in the .txt file does not necessarily mean that it belongs to a lower level in the circuit. The corresponding level of every element is determined from its input. Below, the list of the types of inputs based on their syntax in the .txt file:
1) i1, i2, ..., in: input level inputs. Elements with these kind of inputs are elements contained in the first level of the circuit.
2) t1, t2, ..., tn: intermediate level inputs. Elements with these kind of inputs are elements contained in every level of the circuit that is not the input level.
3) o1, o2, ..., on: the outputs of the circuit. 

### Described circuit:
![Screenshot 2023-04-28 230841](https://user-images.githubusercontent.com/48795138/235244030-37d9ff24-a4c5-4b13-ade0-1a6bf3598aff.png)

### Corresponding input file structure:


![my_plot](https://user-images.githubusercontent.com/48795138/235240263-b75c4634-98eb-449
f-b275-175f9b920f8e.jpg)
