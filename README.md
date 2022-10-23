# sudoku-using-grover

This jupyter notebook illustrates implementing Grover's algorithm to solve 3 by 3 and 4 by 4 sudokus in qiskit.

## Essence of Grover's algorithm

Grover's algorithm is a quantum algorithm that can be used to perform unstructured search. Suppose we have a desired winner state *w*. Then we begin with initial state *s*. Grover's algorithm consists of amplitude reflection and amplification. Let *s'* be an arbitary state in which is orhtonormal to the state *s*. We first reflect *s* with respect to *s'*. Then we perform additional reflection with respec to *s* this time. Then, we will obtain a state closer to *w*. By repeating this operation multiple times we will obtain a state that is almost identical to state *w*. For system with N variables, we can  obtain the state *w*, by repeating this operation root N times.

## Sudoku
The value of each number in the box is stroed in qubits in binary. To solve sudoku using Grover's algorithm, we need 2 oracles. First one is the one that performs reflection if and only if the state corresponds to the solution of the sudoku. Second oracle we have to implement is the one that performs additional reflection with respect to uniformly superpositioned state. 
Given these requirements, it is sensible to create 3 different registers. First one for storing the value of numbers in sudoku box. Second one for checking the conditions. Last one with only one qubit that checks the all of the conditions, meaning that it will perform add 1 mod 2 to its state if all of the conditions are satisfied. We can set the last register to |-> state, so that the overall state is going to reflect if and only if all of the conditions are satisfied.

## 3 by 3 sudoku
For 3 by 3 sudoku, we have to arrange 3 numbers: 0,1 and 2(00, 01, and 10 in binary). Therefore we would need 2 qubits to store the value of the number in each box. In order to check whether certain combination satisfies the sudoku rules, we would need to check 2 conditions. First, we have to check whether the 1s are in the right position. For example, when we are considering the combination for the firt row, possible combination is 00 01 10. We can notice that the two 1s are placed in different digits. Second condition is to check whether there is 11 in any of the boxes. These 2 conditions were implemented in qauntum circuit by using CNOT gates. Two conditions needs to be checked for each row or column. One qubit is requird to check each condition. Hence, I needed 30 qubits in total to completely run the algorithm. GIven circumstances, I did not have acceess to devies that can run this amount of quits so I verified the algorithm by running it ovor just one row. 

## 4 by 4 sudoku
Similar methodology was used to solve 4 by 4 sudoku. However, for condition register I have only used 1 qubits with mutiple toffoli gates. The controll qubits for each toffoli gate were the qubits representing the numbers in sudoku that should be 1 to satisfy the sudoku conditions. Again, I needed 40 qubits to simulate the entire sudoku, so I just tested one row to verify the algorithm I designed. 
