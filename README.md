# sudoku-using-grover

This jupyter notebook illustrates implementing Grover's algorithm to solve 3 by 3 and 4 by 4 sudokus in qiskit.

## Essence of Grover's algorithm

Grover's algorithm is a quantum algorithm that can be used to perform unstructured search. Suppose we have a desired winner state *w*. Then we begin with initial state *s*. Grover's algorithm consists of amplitude reflection and amplification. Let *s'* be an arbitary state in which is orhtonormal to the state *s*. We first reflect *s* with respect to *s'*. Then we perform additional reflection with respec to *s* this time. Then, we will obtain a state closer to *w*. By repeating this operation multiple times we will obtain a state that is almost identical to state *w*. For system with N variables, we can  obtain the state *w*, by repeating this operation root N times.

## Sudoku
The value of each number in the box is stroed in qubits in binary. To solve sudoku using Grover's algorithm, we need 2 oracles. First one is the one that performs reflection if and only if the state corresponds to the solution of the sudoku. Second oracle we have to implement is the one that performs additional reflection with respect to uniformly superpositioned state. 

## 3 by 3 sudoku


## 4 by 4 sudoku
