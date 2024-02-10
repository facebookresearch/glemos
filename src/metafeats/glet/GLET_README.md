GraphLET - A Higher-order Subgraph Statistics Package
=================


## Synopsis
A fast efficient parallel library for computing higher-order subgraph statistics of very large graphs.
The library is designed to be fast for both large sparse graphs as well as dense graphs.

GLet is implemented as a command-line utility with no external dependencies. 
The package requires only a compiler supporting unordered_map from the C++11 standard.

## Compile
Current GCC compiler requires a -std=c++11 flag. For example, you can compile the source using MinGW compiler on Windows with:
	g++ -O3 -std=c++11 -o glet.exe glet.cpp
	
For Mac/Linux, the source can be compiled as:
    make clean
    make -j4

See Makefile for other options. 


## How to use GLET
The GLET tool has only four command-line arguments:

	Usage: glet [options] input_filename [output_filename]
	options:
	    -t type         set type of graphlet node/edge (default node)
	    -k size         size of graphlets to compute (default 4)
	    -w threads      number of threads (default max)
	    -v verbose      show information or not (default 0)
	    -g graph type   type of input: asym, sym, bip (default: sym)
	    
	Note if `-g bip` then only relevant orbits are computed (4-cycles, etc.)
	
	Example: ./glet -k 4 -t node test.edges test.4-node-counts

To see the above options, simply type ``./glet`` (assuming the source has been compiled and/or has an executable available).

The input and output file arguments must be declared at the end with the input file first, followed by the output file.

#### Type
This indicates whether the graphlets should be computed on the nodes or edges. Currently, we only support nodes. See the ``-t`` flag below.

	./glet -k 5 -t node test.edges test.5-counts

#### Size
The size of k-vertex graphlet orbits to count. Options are 4 or 5.

For orbits of size k=4,

	./glet -k 4 -t node test.edges test.4-counts
	
For orbits of size k=5, 

	./glet -k 5 -t node test.edges test.5-counts

#### Input file
Input file encoding the network is in a simple text format. Each line of the file encodes an undirected edge. Node ids can begin at 0 or 1. 

#### Output file
Output file will consist of N (or M) lines, one for each node (or edge) in the input graph. Each line of the output file consists of 15 (k=4 graphlet orbits for nodes) or 73 (k=5 graphlet orbits for edges) space-separated orbit counts.

3. GRAPHLET SIZE:  The size of k-vertex graphlets to count. Options: 4/5.
4. GRAPHLET COUNT: This indicates whether the graphlets should be computed on the nodes or edges. Options: 0=nodes, 1=edges.

### Graphlet Orbits for Nodes

## EXAMPLES
#### 4-Node Graphlet Orbits
To compute all graphlet orbits of size 4 for nodes:

	./glet -k 4 -t node gletdata/test.edges gletdata/test.4-graphlets


To compute all graphlet orbits of size 4 for edges:

	./glet -k 4 -t edge gletdata/test.edges gletdata/test.4-edge-graphlets



#### 5-Node Graphlet Orbits
To compute all graphlet orbits of size 5 for nodes:

	./glet -k 5 -t node gletdata/test.edges gletdata/test.5-graphlets
	


##NOTES
Node graphlet counts: 

	degree, 2-star (pos1), 2-star (pos2), triangle, ....
		