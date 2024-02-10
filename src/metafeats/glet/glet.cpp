// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

/*!\mainpage GraphLET (GLET) Package
 *
 * \section INTRORODUCTION
 *
 * A fast efficient parallel library for computing higher-order subgraph statistics of very large graphs.
 * The library is designed to be fast for both large sparse graphs as well as dense graphs.
 * See the GLET_README for a complete list of features.
 *
 * \section AUTHORS
 *
 * Ryan A. Rossi    (rrossi@alumni.purdue.edu),<BR>
 * Nesreen K. Ahmed (n.kamel@gmail.com)<BR>
 *
 * \section CITING
 *
 * If you used GLET for your publication, please cite the following paper:
 *
 * 		@inproceedinGRAPHLET_SIZE{ahmed2015icdm,
 * 			title={Efficient Graphlet Counting for Large Networks},
 * 			author={Nesreen K. Ahmed and Jennifer Neville and Ryan A. Rossi and Nick Duffield},
 * 			booktitle={ICDM},
 * 			pages={1--10},
 * 			year={2015}
 * 		}
 *
 * \section COPYRIGHT & CONTACT
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) 2012-2017,<BR>
 * Ryan A. Rossi (rrossi@alumni.purdue.edu), Nesreen K. Ahmed (n.kamel@gmail.com), All rights reserved.<BR><BR>
 *
 */

#ifdef WIN32
#else
#include <sys/time.h>
#include <unistd.h>
#endif
#include <stdlib.h>
#include <float.h>
#include <cstddef>
#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>      // std::stringstream, std::stringbuf
#include <limits>
#include "math.h"
#include <vector>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <ctime>
#include <fstream>
#include <set>
#include <unordered_map>
#include <algorithm>


#include <sys/types.h>
#include <sys/timeb.h>
#include <sys/time.h>


#define schedule_type dynamic
#define schedule_type_auto auto

#ifdef _OPENMP
#  include <omp.h>
#else
int omp_get_max_threads()       { return 1; }
void omp_set_num_threads(int64)   {}
int omp_get_thread_num()        { return 0; }
#endif



using namespace std;

typedef double double64;
typedef long long int64;

fstream fin, fout; // input and output files
int64 GRAPHLET_SIZE=5; // graphlet SIZE
int64 GRAPHLET_TYPE = 0; // graphlet count TYPE: 0 = nodes, 1 = edges
int64 NUM_GRAPHLETS = 73;  // number of graphlet counts: 73 graphlet counts for k=5, or 15 for k=4 graphlets
int64 NUM_ELE = 0; // number of nodes or edges, depending on the GRAPHLET COUNT TYPE (see GRAPHLET_TYPE variable above). The NUM_ELE variable is set in the preprocessing() function.
//bool verbose = 0;
bool fix_start_idx = true; /// read_edge_list, etc. sets to false, if vertex ids begin at 0, otherwise, all vertex ids are deincremented by 1, since it assumed they start at 1


typedef pair<int64,int64> PII;

struct TIII {
		int64 first, second, third;
		inline TIII(int64 const f, int64 const s, int64 const t) : first(f), second(s), third(t) {}
};

struct PAIR {
		int64 a, b;
		inline PAIR(int64 const aa, int64 const bb) : a(min(aa, bb)), b(max(aa, bb)) {}
		inline bool operator <(const PAIR &other) const { return (a < other.a) || (a == other.a && b < other.b); }
		inline bool operator ==(const PAIR &other) const { return a == other.a && b == other.b; }
};

struct hash_PAIR {
		inline size_t operator()(PAIR const &x) const { return (x.a << 8) ^ (x.b << 0); }
};


struct TRIPLE {
		int64 a, b, c;
		TRIPLE(int64 const a0, int64 const b0, int64 const c0) : a(a0), b(b0), c(c0) {
			if (a > b) swap(a, b);
			if (b > c) swap(b, c);
			if (a > b) swap(a, b);
		}

		inline bool operator <(TRIPLE const &other) const {
			return (a < other.a) || (a == other.a && (b < other.b || (b == other.b && c < other.c)));
		}
		inline bool operator ==(TRIPLE const &other) const {
			return a == other.a && b == other.b && c == other.c;
		}
};

struct hash_TRIPLE {
		inline size_t operator ()(TRIPLE const &x) const { return (x.a << 16) ^ (x.b << 8) ^ (x.c << 0); }
};

#define adj_chunk (8 * sizeof(int64))

unordered_map<PAIR, int64, hash_PAIR> common2;
unordered_map<TRIPLE, int64, hash_TRIPLE> common3;
unordered_map<PAIR, int64, hash_PAIR>::iterator common2_it;
unordered_map<TRIPLE, int64, hash_TRIPLE>::iterator common3_it;

#define common3_get(x) (((common3_it = common3.find(x)) != common3.end()) ? (common3_it->second) : 0)
#define common2_get(x) (((common2_it = common2.find(x)) != common2.end()) ? (common2_it->second) : 0)


int64 n,m,d_max; // n = number of nodes, m = number of edges
int64 *deg; // degrees of individual nodes
PAIR *edges; // list of edges
//PAIR const *const edges_end;

int64 **adj; // adj[x] - adjacency list of node x
PII **inc; // inc[x] - incidence list of node x: (y, edge id)
//inline
//bool adjacent_list(int64 x, int64 y) { return binary_search(adj[x],adj[x]+deg[x],y); }
int64 *adj_matrix; // compressed adjacency matrix
//const int64 adj_chunk = 8*sizeof(int64);
//inline
//bool adjacent_matrix(int64 x, int64 y) { return adj_matrix[(x*n+y)/adj_chunk]&(1<<((x*n+y)%adj_chunk)); }
////inline bool (*adjacent)(int64,int64);
bool (*adjacent)(int64,int64);


inline bool adjacent_list(int64 const x, int64 const y) { return binary_search(adj[x], adj[x] + deg[x], y); }
inline bool adjacent_matrix(int64 const x, int64 const y) { return adj_matrix[(x * n + y) / adj_chunk] & (1 << ((x * n + y) % adj_chunk)); }
inline int64 getEdgeId(int64 const x, int64 const y) { return inc[x][lower_bound(adj[x], adj[x] + deg[x], y) - adj[x]].second; }


int64 **orbit; // orbit[x][o] - how many times does node x participate in orbit o
//int64 **orbit; // orbit[x][o] - how many times does node x participate in orbit o
//int64 **eorbit; // eorbit[e][i] - how many times does edge e participate in orbit i

double get_time() {
	struct timeval t;
	gettimeofday(&t, 0);
	return (t.tv_sec*1.0 + t.tv_usec/1000000.0);
}

double tic() { return get_time(); }
void toc(double & start) { start = get_time() - start; }

/// number of jobs each worker is assigned (at a time). Must be a positive int64eger.
#define BLOCK_SIZE 64
#define SCHEDULE dynamic
//int64 block_size;


/// Properties of graph
bool is_adj, is_weighted, is_GRAPHLET_SIZEtats, verbose;
string fn;

/* stores the (row/col) indices of each nonzero (row/col ind) */
vector<int64> E; // edges
/*  stores the position of the neighbors in edges */
vector<long long> V; // vertices
/*  input edge weights (if given) */
//vector<double> wt;

/**
 * @brief Map to get original ids
 *
 * NOTE: Vertex IDs are only mapped when needed.
 * For instance, if graph file uses string names for vertices instead of numeric IDS.
 * Also, graph files with certain problems (gaps in IDs, etc.) are remapped to enable more efficient computations
 */
vector<int64> vertex_lookup;


/**
 * @brief Get the neighbor (vertex id)
 * @param pos is the position of the neighbor in edges for a specific vertex
 */
inline long long get_neighbor(int64 & pos) { return E[pos]; }
inline int64 num_vertices() { return V.size() - 1; }
inline int64 num_edges() { return E.size()/2; }

/**
 * @brief Degree of a vertex v (Number of neighbors adjacency to v).
 *
 * @param v is the id of the vertex
 * @return degree or number of neighbors adjacent to v denoted as $|N(v)|$ or $d_{v}$
 */
inline int64 get_degree(long long & v) { return V[v+1] - V[v]; }

/**
 * @brief Density of the graph G
 *
 * @return Graph density
 */
inline double64 density() { return (double64)num_edges() / (num_vertices() * (num_vertices() - 1.0) / 2.0); }

/**
 * @brief Computes density assuming n vertices and m edges (given as input)
 *
 * @param n is the number of vertices
 * @param m is the number of edges
 * @return Graph density resulting from n vertices and m edges given as input
 */
inline double64 density(int64 n, int64 m) { return (double64) m / (n * (n - 1.0) / 2.0); }

/**
 * @brief Get the extension from a full filename given as input
 *
 * @param filename a filename
 * @return file extension, e.g., "filename.txt", then ".txt" is returned.
 */
string get_file_extension(const string& filename) {
	string::size_type result;
	string fileExtension = "";
	result = filename.rfind('.', filename.size() - 1);
	if(result != string::npos)
		fileExtension = filename.substr(result+1);
	return fileExtension;
}


/**
 * @brief Gets the filename from an arbitrary path
 * @param s string containing the path of the file
 * @return filename consisting of the name and extension, that is, "file.txt"
 */
string get_filename_from_path(const string s) {
	char sep = '/';
#ifdef _WIN32
	sep = '\\';
#endif
	size_t i = s.rfind(sep, s.length( ));
	if (i != string::npos) {
		return(s.substr(i+1, s.length( ) - i));
	}
	return s;
}

/**
 * Get filename only: 
 */
string get_filename_only(const string s) {
	char sep = '.';
#ifdef _WIN32
	sep = '\\';
#endif
	size_t i = s.rfind(sep, s.length( ));
	if (i != string::npos) {
		return(s.substr(0, i));
	}
	return s;
}


bool detect_weighted_graph(string & line, string & delim) {
	int64 num_tokens = 0;
	string buf; // Have a buffer string
	stringstream ss(line); // Insert the string int64o a stream
	vector<string> tokens; // Create vector to hold our words

	while (ss >> buf) { tokens.push_back(buf); }
	if (verbose) printf("number of tokens in line = %lu \n", tokens.size());
	if (tokens.size() == 3) return true; // weighted graph (3rd column)
	return false;   // unweighted, only edge list with two columns
}

void detect_delim(string & line, string & delim) {
	if (delim == "") {
		std::size_t prev = 0, pos;
		std::string tab_spaces("   ");
		if ((pos = line.find_first_of(',', prev)) != std::string::npos) {
			delim = ',';
		}
		else if ((pos = line.find_first_of('\t', prev)) != std::string::npos) {
			delim = '\t';
		}
		else if ((pos = line.find(tab_spaces)) != std::string::npos) {
			printf("found tab-spaces delimiter \n");
			delim = "   ";
		}
		else if ((pos = line.find_first_of(' ', prev)) != std::string::npos) {
			delim = ' ';
		}
	}

	if (delim == "") {
		if (get_file_extension(fn) == "csv")
			delim = ',';
		else if (get_file_extension(fn) == "tab")
			delim = '\t';
		else if (get_file_extension(fn) == "mtx")
			delim = ' ';
		else
			delim = ' ';
		if (verbose) cout << "[glet]  no delimiter recognized, using \"" << delim.c_str()[0] << "\" as delimiter!" <<endl;
	}
	else
		if (verbose) cout << "[glet]  detected \"" << delim << "\" as delimiter! " <<endl;
}

inline
void get_token(int64 & v, string & line, string & delim, size_t & pos, size_t & prev) {
	if ((pos = line.find(delim, prev)) != std::string::npos) {
		if (pos > prev) {
			v = atoi(line.substr(prev, pos-prev).c_str());
		}
		prev = pos+1;
	}
	else if (prev < line.length())
		v = atoi(line.substr(prev, std::string::npos).c_str());
}

inline
void get_token(double & weight, string & line, string & delim, size_t & pos, size_t & prev, bool & is_weighted_graph) {
	if ((pos = line.find(delim, prev)) != std::string::npos) {
		if (pos > prev) {
			weight = atof(line.substr(prev, pos-prev).c_str());
		}
		prev = pos+1;
	}
	else if (prev < line.length())
		weight = atof(line.substr(prev, std::string::npos).c_str());
}

//bool fix_start_idx;

/**
 * /brief Reads a general edge list, makes limited assumptions about the graph
 *
 * WEIGHTS: All weights are discarded, unless the graph is temporal
 * LABELS:  Vertices are relabeled, and the old ids are discarded, unless specified.
 */
int64 read_edge_list(const string& filename) {
	cout << "[reading generic edge list: read_edge_list func]  filename: " << filename <<endl;
	map< int64, vector<int64> > vert_list;
	int64 v = -1, u = -1, num_es = 0, self_edges = 0;
	double weight;
	string delimiters = " ,\t", delim="", line="", token="";
	string graph_exif = "";

	ifstream file (filename.c_str());
	if (!file) { if (verbose) cout << filename << "File not found!" <<endl; return 0; }

	// check if vertex ids start at 0 or 1
	is_weighted = false;
	fix_start_idx = true;
	bool ignore_first_line = false;
	stringstream iss;

	// save graph info/comments at top of file
	while(std::getline(file, line) && (line[0] == '#' || line[0] == '%')) {
		graph_exif += line;
		if (line.find("MatrixMarket matrix coordinate pattern symmetric") != std::string::npos) {
			delim = ' ';
			ignore_first_line = true;
		}
	}

	bool is_mtx_format = false;
	int64 num_verts = 0, num_edges = 0;
//	cout << "file extension: " << get_file_extension(filename) <<endl;
	if (get_file_extension(filename) == "mtx") {
		is_mtx_format=true;
		//        if (verbose)
		cout << "[glet: edge list graph reader]  mtx file detected!" <<endl;
		iss << line;
		int64 cols = 0;
//		iss >> num_verts >> cols >> num_edges;
//		if(num_verts!=cols) { cout<<"[glet]  error: this is not a square matrix, attempting to proceed."<<endl; }
		std::getline(file, line); // get next line
	}

	// detect the delim for reading the graph
	detect_delim(line,delim);

	// detect if line has three columns, third is assumed to be for weights
	is_weighted = detect_weighted_graph(line,delim);
	if (is_weighted)    printf("weighted graph detected \n");

	int64 num_self_loops=0;
	int64 max_node_id = 0; // largest vertex id (assumed to be int64s)
//	int64 min_node_id = 10000000;
//	cout << "[before]  min_node_id=" << min_node_id <<endl;
	m=0;
	// handle the first line (find starting vertex id)
//	if (is_mtx_format==false && line!="") { /// NOT mtx format, since if mtx format, then this line is simply the number of nodes/edges, and NOT an ACTUAL EDGE, which is what we assume for ".edges" format
	if (line != "") {
		iss.clear();
		iss.str(line);
		weight=-1;
		iss >> v >> u >> weight;
		cout << "[glet: edge list reader]  initial line: " << v << ", " << u << ", " << weight <<endl; // weight will be -1, if no third value

		if (v == 0 || u == 0) { fix_start_idx = false; }
		if (v==u) { num_self_loops++; }
		else {
			if (v > max_node_id) max_node_id = v;
			if (u > max_node_id) max_node_id = u;
			m++;
		}
	}


	if (verbose) cout << "[glet: graph reader]  reading a general edge list (limited assumptions)" <<endl;
	// find starting vertex id, compute the number of vertices to expect (since gaps in vertex ids are possible)
	while(std::getline(file, line)) {
		if (line != "") { // ensure line actually contains data
			iss << line;

			// first line contained three values, but second line has only two, then we assume the first line was a mistake in the mtx
//			if (is_weighted && m<3 && detect_weighted_graph(line,delim)==false) is_weighted=false;

			// ignore comments
			if (line[0] == '%' || line[0] == '#') continue;

			std::size_t prev = 0, pos;
			get_token(v,line,delim,pos,prev);
			get_token(u,line,delim,pos,prev);

			if (v == 0 || u == 0) { fix_start_idx = false; }
			if (v==u) { num_self_loops++; continue; }
			if (v > max_node_id) max_node_id = v;
			if (u > max_node_id) max_node_id = u;

			m++;
		}
	}

	cout << "fix_start_idx = " << fix_start_idx <<endl;

	n = max_node_id+1;
	if (fix_start_idx) n = max_node_id; // set the number of nodes

	cout << "[glet: graph reader]  num self loops: " << num_self_loops <<endl;
	if (verbose) cout << "[glet: graph reader]  largest vertex id is " << max_node_id <<endl;

	file.close();
	if (verbose) {
		if (fix_start_idx) cout << "[glet: graph reader]  vertex ids from the file begin at 1" <<endl;
		else cout << "[glet: graph reader]  vertex ids begin at 0" <<endl;
	}

	ifstream fin (filename.c_str());
	if (!fin) { cout << filename << "Error: file not found!" <<endl; return 0; }


	// read input graph
	int64 d_max=0;
	edges = (PAIR*)calloc(m,sizeof(PAIR));
	deg = (int64*)calloc(n,sizeof(int64));

	int64 edge_id = 0;
	int64 vertex_id = 0;
	while(std::getline(fin, line)) {
		if (line != "") { // ensure line actually contains data
			iss << line;
			if (line[0] == '%' || line[0] == '#') continue;

			std::size_t prev = 0, pos; // prev is last location in the line
			get_token(v,line,delim,pos,prev);
			get_token(u,line,delim,pos,prev);
			if (fix_start_idx) {
				v--;
				u--;
			}
			if (v == u)  self_edges++;
			else {
				if (n<20) {
					cout << "v=" << v << ", u=" << u <<endl;
				}
				deg[v]++; deg[u]++;
				edges[edge_id]=PAIR(v,u);
				edge_id++;
			}
		}
	}
	fin.close();

	for (int64 i=0;i<n;i++) d_max=max(d_max,deg[i]);
	cout << "|V|    (num nodes) = " << n <<endl;
	cout << "|E|    (num edges) = " << m <<endl;
	cout << "rho      (density) = " << double(density(n,m)) <<endl;
	cout << "d_max (max degree) = " << d_max <<endl;
	fin.close();

	if (n<20) {
		for (int64 i=0;i<m;i++) {
			int64 a=edges[i].a, b=edges[i].b;
			cout << "a=" << a << ", b=" << b <<endl;
		}
	}
	if ((int64)(set<PAIR>(edges,edges+m).size())!=m) {
		cerr << "ERROR: Input file contains **duplicate undirected edges**." <<endl;
		return 0;
	}

    cout << "not setting up an adj matrix (graph is larger than 100MB)" <<endl;
    adjacent = adjacent_list;

	// set up adjacency, incidence lists
	adj = (int64**)calloc(n,sizeof(int64*));
	for (int64 i=0;i<n;i++) adj[i] = (int64*)malloc(deg[i]*sizeof(int64));
	inc = (PII**)calloc(n,sizeof(PII*));
	for (int64 i=0;i<n;i++) inc[i] = (PII*)malloc(deg[i]*sizeof(PII));
	int64 *d = (int64*)calloc(n,sizeof(int64));
	for (int64 i=0;i<m;i++) {
		int64 a=edges[i].a, b=edges[i].b;
		adj[a][d[a]]=b; adj[b][d[b]]=a;
		inc[a][d[a]]=PII(b,i); inc[b][d[b]]=PII(a,i);
		d[a]++; d[b]++;
	}
	for (int64 i=0;i<n;i++) {
		sort(adj[i],adj[i]+deg[i]);
		sort(inc[i],inc[i]+deg[i]);
	}

	cout << "initializing orbit counts" <<endl;

	// int64 NUM_GRAPHLETS=73;
	if (GRAPHLET_TYPE==0) { /// NODE graphlet counts
		NUM_GRAPHLETS=73;
		if (GRAPHLET_SIZE==4) { NUM_GRAPHLETS=15; }
		orbit = (int64**)malloc(n*sizeof(int64*));
		for (int64 i=0;i<n;i++) orbit[i] = (int64*)calloc(NUM_GRAPHLETS,sizeof(int64));
	}
	else {
		NUM_GRAPHLETS=68;
		if (GRAPHLET_SIZE==4) { NUM_GRAPHLETS=12; }
		orbit = (int64**)malloc(m*sizeof(int64*));
		for (int64 i=0;i<m;i++) orbit[i] = (int64*)calloc(NUM_GRAPHLETS,sizeof(int64));
	}
cout << NUM_GRAPHLETS << " graphlet orbits of size " << GRAPHLET_SIZE <<endl;
	
	cout << "finished reading input file" <<endl;
	return 1;
}

/**
 * @brief Specifically designed to be _fast_ for very large graphs
 * Impossible to store full adj of large sparse graphs, instead
 * we create a lookup table for each vertex, and build it on the fly,
 * using this info to mark and essentially remove the multiple edges
 */
void remove_multiple_edges() {
    vector<int64> ind(V.size(),0);
    vector<long long> vs(V.size(),0);
    vector<int64> es;
    es.reserve(E.size());

    int64 start = 0;
    for (int64 i = 0; i < V.size()-1; i++) {
        start = es.size();
        for (long long j = V[i]; j < V[i + 1]; j++) {
            int64 u = E[j];
            if (ind[u] == 0) {
                es.push_back(E[j]);
                ind[u] = 1;
            }
        }
        vs[i] = start;
        vs[i + 1] = es.size();
        for (long long j = V[i]; j < V[i + 1]; j++) { ind[E[j]] = 0; }
    }
    if (verbose) cout << "[glet: graph reader]  removed " << (E.size() - es.size())/2 << " duplicate edges (multigraph)" <<endl;
    if (verbose) cout << "[remove multiple edges] " << E.size() <<endl;
    V = vs;
    E = es;
    vs.clear();
    es.clear();
}

/**
 * @brief Selects appropriate reader for graph
 *
 * @param filename or path of the graph to read
 */
int64 read_graph(const string& filename) {
	//    is_GRAPHLET_SIZEtats = true;
	fn = filename;
	double sec = get_time();
	string ext = get_file_extension(filename);

	if (verbose) cout << "[glet: graph reader]  All graphs are assumed to be undirected" <<endl;
	if (verbose) cout << "[glet: graph reader]  Self-loops and weights (if any) are discarded" <<endl;

	int64 flag=0;
	if (ext == "edges" || ext == "eg2" || ext == "txt" || ext == "csv") {
		if (verbose) cout << "[glet: general graph reader]  reading the edge list" <<endl;
		flag=read_edge_list(filename);
	}
	else if (ext == "mtx") {
		flag=read_edge_list(filename);
	}
	else {
		if (verbose) cout << "[glet: general graph reader] Unsupported graph format. Attempting to read the graph." <<endl;
		flag=read_edge_list(filename);
	}
	if (verbose) cout << "Reading time " << get_time() - sec << endl;
	return flag;
}


/** count edge orbits of 4-node graphlets */
extern "C"
void edge_count4() {
	clock_t startTime, endTime;
	startTime = clock();
	clock_t startTime_all, endTime_all;
	startTime_all = startTime;
	int64 frac,frac_prev;

	unordered_map<PAIR, int64, hash_PAIR> common2;
	unordered_map<TRIPLE, int64, hash_TRIPLE> common3;
	unordered_map<PAIR, int64, hash_PAIR>::iterator common2_it;
	unordered_map<TRIPLE, int64, hash_TRIPLE>::iterator common3_it;

	/// precompute triangles that span over edges
	printf("stage 1 - precomputing common nodes\n");
	int64 *tri = (int64*)calloc(m,sizeof(int64));
	frac_prev=-1;
	for (int64 i=0; i<m; i++) {
		int64 x=edges[i].a, y=edges[i].b;
		for (int64 xi=0,yi=0; xi<deg[x] && yi<deg[y]; ) {
			if (adj[x][xi]==adj[y][yi]) { tri[i]++; xi++; yi++; }
			else if (adj[x][xi]<adj[y][yi]) { xi++; }
			else { yi++; }
		}
	}
	endTime = clock();
	printf("\t%.2f\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
	startTime = endTime;

	/// count clique graphlets on four edges
	printf("stage 2 - counting full graphlets\n");

	int64 *K4 = (int64*)calloc(m,sizeof(int64));
	int64 *neighx = (int64*)calloc(m,sizeof(int64)); // lookup table - edges to neighbors of x
	int64 *neigh = (int64*)calloc(m,sizeof(int64)), nn; // lookup table - common neighbors of x and y
	PII *neigh_edges = (PII*)calloc(m,sizeof(PII)); // list of common neighbors of x and y

	frac_prev=-1;
	for (int64 x=0; x<n; x++) {
		for (int64 nx = 0; nx < deg[x]; nx++) {
			int64 y = inc[x][nx].first, xy = inc[x][nx].second;
			neighx[y] = xy;
		}
		for (int64 nx = 0; nx < deg[x]; nx++) {
			int64 y = inc[x][nx].first, xy = inc[x][nx].second;
			if (y >= x) break;
			nn = 0;
			for (int64 ny = 0; ny < deg[y]; ny++) {
				int64 z = inc[y][ny].first, yz = inc[y][ny].second;
				if (z >= y) break;
				if (neighx[z] == -1) continue;
				int64 xz = neighx[z];
				neigh[nn] = z;
				neigh_edges[nn] = PII(xz, yz);
				nn++;
			}
			for (int64 i=0; i<nn; i++) {
				int64 z = neigh[i], xz = neigh_edges[i].first, yz = neigh_edges[i].second;
				for (int64 j = i + 1; j < nn; j++) {
					int64 w = neigh[j], xw = neigh_edges[j].first, yw = neigh_edges[j].second;
					if (adjacent(z, w)) {
						K4[xy]++;
						K4[xz]++;
						K4[yz]++;
						K4[xw]++;
						K4[yw]++;
					}
				}
			}
		}
		for (int64 nx = 0; nx < deg[x]; nx++) {
			int64 y = inc[x][nx].first;
			neighx[y] = -1;
		}
	}

	/// count full graphlets for the smallest edge
	for (int64 x=0; x<n; x++) {
		for (int64 nx = deg[x] - 1; nx >= 0; nx--) {
			int64 y = inc[x][nx].first, xy = inc[x][nx].second;
			if (y <= x) break;
			nn = 0;
			for (int64 ny = deg[y] - 1; ny >= 0; ny--) {
				int64 z = adj[y][ny];
				if (z <= y) break;
				if (!adjacent(x, z)) continue;
				neigh[nn++] = z;
			}
			for (int64 i = 0; i < nn; i++) {
				int64 z = neigh[i];
				for (int64 j = i + 1; j < nn; j++) {
					int64 zz = neigh[j];
					if (adjacent(z, zz)) K4[xy]++;
				}
			}
		}
	}
	endTime = clock();
	printf("\t%.2f\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
	startTime = endTime;

	/// set up a system of equations relating orbits for every node
	printf("stage 3 - building systems of equations\n");

	int64 *common = (int64*)calloc(m,sizeof(int64));
	int64 *common_list = (int64*)calloc(m,sizeof(int64)), nc=0;
	frac_prev=-1;

	for (int64 x=0; x<n; x++) {
		// common nodes of x and some other node
		for (int64 i=0; i<nc; i++) common[common_list[i]] = 0;
		nc = 0;
		for (int64 nx=0; nx<deg[x];nx++) {
			int64 y = adj[x][nx];
			for (int64 ny = 0; ny < deg[y]; ny++) {
				int64 z = adj[y][ny];
				if (z == x) continue;
				if (common[z] == 0) common_list[nc++]=z;
				common[z]++;
			}
		}

		for (int64 nx=0; nx<deg[x]; nx++) {
			int64 y = inc[x][nx].first, xy = inc[x][nx].second;
			int64 e = xy;
			for (int64 n1 = 0; n1 < deg[x]; n1++) {
				int64 z = inc[x][n1].first, xz = inc[x][n1].second;
				if (z == y) continue;
				if (adjacent(y, z)) { // triangle
					if (x < y) {
						orbit[e][1]++;
						orbit[e][10] += tri[xy] - 1; // 2e_{10}+2e_{11}
						orbit[e][7] += deg[z] - 2; // e_{7}+e_{9}+2e_{11}
					}
					orbit[e][9] += tri[xz] - 1; // e_{9}+4e_{11}
					orbit[e][8] += deg[x] - 2; // e_{8}+e_{9}+4e_{10}+4e_{11}
				}
			}
			for (int64 n1=0; n1<deg[y]; n1++) {
				int64 z = inc[y][n1].first, yz = inc[y][n1].second;
				if (z == x) continue;
				if (!adjacent(x, z)) { // path x-y-z
					orbit[e][0]++;
					orbit[e][6] += tri[yz]; // 2e_{6}+e_{9}
					orbit[e][5] += common[z]  - 1; // 2e_{5}+e_{9}
					orbit[e][4] += deg[y] - 2; // 2e_{4}+2e_{6}+e_{8}+e_{9}
					orbit[e][3] += deg[x] - 1; // 2e_{3}+2e_{5}+e_{8}+e_{9}
					orbit[e][2] += deg[z] - 1; // e_{2}+2e_{5}+2e_{6}+e_{9}
				}
			}
		}
	}

	/// solve system of equations
	for (int64 e=0; e<m; e++) { /// orbit[e][i] - how many times does edge e participate in orbit i
		orbit[e][11] = K4[e];
		orbit[e][10] = (orbit[e][10] -2 * orbit[e][11]) / 2;
		orbit[e][9] = (orbit[e][9] - 4 * orbit[e][11]);
		orbit[e][8] = (orbit[e][8] - orbit[e][9] - 4*orbit[e][10] - 4*orbit[e][11]);
		orbit[e][7] = (orbit[e][7] - orbit[e][9] - 2*orbit[e][11]);
		orbit[e][6] = (orbit[e][6] - orbit[e][9]) / 2;
		orbit[e][5] = (orbit[e][5] - orbit[e][9]) / 2;
		orbit[e][4] = (orbit[e][4] - 2*orbit[e][6] - orbit[e][8] - orbit[e][9]) / 2;
		orbit[e][3] = (orbit[e][3] - 2*orbit[e][5] - orbit[e][8] - orbit[e][9]) / 2;
		orbit[e][2] = (orbit[e][2] - 2*orbit[e][5] - 2*orbit[e][6] - orbit[e][9]);
	}
	endTime = clock();
	printf("\t%.2f\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
	//	printf("[edge graphlet counting]  total: %.2f\n", (double)(endTime-startTime_all)/CLOCKS_PER_SEC);
}



/** count edge orbits of 5-node graphlets */
extern "C" {
void edge_count5() {
	clock_t startTime, endTime;
	startTime = clock();
	clock_t startTime_all, endTime_all;
	startTime_all = startTime;
	int64 frac,frac_prev;

	unordered_map<PAIR, int64, hash_PAIR> common2;
	unordered_map<TRIPLE, int64, hash_TRIPLE> common3;
	unordered_map<PAIR, int64, hash_PAIR>::iterator common2_it;
	unordered_map<TRIPLE, int64, hash_TRIPLE>::iterator common3_it;

	// precompute common nodes
	printf("stage 1 - precomputing common nodes\n");
	frac_prev=-1;
	for (int64 x=0; x<n; x++) {
		for (int64 n1=0;n1<deg[x];n1++) {
			int64 a=adj[x][n1];
			for (int64 n2=n1+1;n2<deg[x];n2++) {
				int64 b=adj[x][n2];
				PAIR ab=PAIR(a,b);
				common2[ab]++;
				for (int64 n3=n2+1;n3<deg[x];n3++) {
					int64 c=adj[x][n3];
					int64 st = adjacent(a,b)+adjacent(a,c)+adjacent(b,c);
					if (st<2) continue;
					TRIPLE abc=TRIPLE(a,b,c);
					common3[abc]++;
				}
			}
		}
	}
	// precompute triangles that span over edges
	int64 *tri = (int64*)calloc(m,sizeof(int64));
	for (int64 i=0; i<m; i++) {
		int64 x=edges[i].a, y=edges[i].b;
		for (int64 xi=0,yi=0; xi<deg[x] && yi<deg[y]; ) {
			if (adj[x][xi]==adj[y][yi]) { tri[i]++; xi++; yi++; }
			else if (adj[x][xi]<adj[y][yi]) { xi++; }
			else { yi++; }
		}
	}
	endTime = clock();
	printf("\t%.2f sec\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
	startTime = endTime;

	// count complete graphlets on five nodes
	printf("stage 2 - counting full graphlets\n");

	int64 *K5 = (int64*)calloc(m,sizeof(int64));
	int64 *neighx = (int64*)calloc(m,sizeof(int64)); // lookup table - edges to neighbors of x
	int64 *neigh = (int64*)calloc(m,sizeof(int64)), nn; // lookup table - common neighbors of x and y
	PII *neigh_edges = (PII*)calloc(m, sizeof(PII)); // list of common neighbors of x and y
	int64 *neigh2 = (int64*)calloc(m, sizeof(int64)), nn2;
	TIII *neigh2_edges = (TIII *)calloc(m, sizeof(TIII));
	frac_prev=-1;

	for (int64 x=0; x<n; x++) {
		for (int64 nx = 0; nx < deg[x]; nx++) {
			int64 y = inc[x][nx].first, xy = inc[x][nx].second;
			neighx[y] = xy;
		}
		for (int64 nx = 0; nx < deg[x]; nx++) {
			int64 y = inc[x][nx].first, xy = inc[x][nx].second;
			if (y >= x) break;
			nn = 0;
			for (int64 ny = 0; ny < deg[y]; ny++) {
				int64 z = inc[y][ny].first, yz = inc[y][ny].second;
				if (z >= y) break;
				if (neighx[z] == -1) continue;
				int64 xz = neighx[z];
				neigh[nn] = z;
				neigh_edges[nn] = PII(xz, yz);
				nn++;
			}
			for (int64 i = 0 ; i < nn; i++) {
				int64 z = neigh[i], xz = neigh_edges[i].first, yz = neigh_edges[i].second;
				nn2 = 0;
				for (int64 j = i + 1; j < nn; j++) {
					int64 w = neigh[j], xw = neigh_edges[j].first, yw = neigh_edges[j].second;
					if (adjacent(z, w)) {
						neigh2[nn2] = w;
						int64 zw = getEdgeId(z,w);
						neigh2_edges[nn2] = TIII(xw, yw, zw);
						nn2++;
					}
				}
				for (int64 i2 = 0; i2 < nn2; i2++) {
					int64 z2 = neigh2[i2];
					int64 z2x = neigh2_edges[i2].first;
					int64 z2y = neigh2_edges[i2].second;
					int64 z2z = neigh2_edges[i2].third;
					for (int64 j2 = i2 + 1; j2 < nn2; j2++) {
						int64 z3 = neigh2[j2];
						int64 z3x = neigh2_edges[j2].first;
						int64 z3y = neigh2_edges[j2].second;
						int64 z3z = neigh2_edges[j2].third;
						if (adjacent(z2, z3)) {
							int64 zid = getEdgeId(z2, z3);
							K5[xy]++;  K5[xz]++;  K5[yz]++;
							K5[z2x]++; K5[z2y]++; K5[z2z]++;
							K5[z3x]++; K5[z3y]++; K5[z3z]++;
							K5[zid]++;
						}
					}
				}
			}
		}
		for (int64 nx = 0; nx < deg[x]; nx++) {
			int64 y = inc[x][nx].first;
			neighx[y] = -1;
		}
	}
	endTime = clock();
	printf("\t%.2f sec\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
	startTime = endTime;




	// set up a system of equations relating orbits for every node
	printf("stage 3 - building systems of equations\n");
	int64 * common_x = (int64 *)calloc(m,sizeof(int64));
	int64 * common_x_list = (int64 *)calloc(m, sizeof(int64)), nc_x = 0;
	int64 * common_y = (int64 *)calloc(m, sizeof(int64));
	int64 * common_y_list = (int64 *)calloc(m, sizeof(int64)), nc_y = 0;
	for (int64 x=0; x < n; x++) {
		// common nodes of x and some other node
		for (int64 i = 0; i < nc_x; i++) common_x[common_x_list[i]] = 0;
		nc_x = 0;
		for (int64 nx = 0; nx < deg[x]; nx++) {
			int64 a = adj[x][nx];
			for (int64 na = 0; na < deg[a]; na++) {
				int64 z = adj[a][na];
				if (z == x) continue;
				if (common_x[z] == 0) common_x_list[nc_x++] = z;
				common_x[z]++;
			}
		}

		for (int64 nx=0; nx < deg[x]; nx++) {
			int64 y = inc[x][nx].first, xy = inc[x][nx].second;
			int64 e = xy;
			if (y >= x) break;

			// common nodes of y and some other node
			for (int64 i = 0; i < nc_y; i++) common_y[common_y_list[i]] = 0;
			nc_y = 0;
			for (int64 ny = 0; ny < deg[y]; ny++) {
				int64 a = adj[y][ny];
				for (int64 na = 0; na < deg[a]; na++) {
					int64 z = adj[a][na];
					if (z == y) continue;
					if (common_y[z] == 0) common_y_list[nc_y++] = z;
					common_y[z]++;
				}
			}

			int64 f_66 = 0, f_65 = 0, f_62 = 0, f_61 = 0, f_60 = 0, f_51 = 0, f_50 = 0; // 11
			int64 f_64 = 0, f_58 = 0, f_55 = 0, f_48 = 0, f_41 = 0, f_35 = 0; // 10
			int64 f_63 = 0, f_59 = 0, f_57 = 0, f_54 = 0, f_53 = 0, f_52 = 0, f_47 = 0, f_40 = 0, f_39 = 0, f_34 = 0, f_33 = 0; // 9
			int64 f_45 = 0, f_36 = 0, f_26 = 0, f_23 = 0, f_19 = 0; // 7
			int64 f_49 = 0, f_38 = 0, f_37 = 0, f_32 = 0, f_25 = 0, f_22 = 0, f_18 = 0; // 6
			int64 f_56 = 0, f_46 = 0, f_44 = 0, f_43 = 0, f_42 = 0, f_31 = 0, f_30 = 0; // 5
			int64 f_27 = 0, f_17 = 0, f_15 = 0; // 4
			int64 f_20 = 0, f_16 = 0, f_13 = 0; // 3
			int64 f_29 = 0, f_28 = 0, f_24 = 0, f_21 = 0, f_14 = 0, f_12 = 0; // 2

			// smaller (3-node) graphlets
			for (int64 nx1 = 0; nx1 < deg[x]; nx1++) {
				int64 z = adj[x][nx1];
				if (z == y) continue;
				if (adjacent(y, z)) orbit[e][1]++;
				else orbit[e][0]++;
			}
			for (int64 ny = 0;ny < deg[y]; ny++) {
				int64 z = adj[y][ny];
				if (z == x) continue;
				if (!adjacent(x, z)) orbit[e][0]++;
			}

			// edge-orbit 11 = (14,14)
			for (int64 nx1 = 0; nx1 < deg[x]; nx1++) {
				int64 a = adj[x][nx1], xa = inc[x][nx1].second;
				if (a == y || !adjacent(y, a)) continue;
				for (int64 nx2 = nx1+1; nx2 < deg[x]; nx2++) {
					int64 b = adj[x][nx2], xb = inc[x][nx2].second;
					if (b == y || !adjacent(y, b) || !adjacent(a, b)) continue;
					int64 ya = getEdgeId(y,a), yb = getEdgeId(y,b), ab = getEdgeId(a,b);
					orbit[e][11]++;
					f_66 += common3_get(TRIPLE(x, y, a)) - 1; // 2e_{66}+6e_{67}
					f_66 += common3_get(TRIPLE(x, y, b)) - 1;
					f_65 += common3_get(TRIPLE(a, b, x)) - 1; // e_{65}+6e_{67}
					f_65 += common3_get(TRIPLE(a, b, y)) - 1;
					f_62 += tri[xy] - 2; // e_{62}+2e_{66}+3e_{67}
					f_61 += (tri[xa] - 2) + (tri[xb] - 2) + (tri[ya] - 2) + (tri[yb] - 2); // e_{61}+2e_{65}+4e_{66}+12e_{67}
					f_60 += tri[ab] - 2; // e_{60}+e_{65}+3e_{67}
					f_51 += (deg[x]  -3) + (deg[y] - 3); // e_{51}+e_{61}+2e_{62}+e_{65}+4e_{66}+6e_{67}
					f_50 += (deg[a] - 3) + (deg[b] - 3); // e_{50}+2e_{60}+e_{61}+2e_{65}+2e_{66}+6e_{67}
				}
			}

			// edge-orbit 10 = (13,13)
			for (int64 nx1 = 0; nx1 < deg[x]; nx1++) {
				int64 a = adj[x][nx1], xa = inc[x][nx1].second;
				if (a == y || !adjacent(y, a)) continue;
				for (int64 nx2 = nx1 + 1; nx2 < deg[x]; nx2++) {
					int64 b = adj[x][nx2], xb = inc[x][nx2].second;
					if (b == y || !adjacent(y, b) || adjacent(a, b)) continue;
					int64 ya = getEdgeId(y, a), yb = getEdgeId(y, b);
					orbit[e][10]++;
					f_64 += common3_get(TRIPLE(a, b, x)) - 1; // e_{64}+2e_{66}
					f_64 += common3_get(TRIPLE(a, b, y)) - 1;
					f_58 += common2_get(PAIR(a, b)) - 2; // e_{58}+e_{64}+e_{66}
					f_55 += (tri[xa] - 1) + (tri[xb] - 1) + (tri[ya] - 1) + (tri[yb] - 1); // e_{55}+4e_{62}+2e_{64}+4e_{66}
					f_48 += tri[xy] - 2; // 3e_{48}+2e_{62}+e_{66}
					f_41 += (deg[a] - 2) + (deg[b] - 2); // e_{41}+e_{55}+2e_{58}+2e_{62}+2e_{64}+2e_{66}
					f_35 += (deg[x] - 3) + (deg[y] - 3); // e_{35}+6e_{48}+e_{55}+4e_{62}+e_{64}+2e_{66}
				}
			}

			// edge-orbit 9 = (12,13)
			for (int64 nx = 0; nx < deg[x]; nx++) {
				int64 a = adj[x][nx], xa = inc[x][nx].second;
				if (a == y) continue;
				for (int64 ny = 0; ny < deg[y]; ny++) {
					int64 b = adj[y][ny], yb = inc[y][ny].second;
					if (b == x || !adjacent(a, b))  continue;
					int64 adj_ya = adjacent(y,a), adj_xb = adjacent(x,b);
					if (adj_ya + adj_xb != 1) continue;
					int64 ab = getEdgeId(a,b);
					orbit[e][9]++;
					if (adj_xb) {
						int64 xb = getEdgeId(x, b);
						f_63 += common3_get(TRIPLE(a, b, y)) - 1; // 2e_{63}+2e_{65}
						f_59 += common3_get(TRIPLE(a, b, x)); // 2e_{59}+2e_{65}
						f_57 += common_y[a] - 2; // e_{57}+2e_{63}+2e_{64}+2e_{65}
						f_54 += tri[yb] - 1; // 2e_{54}+e_{61}+2e_{63}+2e_{65}
						f_53 += tri[xa] - 1; // e_{53}+2e_{59}+2e_{64}+2e_{65}
						f_47 += tri[xb] - 2; // 2e_{47}+2e_{59}+e_{61}+2e_{65}
						f_40 += deg[y] - 2; // e_{40}+2e_{54}+e_{55}+e_{57}+e_{61}+2e_{63}+2e_{64}+2e_{65}
						f_39 += deg[a] - 2; // e_{39}+e_{52}+e_{53}+e_{57}+2e_{59}+2e_{63}+2e_{64}+2e_{65}
						f_34 += deg[x] - 3; // e_{34}+2e_{47}+e_{53}+e_{55}+2e_{59}+e_{61}+2e_{64}+2e_{65}
						f_33 += deg[b] - 3; // e_{33}+2e_{47}+e_{52}+2e_{54}+2e_{59}+e_{61}+2e_{63}+2e_{65}
					} else if (adj_ya) {
						int64 ya = getEdgeId(y, a);
						f_63 += common3_get(TRIPLE(a, b, x)) - 1;
						f_59 += common3_get(TRIPLE(a, b, y));
						f_57 += common_x[b] - 2;
						f_54 += tri[xa] - 1;
						f_53 += tri[yb] - 1;
						f_47 += tri[ya] - 2;
						f_40 += deg[x] - 2;
						f_39 += deg[b] - 2;
						f_34 += deg[y] - 3;
						f_33 += deg[a] - 3;
					}
					f_52 += tri[ab] - 1; // e_{52}+2e_{59}+2e_{63}+2e_{65}
				}
			}

			// edge-orbit 8 = (10,11)
			for (int64 nx = 0; nx < deg[x]; nx++) {
				int64 a = adj[x][nx];
				if (a == y || !adjacent(y,a)) continue;
				for (int64 nx1 = 0; nx1 < deg[x]; nx1++) {
					int64 b = adj[x][nx1];
					if (b == y || b == a || adjacent(y, b) || adjacent(a, b)) continue;
					orbit[e][8]++;
				}
				for (int64 ny1 = 0; ny1 < deg[y]; ny1++) {
					int64 b = adj[y][ny1];
					if (b == x || b == a || adjacent(x, b) || adjacent(a, b)) continue;
					orbit[e][8]++;
				}
			}

			// edge-orbit 7 = (10,10)
			for (int64 nx = 0; nx < deg[x]; nx++) {
				int64 a = adj[x][nx];
				if (a == y || !adjacent(y, a)) continue;
				for (int64 na = 0;na < deg[a]; na++) {
					int64 b = adj[a][na], ab = inc[a][na].second;
					if (b == x || b == y || adjacent(x, b) || adjacent(y, b)) continue;
					orbit[e][7]++;
					f_45 += common_x[b] - 1; // e_{45}+e_{52}+4e_{58}+4e_{60}
					f_45 += common_y[b] - 1;
					f_36 += tri[ab]; // 2e_{36}+e_{52}+2e_{60}
					f_26 += deg[a] - 3; // 2e_{26}+e_{33}+2e_{36}+e_{50}+e_{52}+2e_{60}
					f_23 += deg[b] - 1; // e_{23}+2e_{36}+e_{45}+e_{52}+2e_{58}+2e_{60}
					f_19 += (deg[x] - 2) + (deg[y] - 2); // e_{19}+e_{33}+2e_{41}+e_{45}+2e_{50}+e_{52}+4e_{58}+4e_{60}
				}
			}

			// edge-orbit 6 = (9,11)
			for (int64 ny1 = 0; ny1 < deg[y]; ny1++) {
				int64 a = adj[y][ny1], ya = inc[y][ny1].second;
				if (a == x || adjacent(x, a)) continue;
				for (int64 ny2 = ny1 + 1; ny2 < deg[y]; ny2++) {
					int64 b = adj[y][ny2], yb = inc[y][ny2].second;
					if (b == x || adjacent(x, b) || !adjacent(a,b)) continue;
					int64 ab = getEdgeId(a, b);
					orbit[e][6]++;
					f_49 += common3_get(TRIPLE(y, a, b)); // 3e_{49}+e_{59}
					f_38 += tri[ab] - 1; // e_{38}+3e_{49}+e_{56}+e_{59}
					f_37 += tri[xy]; // e_{37}+e_{53}+e_{59}
					f_32 += (tri[ya] - 1) + (tri[yb] - 1); // 2e_{32}+6e_{49}+e_{53}+2e_{59}
					f_25 += deg[y] - 3; // e_{25}+2e_{32}+e_{37}+3e_{49}+e_{53}+e_{59}
					f_22 += deg[x] - 1; // e_{22}+e_{37}+e_{44}+e_{53}+e_{56}+e_{59}
					f_18 += (deg[a] - 2) + (deg[b] - 2); // e_{18}+2e_{32}+2e_{38}+e_{44}+6e_{49}+e_{53}+2e_{56}+2e_{59}
				}
			}
			for (int64 nx1 = 0; nx1 < deg[x]; nx1++) {
				int64 a = adj[x][nx1], xa = inc[x][nx1].second;
				if (a == y || adjacent(y, a)) continue;
				for (int64 nx2 = nx1 + 1; nx2 < deg[x]; nx2++) {
					int64 b = adj[x][nx2], xb = inc[x][nx2].second;
					if (b == y || adjacent(y, b) || !adjacent(a, b)) continue;
					int64 ab = getEdgeId(a, b);
					orbit[e][6]++;
					f_49 += common3_get(TRIPLE(x, a, b));
					f_38 += tri[ab] - 1;
					f_37 += tri[xy];
					f_32 += (tri[xa] - 1) + (tri[xb] - 1);
					f_25 += deg[x] - 3;
					f_22 += deg[y] - 1;
					f_18 += (deg[a] - 2) + (deg[b] - 2);
				}
			}

			// edge-orbit 5 = (8,8)
			for (int64 nx = 0;nx < deg[x]; nx++) {
				int64 a = adj[x][nx], xa = inc[x][nx].second;
				if (a == y || adjacent(y, a)) continue;
				for (int64 ny = 0; ny < deg[y]; ny++) {
					int64 b = adj[y][ny], yb = inc[y][ny].second;
					if (b == x || adjacent(x, b) || !adjacent(a, b)) continue;
					int64 ab = getEdgeId(a, b);
					orbit[e][5]++;
					f_56 += common3_get(TRIPLE(x, a, b)); // 2e_{56}+2e_{63}
					f_56 += common3_get(TRIPLE(y, a, b));
					f_46 += tri[xy]; // e_{46}+e_{57}+e_{63}
					f_44 += tri[xa] + tri[yb]; // e_{44}+2e_{56}+e_{57}+2e_{63}
					f_43 += tri[ab]; // e_{43}+2e_{56}+e_{63}
					f_42 += common_x[b] - 2; // 2e_{42}+2e_{56}+e_{57}+2e_{63}
					f_42 += common_y[a] - 2;
					f_31 += (deg[x] - 2) + (deg[y] - 2); // e_{31}+2e_{42}+e_{44}+2e_{46}+2e_{56}+2e_{57}+2e_{63}
					f_30 += (deg[a] - 2) + (deg[b] - 2); // e_{30}+2e_{42}+2e_{43}+e_{44}+4e_{56}+e_{57}+2e_{63}
				}
			}

			// edge-orbit 4 = (6,7)
			for (int64 ny1 = 0; ny1 < deg[y]; ny1++) {
				int64 a = adj[y][ny1];
				if (a == x || adjacent(x, a)) continue;
				for (int64 ny2 = ny1 + 1; ny2 < deg[y]; ny2++) {
					int64 b = adj[y][ny2];
					if (b == x || adjacent(x, b) || adjacent(a, b)) continue;
					orbit[e][4]++;
					f_27 += tri[xy]; // e_{27}+e_{34}+e_{47}
					f_17 += deg[y] - 3; // 3e_{17}+2e_{25}+e_{27}+e_{32}+e_{34}+e_{47}
					f_15 += (deg[a] - 1) + (deg[b] - 1); // e_{15}+2e_{25}+2e_{29}+e_{31}+2e_{32}+e_{34}+2e_{42}+2e_{47}
				}
			}
			for (int64 nx1 = 0; nx1 < deg[x]; nx1++) {
				int64 a = adj[x][nx1];
				if (a == y || adjacent(y, a)) continue;
				for (int64 nx2 = nx1+1; nx2<deg[x]; nx2++) {
					int64 b = adj[x][nx2];
					if (b == y || adjacent(y,b) || adjacent(a,b)) continue;
					orbit[e][4]++;
					f_27 += tri[xy];
					f_17 += deg[x] - 3;
					f_15 += (deg[a] - 1) + (deg[b] - 1);
				}
			}

			// edge-orbit 3 = (5,5)
			for (int64 nx = 0; nx < deg[x]; nx++) {
				int64 a = adj[x][nx];
				if (a == y || adjacent(y, a)) continue;
				for (int64 ny = 0; ny < deg[y]; ny++) {
					int64 b = adj[y][ny];
					if (b == x || adjacent(x,b) || adjacent(a,b)) continue;
					orbit[e][3]++;
					f_20 += tri[xy]; // e_{20}+e_{40}+e_{54}
					f_16 += (deg[x] - 2) + (deg[y] - 2); // 2e_{16}+2e_{20}+2e_{22}+e_{31}+2e_{40}+e_{44}+2e_{54}
					f_13 += (deg[a] - 1) + (deg[b] - 1); // e_{13}+2e_{22}+2e_{28}+e_{31}+e_{40}+2e_{44}+2e_{54}
				}
			}

			// edge-orbit 2 = (4,5)
			for (int64 ny = 0; ny < deg[y]; ny++) {
				int64 a = adj[y][ny];
				if (a == x || adjacent(x, a)) continue;
				for (int64 na = 0; na < deg[a]; na++) {
					int64 b = adj[a][na], ab = inc[a][na].second;
					if (b == y || adjacent(y,b) || adjacent(x,b)) continue;
					orbit[e][2]++;
					f_29 += common_y[b] - 1; // 2e_{29}+2e_{38}+e_{45}+e_{52}
					f_28 += common_x[b]; // 2e_{28}+2e_{43}+e_{45}+e_{52}
					f_24 += tri[xy]; // e_{24}+e_{39}+e_{45}+e_{52}
					f_21 += tri[ab]; // 2e_{21}+2e_{38}+2e_{43}+e_{52}
					f_14 += deg[a] - 2; // 2e_{14}+e_{18}+2e_{21}+e_{30}+2e_{38}+e_{39}+2e_{43}+e_{52}
					f_12 += deg[b] - 1; // e_{12}+2e_{21}+2e_{28}+2e_{29}+2e_{38}+2e_{43}+e_{45}+e_{52}
				}
			}
			for (int64 nx = 0;nx<deg[x];nx++) {
				int64 a = adj[x][nx];
				if (a == y || adjacent(y, a)) continue;
				for (int64 na = 0; na < deg[a]; na++) {
					int64 b = adj[a][na], ab = inc[a][na].second;
					if (b == x || adjacent(x, b) || adjacent(y, b)) continue;
					orbit[e][2]++;
					f_29 += common_x[b] - 1;
					f_28 += common_y[b];
					f_24 += tri[xy];
					f_21 += tri[ab];
					f_14 += deg[a] - 2;
					f_12 += deg[b] - 1;
				}
			}

			// solve system of equations
			orbit[e][67] = K5[e];
			orbit[e][66] = (f_66 - 6 * orbit[e][67]) / 2;
			orbit[e][65] = (f_65 - 6 * orbit[e][67]);
			orbit[e][64] = (f_64 - 2 * orbit[e][66]);
			orbit[e][63] = (f_63 - 2 * orbit[e][65]) / 2;
			orbit[e][62] = (f_62 - 2 * orbit[e][66] -3 * orbit[e][67]);
			orbit[e][61] = (f_61 - 2 * orbit[e][65] -4 * orbit[e][66] -12 * orbit[e][67]);
			orbit[e][60] = (f_60 - 1 * orbit[e][65] -3 * orbit[e][67]);
			orbit[e][59] = (f_59 - 2 * orbit[e][65]) / 2;
			orbit[e][58] = (f_58 - 1 * orbit[e][64] -1 * orbit[e][66]);
			orbit[e][57] = (f_57 - 2 * orbit[e][63] -2 * orbit[e][64] -2 * orbit[e][65]);
			orbit[e][56] = (f_56 - 2 * orbit[e][63]) / 2;
			orbit[e][55] = (f_55 - 4 * orbit[e][62] -2 * orbit[e][64] -4 * orbit[e][66]);
			orbit[e][54] = (f_54 - 1 * orbit[e][61] -2 * orbit[e][63] -2 * orbit[e][65]) / 2;
			orbit[e][53] = (f_53 - 2 * orbit[e][59] -2 * orbit[e][64] -2 * orbit[e][65]);
			orbit[e][52] = (f_52 - 2 * orbit[e][59] -2 * orbit[e][63] -2 * orbit[e][65]);
			orbit[e][51] = (f_51 - 1 * orbit[e][61] -2 * orbit[e][62] -1 * orbit[e][65] -4 * orbit[e][66] -6 * orbit[e][67]);
			orbit[e][50] = (f_50 - 2 * orbit[e][60] -1 * orbit[e][61] -2 * orbit[e][65] -2 * orbit[e][66] -6 * orbit[e][67]);
			orbit[e][49] = (f_49 - 1 * orbit[e][59]) / 3;
			orbit[e][48] = (f_48 - 2 * orbit[e][62] -1 * orbit[e][66]) / 3;
			orbit[e][47] = (f_47 - 2 * orbit[e][59] -1 * orbit[e][61] -2 * orbit[e][65]) / 2;
			orbit[e][46] = (f_46 - 1 * orbit[e][57] -1 * orbit[e][63]);
			orbit[e][45] = (f_45 - 1 * orbit[e][52] -4 * orbit[e][58] -4 * orbit[e][60]);
			orbit[e][44] = (f_44 - 2 * orbit[e][56] -1 * orbit[e][57] -2 * orbit[e][63]);
			orbit[e][43] = (f_43 - 2 * orbit[e][56] -1 * orbit[e][63]);
			orbit[e][42] = (f_42 - 2 * orbit[e][56] -1 * orbit[e][57] -2 * orbit[e][63]) / 2;
			orbit[e][41] = (f_41 - 1 * orbit[e][55] -2 * orbit[e][58] -2 * orbit[e][62] -2 * orbit[e][64] -2 * orbit[e][66]);
			orbit[e][40] = (f_40 - 2 * orbit[e][54] -1 * orbit[e][55] -1 * orbit[e][57] -1 * orbit[e][61] -2 * orbit[e][63] -2 * orbit[e][64] -2 * orbit[e][65]);
			orbit[e][39] = (f_39 - 1 * orbit[e][52] -1 * orbit[e][53] -1 * orbit[e][57] -2 * orbit[e][59] -2 * orbit[e][63] -2 * orbit[e][64] -2 * orbit[e][65]);
			orbit[e][38] = (f_38 - 3 * orbit[e][49] -1 * orbit[e][56] -1 * orbit[e][59]);
			orbit[e][37] = (f_37 - 1 * orbit[e][53] -1 * orbit[e][59]);
			orbit[e][36] = (f_36 - 1 * orbit[e][52] -2 * orbit[e][60]) / 2;
			orbit[e][35] = (f_35 - 6 * orbit[e][48] -1 * orbit[e][55] -4 * orbit[e][62] -1 * orbit[e][64] -2 * orbit[e][66]);
			orbit[e][34] = (f_34 - 2 * orbit[e][47] -1 * orbit[e][53] -1 * orbit[e][55] -2 * orbit[e][59] -1 * orbit[e][61] -2 * orbit[e][64] -2 * orbit[e][65]);
			orbit[e][33] = (f_33 - 2 * orbit[e][47] -1 * orbit[e][52] -2 * orbit[e][54] -2 * orbit[e][59] -1 * orbit[e][61] -2 * orbit[e][63] -2 * orbit[e][65]);
			orbit[e][32] = (f_32 - 6 * orbit[e][49] -1 * orbit[e][53] -2 * orbit[e][59]) / 2;
			orbit[e][31] = (f_31 - 2 * orbit[e][42] -1 * orbit[e][44] -2 * orbit[e][46] -2 * orbit[e][56] -2 * orbit[e][57] -2 * orbit[e][63]);
			orbit[e][30] = (f_30 - 2 * orbit[e][42] -2 * orbit[e][43] -1 * orbit[e][44] -4 * orbit[e][56] -1 * orbit[e][57] -2 * orbit[e][63]);
			orbit[e][29] = (f_29 - 2 * orbit[e][38] -1 * orbit[e][45] -1 * orbit[e][52]) / 2;
			orbit[e][28] = (f_28 - 2 * orbit[e][43] -1 * orbit[e][45] -1 * orbit[e][52]) / 2;
			orbit[e][27] = (f_27 - 1 * orbit[e][34] -1 * orbit[e][47]);
			orbit[e][26] = (f_26 - 1 * orbit[e][33] -2 * orbit[e][36] -1 * orbit[e][50] -1 * orbit[e][52] -2 * orbit[e][60]) / 2;
			orbit[e][25] = (f_25 - 2 * orbit[e][32] -1 * orbit[e][37] -3 * orbit[e][49] -1 * orbit[e][53] -1 * orbit[e][59]);
			orbit[e][24] = (f_24 - 1 * orbit[e][39] -1 * orbit[e][45] -1 * orbit[e][52]);
			orbit[e][23] = (f_23 - 2 * orbit[e][36] -1 * orbit[e][45] -1 * orbit[e][52] -2 * orbit[e][58] -2 * orbit[e][60]);
			orbit[e][22] = (f_22 - 1 * orbit[e][37] -1 * orbit[e][44] -1 * orbit[e][53] -1 * orbit[e][56] -1 * orbit[e][59]);
			orbit[e][21] = (f_21 - 2 * orbit[e][38] -2 * orbit[e][43] -1 * orbit[e][52]) / 2;
			orbit[e][20] = (f_20 - 1 * orbit[e][40] -1 * orbit[e][54]);
			orbit[e][19] = (f_19 - 1 * orbit[e][33] -2 * orbit[e][41] -1 * orbit[e][45] -2 * orbit[e][50] -1 * orbit[e][52] -4 * orbit[e][58] -4 * orbit[e][60]);
			orbit[e][18] = (f_18 - 2 * orbit[e][32] -2 * orbit[e][38] -1 * orbit[e][44] -6 * orbit[e][49] -1 * orbit[e][53] -2 * orbit[e][56] -2 * orbit[e][59]);
			orbit[e][17] = (f_17 - 2 * orbit[e][25] -1 * orbit[e][27] -1 * orbit[e][32] -1 * orbit[e][34] -1 * orbit[e][47]) / 3;
			orbit[e][16] = (f_16 - 2 * orbit[e][20] -2 * orbit[e][22] -1 * orbit[e][31] -2 * orbit[e][40] -1 * orbit[e][44] -2 * orbit[e][54]) / 2;
			orbit[e][15] = (f_15 - 2 * orbit[e][25] -2 * orbit[e][29] -1 * orbit[e][31] -2 * orbit[e][32] -1 * orbit[e][34] -2 * orbit[e][42] -2 * orbit[e][47]);
			orbit[e][14] = (f_14 - 1 * orbit[e][18] -2 * orbit[e][21] -1 * orbit[e][30] -2 * orbit[e][38] -1 * orbit[e][39] -2 * orbit[e][43] -1 * orbit[e][52]) / 2;
			orbit[e][13] = (f_13 - 2 * orbit[e][22] -2 * orbit[e][28] -1 * orbit[e][31] -1 * orbit[e][40] -2 * orbit[e][44] -2 * orbit[e][54]);
			orbit[e][12] = (f_12 - 2 * orbit[e][21] -2 * orbit[e][28] -2 * orbit[e][29] -2 * orbit[e][38] -2 * orbit[e][43] -1 * orbit[e][45] -1 * orbit[e][52]);
		}
	}
	endTime = clock();
	printf("\t%.2f sec\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
	startTime = endTime;
}
}

/** count graphlets on max 4 nodes */
extern "C"
void node_count4() {
	clock_t startTime, endTime;
	startTime = clock();
	clock_t startTime_all, endTime_all;
	startTime_all = startTime;
	int64 frac,frac_prev;

	// precompute triangles that span over edges
	printf("stage 1 - precomputing common nodes\n");
	int64 *tri = (int64*)calloc(m,sizeof(int64));
	frac_prev=-1;
	for (int64 i=0; i<m; i++) {
		int64 x=edges[i].a, y=edges[i].b;
		for (int64 xi=0,yi=0; xi<deg[x] && yi<deg[y]; ) {
			if (adj[x][xi]==adj[y][yi]) { tri[i]++; xi++; yi++; }
			else if (adj[x][xi]<adj[y][yi]) { xi++; }
			else { yi++; }
		}
	}
	endTime = clock();
	printf("\t%.2f\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
	startTime = endTime;

	// count full graphlets
	printf("stage 2 - counting full graphlets\n");
	int64 *K4 = (int64*)calloc(n,sizeof(int64));
	int64 *neigh = (int64*)calloc(n,sizeof(int64)), nn;
	frac_prev=-1;

//		#pragma omp parallel for schedule(dynamic,64) firstprivate(neigh)
	//		shared(tmp_edges) \
//			firstprivate(neigh)
	for (int64 x=0; x<n; x++) {
		for (int64 nx=0;nx<deg[x];nx++) {
			int64 y=adj[x][nx];
			if (y >= x) break;
			nn=0;
			for (int64 ny=0;ny<deg[y];ny++) {
				int64 z=adj[y][ny];
				if (z >= y) break;
				if (adjacent(x,z)==0) continue;
				neigh[nn++]=z;
			}
			for (int64 i=0; i<nn; i++) {
				int64 z = neigh[i];
				for (int64 j=i+1; j<nn; j++) {
					int64 zz = neigh[j];
					if (adjacent(z,zz)) {
						K4[x]++; K4[y]++; K4[z]++; K4[zz]++;
					}
				}
			}
		}
	}
	endTime = clock();
	printf("\t%.2f\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
	startTime = endTime;

	// set up a system of equations relating orbits for every node
	printf("stage 3 - building systems of equations\n");

	int64 *common = (int64*)calloc(n,sizeof(int64));
	int64 *common_list = (int64*)calloc(n,sizeof(int64)), nc=0;
	frac_prev=-1;

//	#pragma omp parallel for schedule(dynamic,64) firstprivate(common, common_list)
	for (int64 x=0; x<n; x++) {
		int64 f_12_14=0, f_10_13=0;
		int64 f_13_14=0, f_11_13=0;
		int64 f_7_11=0, f_5_8=0;
		int64 f_6_9=0, f_9_12=0, f_4_8=0, f_8_12=0;
		int64 f_14=K4[x];

		for (int64 i=0; i<nc; i++) common[common_list[i]]=0;
		nc=0;

		//Node graphlet counts:
		// degree, 2-star-edge (P_3), 2-star-center (P_3), triangle (K_3),
		// 4-path-edge (P_4), 4-path-center (P_4), 3-star-edge (claw-edge), 3-star-center (claw-center)
		// 4-cycle (C_4), tailed-tri-tailEdge (paw-tailEdge), tailed-tri-edge (paw-edge), tailed-tri-center (paw-center)
		// chordal-cycle-edge (diamond-edge), chordal-cycle-center (diamond-center), 4-clique (K_4)

		orbit[x][0]=deg[x];
		// x - middle node
		for (int64 nx1=0;nx1<deg[x];nx1++) {
			int64 y=inc[x][nx1].first, ey=inc[x][nx1].second;
			for (int64 ny=0;ny<deg[y];ny++) {
				int64 z=inc[y][ny].first, ez=inc[y][ny].second;
				if (adjacent(x,z)) { // triangle
					if (z<y) {
						f_12_14 += tri[ez]-1;
						f_10_13 += (deg[y]-1-tri[ez])+(deg[z]-1-tri[ez]);
					}
				} else {
					if (common[z]==0) common_list[nc++]=z;
					common[z]++;
				}
			}
			for (int64 nx2=nx1+1;nx2<deg[x];nx2++) {
				int64 z=inc[x][nx2].first, ez=inc[x][nx2].second;
				if (adjacent(y,z)) { // triangle
					orbit[x][3]++;
					f_13_14 += (tri[ey]-1)+(tri[ez]-1);
					f_11_13 += (deg[x]-1-tri[ey])+(deg[x]-1-tri[ez]);
				} else { // path
					orbit[x][2]++;
					f_7_11 += (deg[x]-1-tri[ey]-1)+(deg[x]-1-tri[ez]-1);
					f_5_8 += (deg[y]-1-tri[ey])+(deg[z]-1-tri[ez]);
				}
			}
		}
		// x - side node
		for (int64 nx1=0;nx1<deg[x];nx1++) {
			int64 y=inc[x][nx1].first, ey=inc[x][nx1].second;
			for (int64 ny=0;ny<deg[y];ny++) {
				int64 z=inc[y][ny].first, ez=inc[y][ny].second;
				if (x==z) continue;
				if (!adjacent(x,z)) { // path
					orbit[x][1]++;
					f_6_9 += (deg[y]-1-tri[ey]-1);
					f_9_12 += tri[ez];
					f_4_8 += (deg[z]-1-tri[ez]);
					f_8_12 += (common[z]-1);
				}
			}
		}

		//Node graphlet counts:
		// degree, 2-star-edge (P_3), 2-star-center (P_3), triangle (K_3),
		// 4-path-edge (P_4), 4-path-center (P_4), 3-star-edge (claw-edge), 3-star-center (claw-center)
		// 4-cycle (C_4), tailed-tri-tailEdge (paw-tailEdge), tailed-tri-edge (paw-edge), tailed-tri-center (paw-center)
		// chordal-cycle-edge (diamond-edge), chordal-cycle-center (diamond-center), 4-clique (K_4)

		// solve system of equations
		orbit[x][14]=(f_14);									// cliques
		orbit[x][13]=(f_13_14-6*f_14)/2; 						//
		orbit[x][12]=(f_12_14-3*f_14);
		orbit[x][11]=(f_11_13-f_13_14+6*f_14)/2;
		orbit[x][10]=(f_10_13-f_13_14+6*f_14);
		orbit[x][9]=(f_9_12-2*f_12_14+6*f_14)/2;					//
		orbit[x][8]=(f_8_12-2*f_12_14+6*f_14)/2;					// cliques
		orbit[x][7]=(f_13_14 + f_7_11 - f_11_13-6 * f_14)/6;		// chordal-cycle
		orbit[x][6]=(2*f_12_14+f_6_9-f_9_12-6*f_14)/2;				// tailed-triangle
		orbit[x][5]=(2*f_12_14+f_5_8-f_8_12-6*f_14);				// 4-cycle
		orbit[x][4]=(2*f_12_14+f_4_8-f_8_12-6*f_14); 				// 3-star
	}

	endTime = clock();
	printf("\t%.2f\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
}


/** count graphlets on max 5 nodes */
extern "C"
void node_count5() {
	clock_t startTime, endTime;
	startTime = clock();
	clock_t startTime_all, endTime_all;
	startTime_all = startTime;
	int64 frac,frac_prev;

	// precompute common nodes
	printf("stage 1 - precomputing common nodes\n");
	frac_prev=-1;
	for (int64 x=0; x<n; x++) {
		for (int64 n1=0;n1<deg[x];n1++) {
			int64 a=adj[x][n1];
			for (int64 n2=n1+1;n2<deg[x];n2++) {
				int64 b=adj[x][n2];
				PAIR ab=PAIR(a,b);
				common2[ab]++;
				for (int64 n3=n2+1;n3<deg[x];n3++) {
					int64 c=adj[x][n3];
					int64 st = adjacent(a,b)+adjacent(a,c)+adjacent(b,c);
					if (st<2) continue;
					TRIPLE abc=TRIPLE(a,b,c);
					common3[abc]++;
				}
			}
		}
	}
	// precompute triangles that span over edges
	int64 *tri = (int64*)calloc(m,sizeof(int64));
	//todo parallel codes caused problems, so remove... be careful if added back...
	// as they can cause negative values in the counts sometimes..
//	#pragma omp parallel for schedule(dynamic,64)
	for (int64 i=0; i<m; i++) {
		int64 x=edges[i].a, y=edges[i].b;
		for (int64 xi=0,yi=0; xi<deg[x] && yi<deg[y]; ) {
			if (adj[x][xi]==adj[y][yi]) { tri[i]++; xi++; yi++; }
			else if (adj[x][xi]<adj[y][yi]) { xi++; }
			else { yi++; }
		}
	}
	endTime = clock();
	printf("\t%.2f sec\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
	startTime = endTime;

	// count full graphlets
	printf("stage 2 - counting full graphlets\n");
//	int64 *K5 = (int64*)calloc(n,sizeof(int64));
//	int64 *neigh = (int64*)malloc(n*sizeof(int64)), nn;
//	int64 *neigh2 = (int64*)malloc(n*sizeof(int64)), nn2;

	int64 *K5 = (int64*)calloc(n,sizeof(int64));
	int64 *neigh = (int64*)calloc(n,sizeof(int64)), nn;
	int64 *neigh2 = (int64*)calloc(n,sizeof(int64)), nn2;
	frac_prev=-1;
	for (int64 x=0; x<n; x++) {
		for (int64 nx=0;nx<deg[x];nx++) {
			int64 y=adj[x][nx];
			if (y >= x) break;
			nn=0;
			for (int64 ny=0;ny<deg[y];ny++) {
				int64 z=adj[y][ny];
				if (z >= y) break;
				if (adjacent(x,z)) {
					neigh[nn++]=z;
				}
			}
			for (int64 i=0; i<nn; i++) {
				int64 z = neigh[i];
				nn2=0;
				for (int64 j=i+1; j<nn; j++) {
					int64 zz = neigh[j];
					if (adjacent(z,zz)) {
						neigh2[nn2++]=zz;
					}
				}
				for (int64 i2=0; i2<nn2; i2++) {
					int64 zz = neigh2[i2];
					for (int64 j2=i2+1; j2<nn2; j2++) {
						int64 zzz = neigh2[j2];
						if (adjacent(zz,zzz)) {
							K5[x]++; K5[y]++; K5[z]++; K5[zz]++; K5[zzz]++;
						}
					}
				}
			}
		}
	}
	endTime = clock();
	printf("\t%.2f sec\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
	startTime = endTime;

	int64 *common_x = (int64*)calloc(n,sizeof(int64));
	int64 *common_x_list = (int64*)calloc(n,sizeof(int64)), ncx=0;
	int64 *common_a = (int64*)calloc(n,sizeof(int64));
	int64 *common_a_list = (int64*)calloc(n,sizeof(int64)), nca=0;

	// set up a system of equations relating orbit counts
	printf("stage 3 - building systems of equations\n");
	frac_prev=-1;
//	#pragma omp parallel for schedule(dynamic,64) firstprivate(common_x, common_x_list, common_a, common_a_list)
	for (int64 x=0; x<n; x++) {
		for (int64 i=0; i<ncx; i++) common_x[common_x_list[i]]=0;
		ncx=0;

		// smaller graphlets
		orbit[x][0] = deg[x];
		for (int64 nx1=0;nx1<deg[x];nx1++) {
			int64 a=adj[x][nx1];
			for (int64 nx2=nx1+1;nx2<deg[x];nx2++) {
				int64 b=adj[x][nx2];
				if (adjacent(a,b)) orbit[x][3]++;
				else orbit[x][2]++;
			}
			for (int64 na=0;na<deg[a];na++) {
				int64 b=adj[a][na];
				if (b!=x && !adjacent(x,b)) {
					orbit[x][1]++;
					if (common_x[b]==0) common_x_list[ncx++]=b;
					common_x[b]++;
				}
			}
		}

		int64 f_71=0, f_70=0, f_67=0, f_66=0, f_58=0, f_57=0; // 14
		int64 f_69=0, f_68=0, f_64=0, f_61=0, f_60=0, f_55=0, f_48=0, f_42=0, f_41=0; // 13
		int64 f_65=0, f_63=0, f_59=0, f_54=0, f_47=0, f_46=0, f_40=0; // 12
		int64 f_62=0, f_53=0, f_51=0, f_50=0, f_49=0, f_38=0, f_37=0, f_36=0; // 8
		int64 f_44=0, f_33=0, f_30=0, f_26=0; // 11
		int64 f_52=0, f_43=0, f_32=0, f_29=0, f_25=0; // 10
		int64 f_56=0, f_45=0, f_39=0, f_31=0, f_28=0, f_24=0; // 9
		int64 f_35=0, f_34=0, f_27=0, f_18=0, f_16=0, f_15=0; // 4
		int64 f_17=0; // 5
		int64 f_22=0, f_20=0, f_19=0; // 6
		int64 f_23=0, f_21=0; // 7



		for (int64 nx1=0; nx1<deg[x]; nx1++) {
			int64 a=inc[x][nx1].first, xa=inc[x][nx1].second;

			for (int64 i=0; i<nca; i++) common_a[common_a_list[i]]=0;
			nca=0;
			for (int64 na=0;na<deg[a];na++) {
				int64 b=adj[a][na];
				for (int64 nb=0;nb<deg[b];nb++) {
					int64 c=adj[b][nb];
					if (c==a || adjacent(a,c)) continue;
					if (common_a[c]==0) common_a_list[nca++]=c;
					common_a[c]++;
				}
			}

			// x = orbit-14 (tetrahedron)
			for (int64 nx2=nx1+1;nx2<deg[x];nx2++) {
				int64 b=inc[x][nx2].first, xb=inc[x][nx2].second;
				if (!adjacent(a,b)) continue;
				for (int64 nx3=nx2+1;nx3<deg[x];nx3++) {
					int64 c=inc[x][nx3].first, xc=inc[x][nx3].second;
					if (!adjacent(a,c) || !adjacent(b,c)) continue;
					orbit[x][14]++;
					f_70 += common3_get(TRIPLE(a,b,c))-1;
					f_71 += (tri[xa]>2 && tri[xb]>2)?(common3_get(TRIPLE(x,a,b))-1):0;
					f_71 += (tri[xa]>2 && tri[xc]>2)?(common3_get(TRIPLE(x,a,c))-1):0;
					f_71 += (tri[xb]>2 && tri[xc]>2)?(common3_get(TRIPLE(x,b,c))-1):0;
					f_67 += tri[xa]-2+tri[xb]-2+tri[xc]-2;
					f_66 += common2_get(PAIR(a,b))-2;
					f_66 += common2_get(PAIR(a,c))-2;
					f_66 += common2_get(PAIR(b,c))-2;
					f_58 += deg[x]-3;
					f_57 += deg[a]-3+deg[b]-3+deg[c]-3;
				}
			}

			// x = orbit-13 (diamond)
			for (int64 nx2=0;nx2<deg[x];nx2++) {
				int64 b=inc[x][nx2].first, xb=inc[x][nx2].second;
				if (!adjacent(a,b)) continue;
				for (int64 nx3=nx2+1;nx3<deg[x];nx3++) {
					int64 c=inc[x][nx3].first, xc=inc[x][nx3].second;
					if (!adjacent(a,c) || adjacent(b,c)) continue;
					orbit[x][13]++;
					f_69 += (tri[xb]>1 && tri[xc]>1)?(common3_get(TRIPLE(x,b,c))-1):0;
					f_68 += common3_get(TRIPLE(a,b,c))-1;
					f_64 += common2_get(PAIR(b,c))-2;
					f_61 += tri[xb]-1+tri[xc]-1;
					f_60 += common2_get(PAIR(a,b))-1;
					f_60 += common2_get(PAIR(a,c))-1;
					f_55 += tri[xa]-2;
					f_48 += deg[b]-2+deg[c]-2;
					f_42 += deg[x]-3;
					f_41 += deg[a]-3;
				}
			}

			// x = orbit-12 (diamond)
			for (int64 nx2=nx1+1;nx2<deg[x];nx2++) {
				int64 b=inc[x][nx2].first, xb=inc[x][nx2].second;
				if (!adjacent(a,b)) continue;
				for (int64 na=0;na<deg[a];na++) {
					int64 c=inc[a][na].first, ac=inc[a][na].second;
					if (c==x || adjacent(x,c) || !adjacent(b,c)) continue;
					orbit[x][12]++;
					f_65 += (tri[ac]>1)?common3_get(TRIPLE(a,b,c)):0;
					f_63 += common_x[c]-2;
					f_59 += tri[ac]-1+common2_get(PAIR(b,c))-1;
					f_54 += common2_get(PAIR(a,b))-2;
					f_47 += deg[x]-2;
					f_46 += deg[c]-2;
					f_40 += deg[a]-3+deg[b]-3;
				}
			}

			// x = orbit-8 (cycle)
			for (int64 nx2=nx1+1; nx2<deg[x]; nx2++) {
				int64 b=inc[x][nx2].first, xb=inc[x][nx2].second;
				if (adjacent(a,b)) continue;
				for (int64 na=0; na<deg[a]; na++) {
					int64 c=inc[a][na].first, ac=inc[a][na].second;
					if (c==x || adjacent(x,c) || !adjacent(b,c)) continue;
					orbit[x][8]++;
					f_62 += (tri[ac]>0)?common3_get(TRIPLE(a,b,c)):0;
					f_53 += tri[xa]+tri[xb];
					f_51 += tri[ac]+common2_get(PAIR(c,b));
					f_50 += common_x[c]-2;
					f_49 += common_a[b]-2;
					f_38 += deg[x]-2;
					f_37 += deg[a]-2+deg[b]-2;
					f_36 += deg[c]-2;
				}
			}

			// x = orbit-11 (paw)
			for (int64 nx2=nx1+1;nx2<deg[x];nx2++) {
				int64 b=inc[x][nx2].first, xb=inc[x][nx2].second;
				if (!adjacent(a,b)) continue;
				for (int64 nx3=0;nx3<deg[x];nx3++) {
					int64 c=inc[x][nx3].first, xc=inc[x][nx3].second;
					if (c==a || c==b || adjacent(a,c) || adjacent(b,c)) continue;
					orbit[x][11]++;
					f_44 += tri[xc];
					f_33 += deg[x]-3;
					f_30 += deg[c]-1;
					f_26 += deg[a]-2+deg[b]-2;
				}
			}

			// x = orbit-10 (paw)
			for (int64 nx2=0;nx2<deg[x];nx2++) {
				int64 b=inc[x][nx2].first, xb=inc[x][nx2].second;
				if (!adjacent(a,b)) continue;
				for (int64 nb=0;nb<deg[b];nb++) {
					int64 c=inc[b][nb].first, bc=inc[b][nb].second;
					if (c==x || c==a || adjacent(a,c) || adjacent(x,c)) continue;
					orbit[x][10]++;
					f_52 += common_a[c]-1;
					f_43 += tri[bc];
					f_32 += deg[b]-3;
					f_29 += deg[c]-1;
					f_25 += deg[a]-2;
				}
			}

			// x = orbit-9 (paw)
			for (int64 na1=0;na1<deg[a];na1++) {
				int64 b=inc[a][na1].first, ab=inc[a][na1].second;
				if (b==x || adjacent(x,b)) continue;
				for (int64 na2=na1+1;na2<deg[a];na2++) {
					int64 c=inc[a][na2].first, ac=inc[a][na2].second;
					if (c==x || !adjacent(b,c) || adjacent(x,c)) continue;
					orbit[x][9]++;
					f_56 += (tri[ab]>1 && tri[ac]>1)?common3_get(TRIPLE(a,b,c)):0;
					f_45 += common2_get(PAIR(b,c))-1;
					f_39 += tri[ab]-1+tri[ac]-1;
					f_31 += deg[a]-3;
					f_28 += deg[x]-1;
					f_24 += deg[b]-2+deg[c]-2;
				}
			}

			// x = orbit-4 (path)
			for (int64 na=0;na<deg[a];na++) {
				int64 b=inc[a][na].first, ab=inc[a][na].second;
				if (b==x || adjacent(x,b)) continue;
				for (int64 nb=0;nb<deg[b];nb++) {
					int64 c=inc[b][nb].first, bc=inc[b][nb].second;
					if (c==a || adjacent(a,c) || adjacent(x,c)) continue;
					orbit[x][4]++;
					f_35 += common_a[c]-1;
					f_34 += common_x[c];
					f_27 += tri[bc];
					f_18 += deg[b]-2;
					f_16 += deg[x]-1;
					f_15 += deg[c]-1;
				}
			}

			// x = orbit-5 (path)
			for (int64 nx2=0;nx2<deg[x];nx2++) {
				int64 b=inc[x][nx2].first, xb=inc[x][nx2].second;
				if (b==a || adjacent(a,b)) continue;
				for (int64 nb=0;nb<deg[b];nb++) {
					int64 c=inc[b][nb].first, bc=inc[b][nb].second;
					if (c==x || adjacent(a,c) || adjacent(x,c)) continue;
					orbit[x][5]++;
					f_17 += deg[a]-1;
				}
			}

			// x = orbit-6 (claw)
			for (int64 na1=0;na1<deg[a];na1++) {
				int64 b=inc[a][na1].first, ab=inc[a][na1].second;
				if (b==x || adjacent(x,b)) continue;
				for (int64 na2=na1+1; na2<deg[a]; na2++) {
					int64 c=inc[a][na2].first, ac=inc[a][na2].second;
					if (c==x || adjacent(x,c) || adjacent(b,c)) continue;
					orbit[x][6]++;
					f_22 += deg[a]-3;
					f_20 += deg[x]-1;
					f_19 += deg[b]-1+deg[c]-1;
				}
			}

			// x = orbit-7 (claw)
			for (int64 nx2=nx1+1;nx2<deg[x];nx2++) {
				int64 b=inc[x][nx2].first, xb=inc[x][nx2].second;
				if (adjacent(a,b)) continue;
				for (int64 nx3=nx2+1;nx3<deg[x];nx3++) {
					int64 c=inc[x][nx3].first, xc=inc[x][nx3].second;
					if (adjacent(a,c) || adjacent(b,c)) continue;
					orbit[x][7]++;
					f_23 += deg[x]-3;
					f_21 += deg[a]-1+deg[b]-1+deg[c]-1;
				}
			}
		}

		// solve equations
		orbit[x][72] = K5[x];
		orbit[x][71] = (f_71-12*orbit[x][72])/2;
		orbit[x][70] = (f_70-4*orbit[x][72]);
		orbit[x][69] = (f_69-2*orbit[x][71])/4;
		orbit[x][68] = (f_68-2*orbit[x][71]);
		orbit[x][67] = (f_67-12*orbit[x][72]-4*orbit[x][71]);
		orbit[x][66] = (f_66-12*orbit[x][72]-2*orbit[x][71]-3*orbit[x][70]);
		orbit[x][65] = (f_65-3*orbit[x][70])/2;
		orbit[x][64] = (f_64-2*orbit[x][71]-4*orbit[x][69]-1*orbit[x][68]);
		orbit[x][63] = (f_63-3*orbit[x][70]-2*orbit[x][68]);
		orbit[x][62] = (f_62-1*orbit[x][68])/2;
		orbit[x][61] = (f_61-4*orbit[x][71]-8*orbit[x][69]-2*orbit[x][67])/2;
		orbit[x][60] = (f_60-4*orbit[x][71]-2*orbit[x][68]-2*orbit[x][67]);
		orbit[x][59] = (f_59-6*orbit[x][70]-2*orbit[x][68]-4*orbit[x][65]);
		orbit[x][58] = (f_58-4*orbit[x][72]-2*orbit[x][71]-1*orbit[x][67]);
		orbit[x][57] = (f_57-12*orbit[x][72]-4*orbit[x][71]-3*orbit[x][70]-1*orbit[x][67]-2*orbit[x][66]);
		orbit[x][56] = (f_56-2*orbit[x][65])/3;
		orbit[x][55] = (f_55-2*orbit[x][71]-2*orbit[x][67])/3;
		orbit[x][54] = (f_54-3*orbit[x][70]-1*orbit[x][66]-2*orbit[x][65])/2;
		orbit[x][53] = (f_53-2*orbit[x][68]-2*orbit[x][64]-2*orbit[x][63]);
		orbit[x][52] = (f_52-2*orbit[x][66]-2*orbit[x][64]-1*orbit[x][59])/2;
		orbit[x][51] = (f_51-2*orbit[x][68]-2*orbit[x][63]-4*orbit[x][62]);
		orbit[x][50] = (f_50-1*orbit[x][68]-2*orbit[x][63])/3;
		orbit[x][49] = (f_49-1*orbit[x][68]-1*orbit[x][64]-2*orbit[x][62])/2;
		orbit[x][48] = (f_48-4*orbit[x][71]-8*orbit[x][69]-2*orbit[x][68]-2*orbit[x][67]-2*orbit[x][64]-2*orbit[x][61]-1*orbit[x][60]);
		orbit[x][47] = (f_47-3*orbit[x][70]-2*orbit[x][68]-1*orbit[x][66]-1*orbit[x][63]-1*orbit[x][60]);
		orbit[x][46] = (f_46-3*orbit[x][70]-2*orbit[x][68]-2*orbit[x][65]-1*orbit[x][63]-1*orbit[x][59]);
		orbit[x][45] = (f_45-2*orbit[x][65]-2*orbit[x][62]-3*orbit[x][56]);
		orbit[x][44] = (f_44-1*orbit[x][67]-2*orbit[x][61])/4;
		orbit[x][43] = (f_43-2*orbit[x][66]-1*orbit[x][60]-1*orbit[x][59])/2;
		orbit[x][42] = (f_42-2*orbit[x][71]-4*orbit[x][69]-2*orbit[x][67]-2*orbit[x][61]-3*orbit[x][55]);
		orbit[x][41] = (f_41-2*orbit[x][71]-1*orbit[x][68]-2*orbit[x][67]-1*orbit[x][60]-3*orbit[x][55]);
		orbit[x][40] = (f_40-6*orbit[x][70]-2*orbit[x][68]-2*orbit[x][66]-4*orbit[x][65]-1*orbit[x][60]-1*orbit[x][59]-4*orbit[x][54]);
		orbit[x][39] = (f_39-4*orbit[x][65]-1*orbit[x][59]-6*orbit[x][56])/2;
		orbit[x][38] = (f_38-1*orbit[x][68]-1*orbit[x][64]-2*orbit[x][63]-1*orbit[x][53]-3*orbit[x][50]);
		orbit[x][37] = (f_37-2*orbit[x][68]-2*orbit[x][64]-2*orbit[x][63]-4*orbit[x][62]-1*orbit[x][53]-1*orbit[x][51]-4*orbit[x][49]);
		orbit[x][36] = (f_36-1*orbit[x][68]-2*orbit[x][63]-2*orbit[x][62]-1*orbit[x][51]-3*orbit[x][50]);
		orbit[x][35] = (f_35-1*orbit[x][59]-2*orbit[x][52]-2*orbit[x][45])/2;
		orbit[x][34] = (f_34-1*orbit[x][59]-2*orbit[x][52]-1*orbit[x][51])/2;
		orbit[x][33] = (f_33-1*orbit[x][67]-2*orbit[x][61]-3*orbit[x][58]-4*orbit[x][44]-2*orbit[x][42])/2;
		orbit[x][32] = (f_32-2*orbit[x][66]-1*orbit[x][60]-1*orbit[x][59]-2*orbit[x][57]-2*orbit[x][43]-2*orbit[x][41]-1*orbit[x][40])/2;
		orbit[x][31] = (f_31-2*orbit[x][65]-1*orbit[x][59]-3*orbit[x][56]-1*orbit[x][43]-2*orbit[x][39]);
		orbit[x][30] = (f_30-1*orbit[x][67]-1*orbit[x][63]-2*orbit[x][61]-1*orbit[x][53]-4*orbit[x][44]);
		orbit[x][29] = (f_29-2*orbit[x][66]-2*orbit[x][64]-1*orbit[x][60]-1*orbit[x][59]-1*orbit[x][53]-2*orbit[x][52]-2*orbit[x][43]);
		orbit[x][28] = (f_28-2*orbit[x][65]-2*orbit[x][62]-1*orbit[x][59]-1*orbit[x][51]-1*orbit[x][43]);
		orbit[x][27] = (f_27-1*orbit[x][59]-1*orbit[x][51]-2*orbit[x][45])/2;
		orbit[x][26] = (f_26-2*orbit[x][67]-2*orbit[x][63]-2*orbit[x][61]-6*orbit[x][58]-1*orbit[x][53]-2*orbit[x][47]-2*orbit[x][42]);
		orbit[x][25] = (f_25-2*orbit[x][66]-2*orbit[x][64]-1*orbit[x][59]-2*orbit[x][57]-2*orbit[x][52]-1*orbit[x][48]-1*orbit[x][40])/2;
		orbit[x][24] = (f_24-4*orbit[x][65]-4*orbit[x][62]-1*orbit[x][59]-6*orbit[x][56]-1*orbit[x][51]-2*orbit[x][45]-2*orbit[x][39]);
		orbit[x][23] = (f_23-1*orbit[x][55]-1*orbit[x][42]-2*orbit[x][33])/4;
		orbit[x][22] = (f_22-2*orbit[x][54]-1*orbit[x][40]-1*orbit[x][39]-1*orbit[x][32]-2*orbit[x][31])/3;
		orbit[x][21] = (f_21-3*orbit[x][55]-3*orbit[x][50]-2*orbit[x][42]-2*orbit[x][38]-2*orbit[x][33]);
		orbit[x][20] = (f_20-2*orbit[x][54]-2*orbit[x][49]-1*orbit[x][40]-1*orbit[x][37]-1*orbit[x][32]);
		orbit[x][19] = (f_19-4*orbit[x][54]-4*orbit[x][49]-1*orbit[x][40]-2*orbit[x][39]-1*orbit[x][37]-2*orbit[x][35]-2*orbit[x][31]);
		orbit[x][18] = (f_18-1*orbit[x][59]-1*orbit[x][51]-2*orbit[x][46]-2*orbit[x][45]-2*orbit[x][36]-2*orbit[x][27]-1*orbit[x][24])/2;
		orbit[x][17] = (f_17-1*orbit[x][60]-1*orbit[x][53]-1*orbit[x][51]-1*orbit[x][48]-1*orbit[x][37]-2*orbit[x][34]-2*orbit[x][30])/2;
		orbit[x][16] = (f_16-1*orbit[x][59]-2*orbit[x][52]-1*orbit[x][51]-2*orbit[x][46]-2*orbit[x][36]-2*orbit[x][34]-1*orbit[x][29]);
		orbit[x][15] = (f_15-1*orbit[x][59]-2*orbit[x][52]-1*orbit[x][51]-2*orbit[x][45]-2*orbit[x][35]-2*orbit[x][34]-2*orbit[x][27]);
	}
	endTime = clock();
	printf("\t%.2f sec\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
}



#include <cstring>

enum {NODE, EDGE};
class parameter {
	public:
		int64 type; // graphlet type, node/edge
		int64 k; // graphlet size, {4,5}
		int64 threads;
		int64 verbose;
		parameter() {
			type = NODE;
			k = 4;
			threads = omp_get_max_threads();
			verbose = 0;
		}
};

void exit_with_help() {
	printf(
			"Usage: glet [options] input_filename [output_filename]\n"
			"options:\n"
			"    -t type       set type of graphlet node/edge (default node)\n"
			"    -k size       size of graphlets to compute (default 4)\n"
			"    -w threads    number of threads (default max)\n"
			"    -v verbose    show information or not (default 0)\n\n"
			"Example: ./glet -k 4 -t node test.edges test.4-node-counts\n"
	);
	exit(1);
}

char input_file_name[1024], output_file_name[1024];
string input_filename, output_filename, filename_only;
parameter parse_command_line(int argc, char **argv) {
	parameter param;   // default values have been set by the constructor

	int64 i;
	/// parse options
	for(i=1;i<argc;i++) {
		if(argv[i][0] != '-') break;
		if(++i>=argc) exit_with_help();
		switch(argv[i-1][1]) {
			case 't': {
				string type_str = string(argv[i]);
				param.type = NODE; // default
				if (type_str=="edge") param.type = EDGE;
				break;
			}
			case 'k': // graphlet size
				param.k = atoi(argv[i]);
				break;
			case 'w':
				param.threads = atoi(argv[i]);
				break;
			case 'v':
				param.verbose = atoi(argv[i]);
				break;
			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}
	omp_set_num_threads(param.threads);

	/// determine filenames
	if(i>=argc) exit_with_help();
	strcpy(input_file_name, argv[i]);
	if(i<argc-1) strcpy(output_file_name,argv[i+1]);
	else {
		char *p = argv[i]+ strlen(argv[i])-1;
		while (*p == '/') *p-- = 0;
		p = strrchr(argv[i],'/');
		if(p==NULL) p = argv[i]; else ++p;
		sprintf(output_file_name,"%s.model",p);
	}
	input_filename = string(input_file_name);
	output_filename = string(output_file_name);
	return param;
}



//./glet test-graphs/tailed-tri-1edge.edges output/tailed-tri-1edge.node_counts 4 0
int64 init(parameter p) {
	/// set params
	GRAPHLET_SIZE = p.k; // graphlet size, valid sizes: 4 or 5
	GRAPHLET_TYPE = p.type; // graphlet type

	/// open input/output files
	fin.open(input_file_name, fstream::in);
	fout.open(output_file_name, fstream::out | fstream::binary);
	if (fin.fail()) { cerr << "Failed to open INPUT file " << input_file_name << endl; return 0; }
	if (fout.fail()) { cerr << "Failed to open OUTPUT file " << output_file_name << endl; return 0; }
	return read_graph(input_file_name);
}



/**
 * @param k: 	 graphlet size, e.g., k-node graphlets where k=4 or 5 nodes
 * @param num_ele: number of graph elements, i.e., number of nodes (n)
 */
void write_node_results(int64 k=5, bool write_node_first=true, string delim=",") {
	int64 num_neg_counts=0;
	int64 no[] = {0,0,1,4,15,73}; // nodes

	//Node graphlet counts:
	// degree, 2-star-edge (P_3), 2-star-center (P_3), triangle (K_3),
	// 4-path-edge (P_4), 4-path-center (P_4), 3-star-edge (claw-edge), 3-star-center (claw-center)
	// 4-cycle (C_4), tailed-tri-tailEdge (paw-tailEdge), tailed-tri-edge (paw-edge), tailed-tri-center (paw-center)
	// chordal-cycle-edge (diamond-edge), chordal-cycle-center (diamond-center), 4-clique (K_4)


	for (int64 i=0; i<n; i++) {
		int64 v = i;
		if (fix_start_idx==true) {
			v=v+1;
		}

		if (n<20) { cout << "v=" << (fix_start_idx ? i+1:i) << ": "; } // since if vertex ids were indeed fixed, then they all have been deincremented by 1
		if (write_node_first) { fout << v << delim; }
		for (int64 j=0; j<no[k]; j++) {
//			if (orbit[i][j]<0) { cout<<"ERROR: Negative graphlet count.  (node="<<(fix_start_idx ? i+1:i)<<", graphlet="<<(j+1)<<", value="<<orbit[i][j]<<")"<<endl; }
			if (j!=0) fout << delim;
			if (orbit[i][j]<0) {
				num_neg_counts++;
				orbit[i][j] = abs(orbit[i][j]);
//				cout<<"ERROR: Negative graphlet count.  (edge="<<(i)<<", graphlet="<<(j+1)<<", value="<<orbit[i][j]<<")"<<endl;
			}
			fout << orbit[i][j];
			if (n<20) cout << orbit[i][j] << "\t";
		}
		fout << endl;
		if (n<20) cout <<endl;
	}
	fout.close();
	if (num_neg_counts>0) cout << "" << num_neg_counts << " negative graphlet counts." <<endl;
}

/**
 * Write edge, then graphlet counts... For example: v, u, g_1, ..., g_k
 * @param k: 	 graphlet size, e.g., k-node graphlets where k=4 or 5 nodes
 * @param num_ele: number of graph elements, i.e., number of edges (m)
 *
 * 4-edge orbit names:
 *------------------------
 * 2-stars (P_3), triangles (K_3), 
 * 4-path-edge (P_4), 4-path-center (P_4), 3-star (claw), 4-cycle (C_4), 
 * tailed-tri-tailEdge (paw-tailEdge), tailed-tri-edge (paw-edge), tailed-tri-center (paw-center),
 * chordal-cycle-edge (diamond-edge), chordal-cycle-center (diamond-center), 4-clique (K_4)
 * 
 */
void write_edge_results(int64 k=5, bool write_edge_first=true, string delim=",") {
	//	int64 no[] = {0,0,1,4,15,73}; // 12, 68 edge graphlets
	int64 num_neg_counts = 0;
	int64 no[] = {0,0,1,4,12,68}; // edge graphlets
	for (int64 i=0; i<m; i++) {

		int64 v=edges[i].a, u=edges[i].b;
		if (fix_start_idx==true) {
			v=v+1;
			u=u+1;
		}

		if (i<20 || n<20) { cout << "e=" << (i) << " ("<< v << "," << u << "): "; }

		if (write_edge_first==true) fout << v << delim << u << delim;
		for (int64 j=0; j<no[k]; j++) {
			if (orbit[i][j]<0) {
				num_neg_counts++;
				cout<<"ERROR: Negative graphlet count. (setting to 0)  (edge="<<(i)<<", graphlet="<<(j+1)<<", value="<<orbit[i][j]<<")"<<endl;
				// orbit[i][j] = abs(orbit[i][j]);
				orbit[i][j] = 0;
				
			}
			if (j!=0) fout << delim;
			fout << orbit[i][j];
			if (n<20) cout << orbit[i][j] << "\t";
		}
		fout << endl;
		if (n<20) cout <<endl;
		if (num_neg_counts>0) cout << "" << num_neg_counts << " negative graphlet counts." <<endl;
	}
	fout.close();
}

/* Write edge list to file (to ensure exact mapping of ids) */
void write_edge_list(string delim=",", string output_edgelist_fn="test.edgelist") {

	cout << "initial/before output filename: " << output_edgelist_fn <<endl; 
	fstream fout_edges; // input and output files, note fout_edges is for outputting the edge list (in addition to the node graphlet orbit counts)
	if (output_edgelist_fn=="test.edgelist") {
		output_edgelist_fn = string(output_filename);
		cout << "output filename: " << output_edgelist_fn <<endl; 
		output_edgelist_fn = get_filename_only(output_edgelist_fn) + ".edgelist";
	}
	cout << "[write_edge_list]  output filename: " << output_edgelist_fn <<endl; 
	

	fout_edges.open(output_edgelist_fn, fstream::out | fstream::binary);
	if (fout_edges.fail()) { cerr << "Failed to open OUTPUT file for EDGE LIST " << output_edgelist_fn << endl; exit(0); }

	for (int64 i=0; i<m; i++) {

		int64 v=edges[i].a, u=edges[i].b;
		if (fix_start_idx==true) { v=v+1; u=u+1; }

		fout_edges << v << delim << u << endl;
	}
	fout_edges.close();
}



int main(int argc, char *argv[]) {
	parameter param = parse_command_line(argc, argv);
	if (!init(param)) { cerr << "Stopping!" << endl; return 0; }

	double sec = tic();
	if (GRAPHLET_TYPE==0) { /// NODE graphlet counts
		if (GRAPHLET_SIZE==4) node_count4();
		if (GRAPHLET_SIZE==5) node_count5();
	}
	if (GRAPHLET_TYPE==1) { /// EDGE graphlet counts
		if (GRAPHLET_SIZE==4) edge_count4();
		if (GRAPHLET_SIZE==5) edge_count5();
	}
	toc(sec);
	cout << "TOTAL TIME: " << sec <<endl;
	// if (GRAPHLET_TYPE==0) { write_node_results(GRAPHLET_SIZE); } /// NODE graphlet counts
	// if (GRAPHLET_TYPE==1) { write_edge_results(GRAPHLET_SIZE); } /// EDGE graphlet counts

	bool write_ids_first = true;
	string delim = ",";
	if (GRAPHLET_TYPE==0) { 
		write_node_results(GRAPHLET_SIZE,write_ids_first,delim); 
		write_edge_list(delim); 
	} /// NODE graphlet counts
	if (GRAPHLET_TYPE==1) { write_edge_results(GRAPHLET_SIZE,write_ids_first,delim); } /// EDGE graphlet counts

	return 0;
}
