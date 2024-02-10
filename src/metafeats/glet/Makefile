CC = g++
CFLAGS = -O3 -std=c++0x
#CFLAGS = -O2 -std=c++0x
CFLAGS += -fopenmp 
CFLAGS += -ffast-math -funroll-loops -fno-strict-aliasing \
	 -fomit-frame-pointer -fexpensive-optimizations -funroll-loops \
	 -fmove-loop-invariants -fprefetch-loop-arrays -ftree-loop-optimize \
	 -ftree-vect-loop-version -ftree-vectorize
	 
#CC = g++
#CFLAGS = -O2 -std=c++0x
 
glet: glet.cpp
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -f glet glet.o
