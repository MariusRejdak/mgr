#
# Makefile
# Author: Marius Rejdak
#


OBJ = bitonicSort.out mergeSort.out oddEvenMergeSort.out quickSort.out radixSort.out thrustSort.out stlSort.out

all: $(OBJ)

%.out: %.cu utils.h cuda_utils.h
	nvcc -arch=sm_21 -O2 -m64 $< -o $@

%.out: %.cpp utils.h
	g++ -O2 -m64 $< -o $@

clean:
	rm -f *.out *.csv

test:
	for file in $(OBJ); do \
		echo "Generate $$file"; \
    	bash -c "./$$file > $$file.csv"; \
	done
