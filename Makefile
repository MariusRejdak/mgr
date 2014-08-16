OBJ = bitonicSort.out mergeSort.out oddEvenMergeSort.out quickSort.out radixSort.out thrustSort.out

all: $(OBJ)

%.out: %.cu utils.h cuda_utils.h
	nvcc -arch=sm_21 -O2 -m64 $< -o $@

clean:
	rm *.out

test: all
	for file in $(OBJ); do \
		echo ""; \
		echo "Testing $$file"; \
    	bash -c "time ./$$file"; \
	done
