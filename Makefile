$(SOURCES) = bitonicSort.cu mergeSort.cu oddEvenMergeSort.cu quickSort.cu radixSort.cu thrustSort.cu

all: bitonicSort.out oddEvenMergeSort.out quickSort.out radixSort.out thrustSort.out

%.out: %.cu
	nvcc -arch=sm_21 -O2 -m64 $< -o $@

clean:
	rm *.out

