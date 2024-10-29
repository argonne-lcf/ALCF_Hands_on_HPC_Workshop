CXX=CC
BINS=saxpy

all: $(BINS)

saxpy: saxpy.cu
	$(CXX) $(CXXFLAGS) -o saxpy saxpy.cu

clean:
	rm -f $(BINS)
