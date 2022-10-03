
CXX = clang++
CXXOFFLOAD = -fopenmp -fopenmp-targets=nvptx64
CXXFLAGS = -Wno-unknown-cuda-version
EXE = map map_function map_unstructured_function

all : $(EXE)

% : %.cpp
	$(CXX) $(CXXFLAGS) $(CXXOFFLOAD) $< -o $@

clean:
	rm -f $(EXE)
