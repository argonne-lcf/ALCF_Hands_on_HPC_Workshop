# this assumes that the NVIDIA compiler is used

CXX = CC
CXXOFFLOAD = -mp=gpu -gpu=cc80
EXE = 01_target_construct 02_target_teams_parallel 03_map

all : $(EXE)

% : %.cpp
	$(CXX) $(CXXFLAGS) $(CXXOFFLOAD) $< -o $@

clean:
	rm -f $(EXE)
