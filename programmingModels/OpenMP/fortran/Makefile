# this assumes that the IBM XL compiler is used

CXX = ftn
CXXOFFLOAD = -mp=gpu -gpu=cc80
CXXFLAGS = 
EXE = 01_target_construct 02_target_teams_parallel 03_map

all : $(EXE)

% : %.f90
	$(CXX) $(CXXFLAGS) $(CXXOFFLOAD) $< -o $@

clean:
	rm -f $(EXE) *.mod
