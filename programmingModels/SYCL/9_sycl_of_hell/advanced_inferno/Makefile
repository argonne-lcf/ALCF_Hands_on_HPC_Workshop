CXX ?= dpcpp
CXXFLAGS ?= -std=c++17

SRCS := $(wildcard *.cpp)
PRGS := $(patsubst %.cpp,%,$(SRCS))
RUN :=  $(addprefix run_, $(PRGS))

.PHONY: run_all
run_all: $(sort $(RUN))

.PHONY: all
all: $(PRGS)

%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

9_matrix_mul_mkl: 9_matrix_mul_mkl.cpp
	-$(CXX) -o $@  $^ $(CXXFLAGS) -DMKL_ILP64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core

run_%:%
	./$^

.PHONY: clean
clean:
	rm -f $(PRGS)
