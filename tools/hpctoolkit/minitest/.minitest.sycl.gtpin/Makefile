CPPFLAGS=-DUSE_MPI
OPT=-g -O2 -fsycl
LIBS=-lm

SOURCES=src/single.cc src/syclgpu.cc

EXEC=minitest.sycl
EXEC_PATH=./$(EXEC)

MD=hpctoolkit-$(EXEC).m # measurement directory
DB=hpctoolkit-$(EXEC).db # database directory

CXX=mpicxx

default:
	@echo
	@echo ---------------------------------------------------------------------------------------
	@echo This Makefile is for building a simple program $(EXEC) and then collecting, analyzing,
	@echo and viewing performance data for it.
	@echo
	@echo IMPORTANT: you must be on a compute node to use the run target
	@echo
	@echo The Makefile supports five commands:
	@echo "  build:     compile the executable $(EXEC)"
	@echo "  run:       measure a 2-ranks + 2 GPU tile execution of $(EXEC) using hpcrun"
	@echo "  view:      view the performance database produced by hpcprof"
	@echo "  clean:     remove qmcpack output and hpctoolkit measurements"
	@echo "  veryclean: clean + remove hpctoolkit database with analysis results"
	@echo ---------------------------------------------------------------------------------------

build: $(EXEC_PATH)

run: $(DB)

view:
	@if [ -d $(DB)]; then echo hpcviewer $(DB); hpcviewer $(DB); \
	else echo You must first use \'make run\' on a compute node to build $(DB); fi 
	
$(DB): $(EXEC_PATH)
	mpiexec -np 2 hpcrun -o $(MD) -e CPUTIME -e gpu=level0,inst=count $(EXEC_PATH)
	hpcstruct $(MD)
	hpcprof -o $(DB) $(MD)

$(EXEC_PATH): $(SOURCES)
	$(CXX) $(CPPFLAGS) $(OPT) -o $@ $(SOURCES) $(LIBS)

clean:
	rm -rf $(MD) $(EXEC)

veryclean: clean
	rm -rf $(DB)

