CPPFLAGS=-DUSE_MPI
OPT=-g -O2 -fsycl
LIBS=-lm

SOURCES=src/single.cc src/syclgpu.cc

HPCRUN_PC=/soft/perftools/hpctoolkit/.install/2025-05-incite/hpctoolkit/branches/pcsampling-preview/installdir-db7e75c/bin/hpcrun
LP=LD_LIBRARY_PATH=/soft/perftools/hpctoolkit/.install/2025-05-incite/hpctoolkit/branches/pcsampling-preview/installdir-db7e75c/lib64:$(LD_LIBRARY_PATH)

EXEC=minitest.sycl
EXEC_PATH=./$(EXEC)

MD=hpctoolkit-$(EXEC).pc.m # measurement directory
DB=hpctoolkit-$(EXEC).pc.db # database directory

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
	$(LP) mpiexec -np 1 $(HPCRUN_PC) -o $(MD) -e CPUTIME -e gpu=level0,pc $(EXEC_PATH)
	hpcstruct -x libyaksa $(MD)
	hpcprof -o $(DB) $(MD)


$(EXEC_PATH): $(SOURCES)
	$(CXX) $(CPPFLAGS) $(OPT) -o $@ $(SOURCES) $(LIBS)

clean:
	rm -rf $(MD) $(EXEC)

veryclean: clean
	rm -rf $(DB)

