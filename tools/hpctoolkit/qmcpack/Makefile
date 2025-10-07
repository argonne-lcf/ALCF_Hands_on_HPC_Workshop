default:
	@echo ---------------------------------------------------------------------------------------
	@echo This Makefile is for collecting, analyzing, and viewing performance data
	@echo for a prebuilt benchmark that is part of QMCPACK - a sophisticated quantum Monte Carlo
	@echo materials code that was developed as part of the Exascale Computing Project.
	@echo
	@echo The Makefile supports five commands:
	@echo "  run:       measure a 12-rank + 12 GPU tile execution of qmcpack using hpcrun"
	@echo "  analyze:   analyze binaries using hpcstruct and postprocess measurements with hpcprof"
	@echo "  view:      view the performance database produced by hpcprof"
	@echo "  clean:     remove qmcpack output and hpctoolkit measurements"
	@echo "  veryclean: clean + remove hpctoolkit database with analysis results"
	@echo ---------------------------------------------------------------------------------------

run:
	time ./measure_qmcpack.sh |& tee hpcrun.log

analyze:
	time hpcstruct -x libyaksa -x libmkl qmcpack.m |& tee hpcstruct.log  # skip some large unimportant libraries
	rm -rf qmcpack.db
	time hpcprof --dwarf-max-size=unlimited -o qmcpack.db qmcpack.m |& tee hpcprof.log # analyze all line map info 

view:
	hpcviewer qmcpack.db

clean:
	rm -rf *.xml *.out *.dat *.h5              # qmcpack files
	rm -rf qmcpack.m core.* *.log              # hpctoolkit measurement files

veryclean: clean
	rm -rf qmcpack.db                          # hpctoolkit result files
