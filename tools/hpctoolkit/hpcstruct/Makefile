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
	time hpcrun -e REALTIME -t -o hpcstruct.m hpcstruct --nocache -j 16 `which hpcstruct` |& tee hpcrun.log
	time hpcstruct hpcstruct.m |& tee hpcstruct.log
	time hpcprof -o hpcstruct.db hpcstruct.m |& tee hpcprof.log

view:
	hpcviewer hpcstruct.db

clean:
	rm -rf hpcstruct.m hpcstruct.hpcstruct core.* *.log   # hpctoolkit measurement files

veryclean: clean
	rm -rf hpcstruct.db                                  # hpctoolkit result files
