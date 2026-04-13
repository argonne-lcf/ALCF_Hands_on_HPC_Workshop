// SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "minitest.h"

double **lptr;
double **rptr;
double **pptr;

/* Parameters governing the size of the test */
#define        N 40000000      /* size of the data arrays used */
#define        NITER 3         /* number of iterations performed by each thread */
#define        KKMAX 2000      /* number of iterations in kernel */

size_t nn = N;
int niter = NITER;
int kkmax = KKMAX;
int kscale = 1;
int omp_num_t;


int
main(int argc, char **argv, char **envp)
{
  /* determine thread count setting */
  char *s = getenv("OMP_NUM_THREADS");
  if (s != NULL) {
    int nthreads = atoi(s);
    if (nthreads != 1) {
      fprintf(stdout, " NOTE: single-thread executable ignored OMP_NUM_THREADS = %d\n", nthreads );
    }
  }

  /* setup_run -- parse the arguments, to reset N and NITER, as requested */
  setup_run(argc, argv);

  /* check number and accessibility of GPU devices */
  initgpu();

  /* set thread count to one */
  omp_num_t = 1;
  fprintf(stdout, "    [%d] This run: %d CPU thread%s; data size = %ld; %d iteration%s; kernel scale = %d\n\n",
    thispid, omp_num_t, (omp_num_t==1 ? "" : "s"), nn, niter, (niter==1? "":"s"), kscale );

  /* Allocate and initialize data */
  allocinitdata(omp_num_t);

  /* perform the number of iterations requested */
  fprintf(stdout, "  [%d] start %d iteration%s\n", thispid, niter, (niter ==1 ? "" : "s") );
  for (int k = 0; k < niter; k++) {
#if 0
    fprintf(stdout, "    [%d] start iteration %d\n", thispid, k);
#endif
    {
      twork(k, 0 );
    }
    mpisync();
#if 0
    fprintf(stdout, "  [%d] end     iteration %d\n", thispid, k);
#endif
  }
  fprintf(stdout, "  [%d] end %d iteration%s\n", thispid, niter,  (niter ==1 ? "" : "s") );

  /* write out various elements in each thread's result array */
  for (int k = 0; k < omp_num_t; k++) {
    output(k, pptr[k], nn, "result p array");
    checkdata(k, pptr[k], nn );
  }

  teardown_run();

  return 0;
}

#include "maincommon.cc"
