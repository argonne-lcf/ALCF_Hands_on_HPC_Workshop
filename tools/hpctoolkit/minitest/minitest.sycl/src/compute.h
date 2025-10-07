/*
 * SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//  This file is #include'd as the core of the computations on the GPU or CPU

#if 0
    // do nothing at all
#endif

#if 0
    // set the result to one in the kernel
    d_p1[i] = 1.;
#endif

#if 0
    // decrement the result by one in the kernel
    d_p1[i] = d_p1[i] -1.;
#endif

#if 0
      // use transcendental function in the kernel
#ifdef INFORTRAN
    do kk = 1, kernmax
       d_p1(i) = d_p1(i) + 1. + (sqrt( exp( log (d_l1(i)*d_l1(i)) ) + exp( log (d_r1(i)*d_r1(i)) ) ) ) / &
                                (sqrt( exp( log (d_l1(i)*d_r1(i)) ) + exp( log (d_r1(i)*d_l1(i)) ) ) )
    end do
#else
    for (int kk = 0 ; kk < kernmax ; kk++ ) {
      d_p1[i] = d_p1[i] + 1.+ (sqrt( exp( log (d_l1[i]*d_l1[i]) ) + exp( log (d_r1[i]*d_r1[i]) ) ) ) /
        ( sqrt (exp( log(d_l1[i]*d_r1[i]) ) + exp( log( (d_r1[i]*d_l1[i]) )) ) );
    }
#endif /* INFORTRAN */
#endif

#if 1
    // do a vector add in the kernel
#ifdef INFORTRAN
    do kk = 1, kernmax
       d_p1(i) = d_p1(i) + d_l1(nelements + 1 - kk) / real(kernmax, KIND=real64) + d_r1(kk) / real(kernmax, KIND=real64)
    end do
#else
    for (int kk = 0 ; kk < kernmax ; kk++ ) {
      d_p1[i] = d_p1[i] + d_l1[nelements - kk] / double(kernmax) + d_r1[kk] / double(kernmax);
    }
#endif /* INFORTRAN */
#endif
