program testdgetrf

  implicit none

  integer, parameter :: M = 1000, N = M
  real(8), allocatable :: A(:,:), B(:)
  integer, allocatable :: IPIV(:)
  real(8), parameter :: eps = 1.0d-10
  real(8) :: rmx

  integer :: i, j, lda, ldb, info1, info2

  allocate(A(M,N), B(M), IPIV(M))

  call random_number(A)

  do concurrent (i = 1 : M)
     A(i,i) = A(i,i) * 10.0d0
     B(i) = sum(A(i,:))
  end do

  ! Factor and solve
  lda = M
  ldb = M

  call dgetrf(M, N, A, lda, IPIV, info1)
  call dgetrs('n', n, 1, A, lda, IPIV, B, ldb, info2)

  if ((info1 .ne. 0) .or. (info2 .ne. 0)) then
     print *, "Test FAILED", info1, info2
  else
     rmx = 0.0d0
     do concurrent (i = 1 : M) ! reduce(max:rmx)
        rmx = max(rmx, abs(B(i) - 1.0d0))
     end do
     if (rmx .gt. eps) then
        print *, "Test FAILED"
     else
        print *, "Test PASSED"
     end if
  end if

end program testdgetrf
