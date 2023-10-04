program  main
  use omp_lib
  implicit none
  integer i
!$omp target teams distribute parallel do simd
    do i=1,10000
       write(*,*) "Thread", omp_get_thread_num(), &
            "out of", omp_get_num_threads() ,&
            "threads in team", omp_get_team_num(), &
            "out of", omp_get_num_teams(), &
            "teams is using index" , i
     end do
!$omp end target teams distribute parallel do simd
end program main
