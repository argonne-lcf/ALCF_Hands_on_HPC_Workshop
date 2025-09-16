program  main
  use omp_lib
  implicit none

  print *, "Number of devices:", omp_get_num_devices()

  !$omp target
   if( .not. omp_is_initial_device() ) then
     print *, "Hello world from accelerator"
   else
     print *, "Hello world from host"
  endif
  !$omp end target

end program main
