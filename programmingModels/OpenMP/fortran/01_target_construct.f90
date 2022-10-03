program  main
  use omp_lib
  implicit none
  integer flag
  
  write(*,*) "Number of devices:", omp_get_num_devices()

  !$omp target map(from:flag)
    if( .not. omp_is_initial_device() ) then
      flag = 1
    else
      flag = 0
   endif
  !$omp end target

   if( flag == 1 ) then
      print *, "Hello world from accelerator"
   else
      print *, "Hello world from host"
   endif

 end program main
