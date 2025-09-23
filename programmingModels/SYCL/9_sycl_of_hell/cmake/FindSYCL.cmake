include( FindPackageHandleStandardArgs )
include( CheckCXXCompilerFlag )
check_cxx_compiler_flag("-fsycl" CXX_HAS_FSYCL)

find_package_handle_standard_args( SYCL
  REQUIRED_VARS CXX_HAS_FSYCL
)

if( SYCL_FOUND AND NOT TARGET SYCL::SYCL )
  add_library( SYCL::SYCL INTERFACE IMPORTED )
  set_target_properties( SYCL::SYCL PROPERTIES
      INTERFACE_COMPILE_OPTIONS  "$<$<COMPILE_LANGUAGE:CXX>:-fsycl>"
      INTERFACE_LINK_OPTIONS     "-fsycl"
  )
endif()
