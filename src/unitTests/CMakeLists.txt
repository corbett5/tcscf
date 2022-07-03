set( dependencyList tcscf gtest )

if( ENABLE_OPENMP )
     set( dependencyList ${dependencyList} openmp )
endif()

if( ENABLE_CUDA )
     set( dependencyList ${dependencyList} cuda )
endif()


#
# Add the standard tests
#
set( portedKernel_test_sources
     blasLapackInterfaceTest.cpp
     AtomicBasisTest.cpp
     HartreeFockTest.cpp )

set( portedKernelTests "" )
foreach( test ${portedKernel_test_sources} )
    get_filename_component( test_name ${test} NAME_WE ) 
    blt_add_executable( NAME ${test_name}
                        SOURCES ${test}
                        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                        DEPENDS_ON ${dependencyList} )

    blt_add_test( NAME ${test_name}
                  COMMAND ${test_name} )

     set( portedKernelTests ${portedKernelTests} ${test_name} )
endforeach()

add_custom_target( tcscfTest
                   DEPENDS ${portedKernelTests} )
