set( dependencyList portedKernels gtest )

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
     portedKernelsTest_common.cpp
     portedKernelsTest_UniformRandomReal.cpp
     portedKernelsTest_FieldPair.cpp
     portedKernelsTest_compression.cpp )

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

#
# Add mpi tests
#
set( portedKernel_mpi_test_sources
     portedKernelsTest_DistributedMesh.cpp
     portedKernelsTest_propagatorFunctions3D.cpp
     portedKernelsTest_interpolation.cpp
     portedKernelsTest_NonLinOp.cpp )

foreach( test ${portedKernel_mpi_test_sources} )
    get_filename_component( test_name ${test} NAME_WE ) 
    blt_add_executable( NAME ${test_name}
                        SOURCES ${test}
                        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                        DEPENDS_ON ${dependencyList} )

    blt_add_test( NAME ${test_name}
                  COMMAND ${test_name}
                  NUM_MPI_TASKS 4 )

     set( portedKernelTests ${portedKernelTests} ${test_name} )
endforeach()

set( portedKernelTests ${portedKernelTests} portedKernelsTest_DistributedMesh )

#
# Add benchmarking executables.
#
set( portedKernel_benchmark_sources
     portedKernelsTest_benchmarkCommunication.cpp )

foreach( test ${portedKernel_benchmark_sources} )
    get_filename_component( test_name ${test} NAME_WE ) 
    blt_add_executable( NAME ${test_name}
                        SOURCES ${test}
                        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                        DEPENDS_ON ${dependencyList} )

     set( portedKernelTests ${portedKernelTests} ${test_name} )
endforeach()

#
# Add TPL test which doesn't depend on portedKernels
#
list(REMOVE_ITEM dependencyList portedKernels)

blt_add_executable( NAME portedKernelsTest_TPL
                    SOURCES portedKernelsTest_TPL.cpp
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON ${dependencyList} RAJA chai umpire adiak caliper fftw mpi zfp::zfp )

blt_add_test( NAME portedKernelsTest_TPL
              COMMAND portedKernelsTest_TPL )

set( portedKernelTests ${portedKernelTests} portedKernelsTest_TPL )

add_custom_target( portedKernelsTest
                   DEPENDS ${portedKernelTests} )
