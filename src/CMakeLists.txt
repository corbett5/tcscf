#
# Specify all headers
#
set( portedKernels_headers
     caliperInterface.hpp
     common.hpp
     config.hpp
     FieldPair.hpp
     LvArrayInterface.hpp
     NonLinOp_Prop3DAcoIsoDenQ_DEO2_FDTD.hpp
     NonLinOp_Prop3DAcoTTIDenQ_DEO2_FDTD.hpp
     NonLinOp_Prop3DAcoVTIDenQ_DEO2_FDTD.hpp
     NonLinOp.hpp
     propagatorFunctions3D.hpp
     RAJAInterface.hpp
     setup.hpp
     stencilDerivatives.hpp
     VTI_TTI_Common.hpp
     zfpInterface.hpp
     mpi.hpp
     DistributedMesh.hpp
     Interpolation.hpp
)

#
# Specify all sources
#
set( portedKernels_sources
     caliperInterface.cpp
     NonLinOp_Prop3DAcoIsoDenQ_DEO2_FDTD.cpp
     NonLinOp_Prop3DAcoTTIDenQ_DEO2_FDTD.cpp
     NonLinOp_Prop3DAcoVTIDenQ_DEO2_FDTD.cpp
     NonLinOp.cpp
     setup.cpp
     mpi.cpp
)

set( dependencyList ${dependencyList} mpi adiak zfp::zfp lvarray optionparser )

if( ENABLE_OPENMP )
    set( dependencyList ${dependencyList} openmp )
endif()

if( ENABLE_CUDA )
    set( dependencyList ${dependencyList} cuda )
endif()

blt_add_library( NAME portedKernels
                 SOURCES ${portedKernels_sources}
                 HEADERS ${portedKernels_headers}
                 DEPENDS_ON ${dependencyList} )

# This is necessary when LvArray isn't installed.
get_target_property( lvarray_SOURCE_DIR lvarray SOURCE_DIR )

target_include_directories( portedKernels PUBLIC ${lvarray_SOURCE_DIR} )

add_subdirectory( unitTests )
