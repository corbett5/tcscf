#
# Specify all headers
#
set( tcscf_headers
	AtomicBasis.hpp
    blasLapackInterface.hpp
    caliperInterface.hpp
    HartreeFock.hpp
    HydrogenLikeBasis.hpp
    LvArrayInterface.hpp
    mathFunctions.hpp
    OchiBasis.hpp
    qmcWrapper.hpp
    setup.hpp
    quadrature.hpp
)

#
# Specify all sources
#
set( tcscf_sources
     blasLapackInterface.cpp
     caliperInterface.cpp
     HartreeFock.cpp
     setup.cpp
     quadrature.cpp
)

set( dependencyList ${dependencyList} blas lapack boost gsl adiak lvarray optionparser )

if( ENABLE_OPENMP )
    set( dependencyList ${dependencyList} openmp )
endif()

if( ENABLE_CUDA )
    set( dependencyList ${dependencyList} cuda )
endif()

blt_add_library( NAME tcscf
                 SOURCES ${tcscf_sources}
                 HEADERS ${tcscf_headers}
                 DEPENDS_ON ${dependencyList} )

# This is necessary when LvArray isn't installed.
get_target_property( lvarray_SOURCE_DIR lvarray SOURCE_DIR )

target_include_directories( tcscf PUBLIC ${lvarray_SOURCE_DIR} )

target_include_directories( tcscf SYSTEM PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../qmc/src )

add_subdirectory( unitTests )
