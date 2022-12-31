# TODO: add CMake lists to subdirectories

#
# Specify all headers
#
set( tcscf_headers
	AtomicBasis.hpp
    caliperInterface.hpp
    HartreeFock.hpp
    HydrogenLikeBasis.hpp
    jastrowFunctions.hpp
    LvArrayInterface.hpp
    mathFunctions.hpp
    OchiBasis.hpp
    RAJAInterface.hpp
    setup.hpp
    orthogonalization.hpp
    SlaterTypeOrbital.hpp
    
    integration/changeOfVariables.hpp
    integration/ChebyshevGauss.hpp
    integration/Lebedev.hpp
    integration/qmcWrapper.hpp
    integration/quadrature.hpp
    integration/TreutlerAhlrichsLebedev.hpp
    integration/integrateAll.hpp
    
    integration/data/LebedevData_3.hpp
    integration/data/LebedevData_5.hpp
    integration/data/LebedevData_7.hpp
    integration/data/LebedevData_9.hpp
    integration/data/LebedevData_11.hpp
    integration/data/LebedevData_13.hpp
    integration/data/LebedevData_15.hpp
    integration/data/LebedevData_17.hpp
    integration/data/LebedevData_19.hpp
    integration/data/LebedevData_21.hpp
    integration/data/LebedevData_23.hpp
    integration/data/LebedevData_25.hpp
    integration/data/LebedevData_27.hpp
    integration/data/LebedevData_29.hpp
    integration/data/LebedevData_31.hpp
    integration/data/LebedevData_35.hpp
    integration/data/LebedevData_41.hpp
    integration/data/LebedevData_47.hpp
    integration/data/LebedevData_53.hpp
    integration/data/LebedevData_59.hpp
    integration/data/LebedevData_65.hpp
    integration/data/LebedevData_71.hpp
    integration/data/LebedevData_77.hpp
    integration/data/LebedevData_83.hpp
    integration/data/LebedevData_89.hpp
    integration/data/LebedevData_95.hpp
    integration/data/LebedevData_101.hpp
    integration/data/LebedevData_107.hpp
    integration/data/LebedevData_113.hpp
    integration/data/LebedevData_119.hpp
    integration/data/LebedevData_125.hpp
    integration/data/LebedevData_131.hpp
)

#
# Specify all sources
#
set( tcscf_sources
     caliperInterface.cpp
     HartreeFock.cpp
     setup.cpp
     SlaterTypeOrbital.cpp

     integration/Lebedev.cpp
)

set( dependencyList ${dependencyList} blas lapack gsl lvarray lvarraydense optionparser )

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
