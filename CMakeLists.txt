cmake_minimum_required( VERSION 3.9 )

project( TCSCF LANGUAGES C CXX )

set( BLT_CXX_STD "c++17" CACHE STRING "Version of C++ standard" FORCE )
set( ENABLE_WARNINGS_AS_ERRORS "ON" CACHE PATH "" )

option( ENABLE_TESTS "Builds tests" ON )
option( ENABLE_LVARRAY_UNIT_TESTS "Builds LvArray unit tests." ON )
option( ENABLE_EXAMPLES "Builds LvArray examples" OFF )
option( ENABLE_BENCHMARKS "Builds LvArray benchmarks" OFF )
option( ENABLE_DOCS "Builds LvArray documentation" OFF )

option( ENABLE_ADDR2LINE "Enable addr2line usage in stacktraces" ON )
option( ENABLE_CUDA "Build with CUDA" OFF )
option( ENABLE_CALIPER "Build with Caliper" OFF )

if( NOT ENABLE_UMPIRE )
    message( FATAL_ERROR "Umpire must be enabled." )
endif()

if( NOT ENABLE_CHAI )
    message( FATAL_ERROR "CHAI must be enabled." )
endif()

if( NOT ENABLE_LVARRAY_UNIT_TESTS )
    # This only disables LvArray's unit tests.
    set( DISABLE_UNIT_TESTS ON )
endif()

include( LvArray/cmake/blt/SetupBLT.cmake )
include( LvArray/cmake/CMakeBasics.cmake )

################################
# RAJA, CHAI, Umpire and Caliper
################################
include( LvArray/cmake/SetupTPL.cmake )

# Add lvarray
add_subdirectory( LvArray )

# Add builtin TPL's (currently just optionparser)
add_subdirectory( thirdparty )

# Add source
add_subdirectory( src )
