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


macro(find_and_register)
    set(singleValueArgs NAME HEADER)
    set(multiValueArgs INCLUDE_DIRECTORIES
                       LIBRARY_DIRECTORIES
                       LIBRARIES
                       EXTRA_LIBRARIES
                       DEPENDS )

    ## parse the arguments
    cmake_parse_arguments(arg
                          "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT DEFINED arg_NAME)
        message(FATAL_ERROR "The find_and_register required parameter NAME specifies the name of the library to register.")
    endif()

    if(NOT DEFINED arg_INCLUDE_DIRECTORIES)
        message(FATAL_ERROR "The find_and_register required parameter INCLUDE_DIRECTORIES specifies the directories to search for the given header.")
    endif()

    if(NOT DEFINED arg_LIBRARY_DIRECTORIES)
        message(FATAL_ERROR "The find_and_register required parameter LIBRARY_DIRECTORIES specifies the directories to search for the given libraries.")
    endif()

    if(NOT DEFINED arg_HEADER)
        message(FATAL_ERROR "The find_and_register required parameter HEADER specifies the header to search for.")
    endif()

    if(NOT DEFINED arg_LIBRARIES)
        message(FATAL_ERROR "The find_and_register required parameter LIBRARIES specifies the libraries to search for.")
    endif()

    find_path(${arg_NAME}_INCLUDE_DIR ${arg_HEADER}
              PATHS ${arg_INCLUDE_DIRECTORIES}
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH)

    if(${arg_NAME}_INCLUDE_DIR STREQUAL ${arg_NAME}_INCLUDE_DIR-NOTFOUND)
        message(FATAL_ERROR "Could not find '${arg_HEADER}' in '${arg_INCLUDE_DIRECTORIES}'")
    endif()

    blt_find_libraries(FOUND_LIBS ${arg_NAME}_LIBRARIES
                       NAMES ${arg_LIBRARIES}
                       PATHS ${arg_LIBRARY_DIRECTORIES}
                       REQUIRED ON)

    blt_import_library(NAME ${arg_NAME}
                         INCLUDES ${${arg_NAME}_INCLUDE_DIR}
                         LIBRARIES ${${arg_NAME}_LIBRARIES} ${arg_EXTRA_LIBRARIES}
                         TREAT_INCLUDES_AS_SYSTEM ON
                         DEPENDS_ON ${arg_DEPENDS})

endmacro(find_and_register)

################################
# GSL
################################
message( STATUS "Using GSL from ${GSL_DIR}" )

find_and_register( NAME gsl
                   INCLUDE_DIRECTORIES ${GSL_DIR}/include
                   LIBRARY_DIRECTORIES ${GSL_DIR}/lib
                   HEADER gsl/gsl_math.h
                   LIBRARIES gsl )

################################
# BLAS and LAPACK
################################
if( NOT DEFINED BLAS_LIBRARIES )
    find_package( BLAS REQUIRED )
endif()

if( NOT DEFINED LAPACK_LIBRARIES )
    find_package( LAPACK REQUIRED )
endif()

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
