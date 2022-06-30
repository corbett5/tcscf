message("adding optionparser library")
    
blt_add_library( NAME     optionparser
                 HEADERS  src/optionparser.h
                 )
                 
target_include_directories( optionparser SYSTEM INTERFACE ${CMAKE_SOURCE_DIR}/thirdparty/optionparser/src )
                 