#pragma once

#include "Macros.hpp"

#include <caliper/cali.h>

#include <memory>

/// Mark a function or scope for timing with a given name
#define TCSCF_MARK_SCOPE( name ) cali::Function __cali_ann##__LINE__( STRINGIZE_NX( name ) )

/// Mark a function for timing using a compiler-provided name
#define TCSCF_MARK_FUNCTION cali::Function __cali_ann##__func__( tcscf::internal::stripPF( __PRETTY_FUNCTION__ ).c_str() )

// Forward declaration of cali::ConfigManager.
namespace cali
{
class ConfigManager;
}

namespace tcscf
{
namespace internal
{

inline std::string stripPF( char const * prettyFunction )
{
  std::string const input( prettyFunction );
  std::string::size_type const end = input.find_first_of( '(' );
  std::string::size_type const beg = input.find_last_of( ' ', end)+1;
  return input.substr( beg, end-beg );
}

} // namespace internal;

class CaliperWrapper
{
public:
  CaliperWrapper( std::string const & options );

  ~CaliperWrapper();

private:
  std::unique_ptr< cali::ConfigManager > _configManager;
};

} // namespace tcscf