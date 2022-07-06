#include "caliperInterface.hpp"

#include <caliper/cali-manager.h>

namespace tcscf
{

CaliperWrapper::CaliperWrapper( std::string const & options ):
  _configManager( std::make_unique< cali::ConfigManager >() )
{
  _configManager->add( options.c_str() );
  LVARRAY_ERROR_IF( _configManager->error(), "Caliper config error: " << _configManager->error_msg() );
  _configManager->start();
}

CaliperWrapper::~CaliperWrapper()
{
  _configManager->flush();
}

} // namespace tcscf