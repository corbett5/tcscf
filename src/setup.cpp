#include "setup.hpp"

// TPL includes
#include "Macros.hpp"

#include <optionparser.h>
#include <umpire/ResourceManager.hpp>
#include <chai/ArrayManager.hpp>

namespace tcscf
{

/**
 * @class Arg a class inheriting from option::Arg that can parse a command line argument.
 */
struct Arg : public option::Arg
{
  /**
   * @brief Parse an unknown option. Unknown options aren't supported so this throws an error.
   * @param option the option to parse.
   * @return option::ARG_ILLEGAL.
   */
  static option::ArgStatus unknown( option::Option const & option, bool )
  {
    LVARRAY_LOG( "Unknown option: " << option.name );
    return option::ARG_ILLEGAL;
  }

  /**
   * @brief Parse a non-empty string option.
   * @param option the option to parse.
   * @return option::ARK_OK if the parse was successful, option::ARG_ILLEGAL otherwise.
   */
  static option::ArgStatus nonEmpty( const option::Option & option, bool )
  {
    if((option.arg != nullptr) && (option.arg[0] != 0))
    {
      return option::ARG_OK;
    }

    LVARRAY_LOG( "Error: " << option.name << " requires a non-empty argument!" );
    return option::ARG_ILLEGAL;
  }

  /**
   * @brief Parse an integer string option.
   * @param option the option to parse.
   * @return option::ARK_OK if the parse was successful, option::ARG_ILLEGAL otherwise.
   */
  static option::ArgStatus integer( const option::Option & option, bool )
  {
    char * endptr = nullptr;
    if((option.arg != nullptr) && strtol( option.arg, &endptr, 10 )) {}
    if((endptr != option.arg) && (*endptr == 0))
    {
      return option::ARG_OK;
    }

    LVARRAY_LOG( "Error: " << option.name << " requires a long-int argument!" );
    return option::ARG_ILLEGAL;
  }

  /**
   * @brief Parse a floating point string option.
   * @param option the option to parse.
   * @return option::ARK_OK if the parse was successful, option::ARG_ILLEGAL otherwise.
   */
  static option::ArgStatus floatingPoint( const option::Option & option, bool )
  {
    char * endptr = nullptr;
    if((option.arg != nullptr) && strtod( option.arg, &endptr )) {}
    if((endptr != option.arg) && (*endptr == 0))
    {
      return option::ARG_OK;
    }

    LVARRAY_LOG( "Error: " << option.name << " requires a floating point argument!" );
    return option::ARG_ILLEGAL;
  }
};

CommandLineOptions parseCommandLineOptions( int argc, char ** argv )
{
  CommandLineOptions commandLineOptions;

  // Set the options structs and parse
  enum optionIndex
  {
    UNKNOWN,
    HELP,
    NMAX,
    LMAX,
    ALPHA,
    R1,
    R2,
    TIMERS,
    SUPPRESS_MOVE_LOGGING
  };

  const option::Descriptor usage[] =
  {
    { UNKNOWN, 0, "", "", Arg::unknown, "USAGE: [options]\n\nOptions:" },
    { HELP, 0, "?", "help", Arg::None, "\t-?, --help" },
    { NMAX, 0, "n", "nMax", Arg::integer, "\t-n, --nMax, \t Maximum principle quantum number." },
    { LMAX, 0, "l", "lMax", Arg::integer, "\t-l, --lMax, \t Maximum angular quantum number." },
    { ALPHA, 0, "a", "alpha", Arg::floatingPoint, "\t-a, --alpha, \t Initial orbital exponent." },
    { R1, 0, "", "r1", Arg::integer, "\t--r1, \t Grid size used for r1 integration." },
    { R2, 0, "", "r2", Arg::integer, "\t--r2, \t Grid size used for r2 integration." },
    { TIMERS, 0, "c", "caliper", Arg::nonEmpty, "\t-c, --caliper, \t String specifying the type of timer output." },
    { SUPPRESS_MOVE_LOGGING, 0, "", "suppress-move-logging", Arg::None, "\t--suppress-move-logging \t Suppress logging of host-device data migration" },
    { 0, 0, nullptr, nullptr, nullptr, nullptr }
  };

  argc -= ( argc > 0 );
  argv += ( argc > 0 );
  option::Stats stats( usage, argc, argv );
  option::Option options[ 100 ];//stats.options_max];
  option::Option buffer[ 100 ];//stats.buffer_max];
  option::Parser parse( usage, argc, argv, options, buffer );

  // Handle special cases
  if( parse.error() || options[HELP] )
  {
    int columns = getenv( "COLUMNS" ) ? atoi( getenv( "COLUMNS" )) : 120;
    option::printUsage( fwrite, stdout, usage, columns );

    if( options[HELP] )
    {
      exit( 0 );
    }

    LVARRAY_LOG( "Bad command line arguments." );
    exit( 1 );
  }

  // Iterate over the remaining inputs
  for( int ii=0; ii<parse.optionsCount(); ++ii )
  {
    option::Option & opt = buffer[ii];
    switch( opt.index() )
    {
      case UNKNOWN:
      {
        break;
      }
      case HELP:
      {
        break;
      }
      case NMAX:
      {
        commandLineOptions.nMax = std::stol( opt.arg );
        LVARRAY_ERROR_IF_LT( commandLineOptions.nMax, 0 );
        break;
      }
      case LMAX:
      {
        commandLineOptions.lMax = std::stol( opt.arg );
        LVARRAY_ERROR_IF_LT( commandLineOptions.lMax, 0 );
        break;
      }
      case ALPHA:
      {
        commandLineOptions.initialAlpha = std::stod( opt.arg );
        LVARRAY_ERROR_IF_LE( commandLineOptions.initialAlpha, 0 );
        break;
      }
      case R1:
      {
        commandLineOptions.r1GridSize = std::stol( opt.arg );
        LVARRAY_ERROR_IF_LE( commandLineOptions.r1GridSize, 0 );
        break;
      }
      case R2:
      {
        commandLineOptions.r2GridSize = std::stol( opt.arg );
        LVARRAY_ERROR_IF_LE( commandLineOptions.r2GridSize, 0 );
        break;
      }
      case TIMERS:
      {
        commandLineOptions.caliperArgs = opt.arg;
        break;
      }
      case SUPPRESS_MOVE_LOGGING:
      {
        commandLineOptions.suppressMoveLogging = true;
        chai::ArrayManager::getInstance()->disableCallbacks();
        break;
      }
    }
  }

  return commandLineOptions;
}

void printHighWaterMarks()
{
  umpire::ResourceManager & rm = umpire::ResourceManager::getInstance();

  // Loop over the allocators.
  for( std::string const & allocatorName : rm.getAllocatorNames() )
  {
    // Skip umpire internal allocators.
    if( allocatorName.rfind( "__umpire_internal", 0 ) == 0 )
      continue;

    // Get the total number of bytes allocated with this allocator across ranks.
    // This is a little redundant since
    std::size_t const mark = rm.getAllocator( allocatorName ).getHighWatermark();
    LVARRAY_LOG( "Umpire " << std::setw( 15 ) << allocatorName << " high water mark: " <<
                 std::setw( 9 ) << LvArray::system::calculateSize( mark ) );
  }
}

} // namespace tcscf