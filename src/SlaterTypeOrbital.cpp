#include "SlaterTypeOrbital.hpp"

#include <unordered_set>
#include <filesystem>
#include <fstream>

namespace tcscf
{

/**
 */
int spdfToL( char const type )
{
  if( type == 'S' ) return 0;
  if( type == 'P' ) return 1;
  if( type == 'D' ) return 2;
  if( type == 'F' ) return 3;

  LVARRAY_ERROR( "Unrecognized type: " << type );
  return -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
std::vector< SlaterTypeOrbital< double > > loadSTO(
  std::string const & directory,
  std::string const & basisSet,
  std::string const & element,
  int const maxL )
{
  std::unordered_set< std::string > const validBasisSets = { "TZ3P", "QZ6P", "ATZ3P", "AQZ6P" };

  LVARRAY_ERROR_IF( !validBasisSets.count( basisSet ), "'" << basisSet << "' is not a valid basis set." );

  std::filesystem::path const directoryPath = directory;
  LVARRAY_ERROR_IF( !std::filesystem::directory_entry( directoryPath / basisSet ).is_directory(),
    "'" << basisSet << "' not found in '" << directory << "'."  );

  std::filesystem::directory_entry file( directoryPath / basisSet / element );
  // LVARRAY_ERROR_IF( file.exists(), file.path() );

  std::ifstream contents( file.path() );

  std::vector< SlaterTypeOrbital< double > > basisFunctions;
  for( std::string line; std::getline( contents, line ); )
  {
    if( line.rfind( "BASIS", 0 ) == 0 )
    {
      while( std::getline( contents, line ), line.rfind( "END", 0 ) != 0 )
      {
        int n;
        char type;
        double alpha;
        std::istringstream lineStream( line );
        if( lineStream >> n >> type >> alpha )
        {
          int const l = spdfToL( type );
          if( l > maxL ) continue;

          for( int m = -l; m <= l; ++m )
          {
            basisFunctions.emplace_back( alpha, n, l, m );
          }
        }
        else if ( !line.empty() )
        {
          LVARRAY_ERROR( "Invalid line in file: " << line );
        }
      }

      break;
    }
  }

  return basisFunctions;
}


} // namespace tcscf