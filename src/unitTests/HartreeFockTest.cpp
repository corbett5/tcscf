#include "../setup.hpp"
#include "../caliperInterface.hpp"
#include  "../HartreeFock.hpp"
#include "../HydrogenLikeBasis.hpp"


#include "testingCommon.hpp"

namespace tcscf::testing
{

TEST( HartreeFock, Helium )
{
  int const Z = 2;

  std::vector< HydrogenLikeBasisFunction< double > > basisFunctions = {
    HydrogenLikeBasisFunction< double >{ Z, 1, 0, +0 },
    HydrogenLikeBasisFunction< double >{ Z, 2, 0, +0 },
    HydrogenLikeBasisFunction< double >{ Z, 2, 1, -1 },
    HydrogenLikeBasisFunction< double >{ Z, 2, 1, +0 },
    HydrogenLikeBasisFunction< double >{ Z, 2, 1, +1 },
    // HydrogenLikeBasisFunction< double >{ Z, 3, 1, -1 },
    // HydrogenLikeBasisFunction< double >{ Z, 3, 1, +0 },
    // HydrogenLikeBasisFunction< double >{ Z, 3, 1, +1 },
    // HydrogenLikeBasisFunction< double >{ Z, 3, 2, -2 },
    // HydrogenLikeBasisFunction< double >{ Z, 3, 2, -1 },
    // HydrogenLikeBasisFunction< double >{ Z, 3, 2, +0 },
    // HydrogenLikeBasisFunction< double >{ Z, 3, 2, +1 },
    // HydrogenLikeBasisFunction< double >{ Z, 3, 2, +2 },
  };

  IndexType const nBasis = basisFunctions.size();

  Array2d< std::complex< double > > const oneElectronTerms = computeCoreMatrix( Z, basisFunctions );
  Array4d< std::complex< double > > const twoElectronTerms = computeR12Matrix( basisFunctions );

  Array2d< std::complex< double > > density( nBasis, nBasis );
  Array2d< std::complex< double > > overlap;

  RCSHartreeFock< std::complex< double > > hfCalculator( 2, nBasis );
  hfCalculator.compute( true, overlap, oneElectronTerms, twoElectronTerms );
}

} // namespace tcscf::testing


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );

  tcscf::CommandLineOptions options = tcscf::parseCommandLineOptions( argc, argv );
  LVARRAY_LOG_VAR( options.caliperArgs );
  tcscf::CaliperWrapper caliperWrapper( options.caliperArgs );

  int const result = RUN_ALL_TESTS();
  tcscf::printHighWaterMarks();

  return result;
}
