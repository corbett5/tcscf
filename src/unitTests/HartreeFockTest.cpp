#include "../setup.hpp"
#include "../caliperInterface.hpp"
#include  "../HartreeFock.hpp"
#include "../HydrogenLikeBasis.hpp"
#include "../OchiBasis.hpp"


#include "testingCommon.hpp"

namespace tcscf::testing
{


TEST( AtomicHartreeFock, Helium )
{
  int const Z = 2;
  int const nMax = 2;

  std::vector< HydrogenLikeBasisFunction< double > > basisFunctions;
  std::vector< AtomicParams > params;

  for( int n = 1; n <= nMax; ++n )
  {
    for( int l = 0; l < n; ++l )
    {
      for( int m = -l; m <= l; ++m )
      {
        basisFunctions.emplace_back( Z, n, l, m );
        params.emplace_back( AtomicParams{ n, l, m } );
      }
    }
  }

  IndexType const nBasis = basisFunctions.size();

  Array2d< double > const coreMatrix( nBasis, nBasis );
  fillCoreMatrix( Z, basisFunctions, coreMatrix );

  Array4d< std::complex< double > > const twoElectronTerms( nBasis, nBasis, nBasis, nBasis );
  fillAtomicR12Array( basisFunctions, twoElectronTerms );

  AtomicRCSHartreeFock< double > hfCalculator( 2, params );
  hfCalculator.compute( coreMatrix, twoElectronTerms );
}


TEST( AtomicHartreeFock, Helium_Ochi )
{
  int const Z = 2;
  int const nMax = 2;
  int const lMax = 1;

  double alpha = 2;

  for( int iter = 0; iter < 4; ++iter )
  {
    LVARRAY_LOG_VAR( alpha );
    
    std::vector< OchiBasisFunction< double > > basisFunctions;
    std::vector< AtomicParams > params;

    for( int n = 0; n <= nMax; ++n )
    {
      for( int l = 0; l <= lMax; ++l )
      {
        for( int m = -l; m <= l; ++m )
        {
          basisFunctions.emplace_back( alpha, n, l, m );
          params.emplace_back( AtomicParams{ n, l, m } );
        }
      }
    }

    IndexType const nBasis = basisFunctions.size();

    Array2d< double > const coreMatrix( nBasis, nBasis );
    fillCoreMatrix( Z, basisFunctions, coreMatrix );
    Array4d< std::complex< double > > const twoElectronTerms( nBasis, nBasis, nBasis, nBasis );
    fillAtomicR12Array( basisFunctions, twoElectronTerms );

    AtomicRCSHartreeFock< double > hfCalculator( 2, params );
    hfCalculator.compute( coreMatrix, twoElectronTerms );

    alpha = std::sqrt( -2 * hfCalculator.eigenvalues[ 0 ] );
  }

  LVARRAY_LOG_VAR( alpha );
}


// TEST( HartreeFock, Helium )
// {
//   int const Z = 2;

//   std::vector< HydrogenLikeBasisFunction< double > > basisFunctions = {
//     HydrogenLikeBasisFunction< double >{ Z, 1, 0, +0 },
//     HydrogenLikeBasisFunction< double >{ Z, 2, 0, +0 },
//     // HydrogenLikeBasisFunction< double >{ Z, 2, 1, -1 },
//     HydrogenLikeBasisFunction< double >{ Z, 2, 1, +0 },
//     // HydrogenLikeBasisFunction< double >{ Z, 2, 1, +1 },
//     // HydrogenLikeBasisFunction< double >{ Z, 3, 1, -1 },
//     // HydrogenLikeBasisFunction< double >{ Z, 3, 1, +0 },
//     // HydrogenLikeBasisFunction< double >{ Z, 3, 1, +1 },
//     // HydrogenLikeBasisFunction< double >{ Z, 3, 2, -2 },
//     // HydrogenLikeBasisFunction< double >{ Z, 3, 2, -1 },
//     // HydrogenLikeBasisFunction< double >{ Z, 3, 2, +0 },
//     // HydrogenLikeBasisFunction< double >{ Z, 3, 2, +1 },
//     // HydrogenLikeBasisFunction< double >{ Z, 3, 2, +2 },
//     // HydrogenLikeBasisFunction< double >{ Z, 4, 0, +0 },
//     // HydrogenLikeBasisFunction< double >{ Z, 5, 0, +0 },
//     // HydrogenLikeBasisFunction< double >{ Z, 6, 0, +0 },
//     // HydrogenLikeBasisFunction< double >{ Z, 7, 0, +0 },
//     // HydrogenLikeBasisFunction< double >{ Z, 8, 0, +0 },
//     // HydrogenLikeBasisFunction< double >{ Z, 9, 0, +0 },
//     // HydrogenLikeBasisFunction< double >{ Z, 10, 0, +0 },

//   };

//   int const basisSize = basisFunctions.size();

//   Array2d< std::complex< double > > const oneElectronTerms = computeCoreMatrix( Z, basisFunctions );
//   Array4d< std::complex< double > > const twoElectronTerms = computeR12Matrix( basisFunctions );

//   Array2d< std::complex< double > > density( basisSize, basisSize );
//   Array2d< std::complex< double > > overlap;

//   RCSHartreeFock< std::complex< double > > hfCalculator( 2, basisSize );
//   hfCalculator.compute( true, overlap, oneElectronTerms, twoElectronTerms );
// }

} // namespace tcscf::testing


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );

  tcscf::CommandLineOptions options = tcscf::parseCommandLineOptions( argc, argv );
  tcscf::CaliperWrapper caliperWrapper( options.caliperArgs );

  int const result = RUN_ALL_TESTS();
  tcscf::printHighWaterMarks();

  return result;
}
