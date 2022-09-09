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
  fillCoreMatrix( ArrayView2d< double > {}, Z, basisFunctions, coreMatrix );

  Array4d< std::complex< double > > const twoElectronTerms( nBasis, nBasis, nBasis, nBasis );
  fillAtomicR12Array( basisFunctions, twoElectronTerms );

  AtomicRCSHartreeFock< double > hfCalculator( 2, params );
  hfCalculator.compute( coreMatrix, twoElectronTerms, 100 );
}


int createBasisFunctions(
  int const nMax,
  int const lMax,
  double const alpha,
  std::vector< OchiBasisFunction< double > > & basisFunctions,
  std::vector< AtomicParams > & params )
{
  basisFunctions.clear();
  params.clear();

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

  return basisFunctions.size();
}


TEST( AtomicHartreeFock, Helium_Ochi )
{
  int const Z = 2;
  int const nMax = 4;
  int const lMax = 0;

  double alpha = 1.34878;

  int const r1r12Radial = 50;
  int const r1r12AngularOrder = 59;

  std::string const integrationScheme = "Multi integral quadrature";

  std::vector< OchiBasisFunction< double > > basisFunctions;
  std::vector< AtomicParams > params;
  int const nBasis = createBasisFunctions( nMax, lMax, alpha, basisFunctions, params );

  LVARRAY_LOG( integrationScheme << ", nMax = " << nMax << ", lMax = " << lMax << ", nBasis = " << nBasis << ", radial ngrid = " << r1r12Radial << ", angular order = " << r1r12AngularOrder );

  Array2d< double > const r1Grid = integration::createGrid(
    integration::ChebyshevGauss< double >( 1000 ),
    integration::changeOfVariables::TreutlerAhlrichs< double >( 0.9 ) );

  integration::TreutlerAhlrichsLebedev< double > r1r12Grid( 0.9, r1r12Radial, r1r12AngularOrder );

  AtomicRCSHartreeFock< double > hfCalculator( 2, params );
  for( int iter = 0; iter < 5; ++iter )
  {
    LVARRAY_LOG_VAR( alpha );

    Array2d< double > const coreMatrix( nBasis, nBasis );
    fillCoreMatrix( r1Grid, Z, basisFunctions, coreMatrix );
    
    Array4d< std::complex< double > > twoElectronTerms;

    // Compute individual integrals
    if( integrationScheme == "quadrature" )
    {
      twoElectronTerms.resize( nBasis, nBasis, nBasis, nBasis );
      fillAtomicR12Array( r1r12Grid, basisFunctions, twoElectronTerms );
    }

    // Multi integral optimization
    if( integrationScheme == "Multi integral quadrature" )
    {
        twoElectronTerms = integrateAllR1R12( r1r12Grid, basisFunctions,
        [] ( double const LVARRAY_UNUSED_ARG( r1 ), double const LVARRAY_UNUSED_ARG( r2 ), double const r12 )
        {
          return 1 / r12;
        }
      );
    }

    // QMC
    if( integrationScheme == "QMC" )
    {
      twoElectronTerms.resize( nBasis, nBasis, nBasis, nBasis );
      fillAtomicR12Array( basisFunctions, twoElectronTerms );
    }

    hfCalculator.compute( coreMatrix, twoElectronTerms, 100 );

    // std::complex< double > energy = hfCalculator.iteration( coreMatrix, twoElectronTerms );
    // LVARRAY_LOG_VAR( energy );

    alpha = std::sqrt( -2 * hfCalculator.eigenvalues[ 0 ] );

    createBasisFunctions( nMax, lMax, alpha, basisFunctions, params );
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
