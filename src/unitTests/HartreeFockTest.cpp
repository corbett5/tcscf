#include "../setup.hpp"
#include "../caliperInterface.hpp"
#include  "../HartreeFock.hpp"
#include "../HydrogenLikeBasis.hpp"
#include "../OchiBasis.hpp"
#include "../jastrowFunctions.hpp"

#include "testingCommon.hpp"

namespace tcscf::testing
{

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
  int const nMax = 2;
  int const lMax = 0;

  double alpha = 1.35468;

  int const r1GridSize = 3000;
  int const r2GridSize = 3000;

  std::vector< OchiBasisFunction< double > > basisFunctions;
  std::vector< AtomicParams > params;
  int const nBasis = createBasisFunctions( nMax, lMax, alpha, basisFunctions, params );

  LVARRAY_LOG( "nMax = " << nMax << ", lMax = " << lMax << ", nBasis = " << nBasis << ", r1 grid size = " << r1GridSize << ", r2 grid size = " << r2GridSize );

  Array2d< double > const coreGrid = integration::createGrid(
    integration::ChebyshevGauss< double >( 1000 ),
    integration::changeOfVariables::TreutlerAhlrichsM4< double >( 1, 0.9 ) );

  AtomicRCSHartreeFock< double > hfCalculator( 2, params );
  Array1d< double > energies;

  int const nIter = 1;
  for( int iter = 0; iter < nIter; ++iter )
  {
    Array2d< double > const coreMatrix( nBasis, nBasis );
    fillCoreMatrix( coreGrid, Z, basisFunctions, coreMatrix );
    
    Array4d< std::complex< double > > twoElectronTerms = integration::integrateAllR1R12< double >( r1GridSize, r2GridSize, basisFunctions,
      [] ( double const r1, Cartesian< double > const & r1C, double const r12, Cartesian< double > const & r12C, double const r2 )
      {
        LVARRAY_UNUSED_VARIABLE( r1 );
        LVARRAY_UNUSED_VARIABLE( r1C );
        LVARRAY_UNUSED_VARIABLE( r12C );
        LVARRAY_UNUSED_VARIABLE( r2 );
        return 1 / r12;
      }
    );

    {
      double const a = 1.5;
      Array2d< int > S( 1, 3 );
      S( 0, 0 ) = 1;

      Array2d< double > c( 1, 2 );
      c( 0, false ) = a / 2;
      c( 0, true ) = a / 4;

      jastrowFunctions::Ochi< double > const u { a, a, c, S };
      Array4d< std::complex< double > > fooBar = integration::integrateAllR1R12< double >( r1GridSize, r2GridSize, basisFunctions,
        [&u] ( double const r1, Cartesian< double > const & r1C, double const r12, Cartesian< double > const & r12C, double const r2 )
        {
          Cartesian< double > const r2C = r1C + r12C;
          Cartesian< double > r21C {-r12C.x(), -r12C.y(), -r12C.z() };

          Cartesian< double > const grad1 = u.gradient( r1, r1C, r12, r12C, r2, false );
          Cartesian< double > const grad2 = u.gradient( r2, r2C, r12, r21C, r1, false );

          return u.laplacian( r1, r1C, r12, r12C, r2, false ) + u.laplacian( r2, r2C, r12, r21C, r1, false ) - dot( grad1, grad1 ) - dot( grad2, grad2 );
        }
      );

      LVARRAY_LOG_VAR( fooBar[ 0 ][ 0 ][ 0 ] );
    }

    double energy = hfCalculator.compute( coreMatrix, twoElectronTerms, 100 ).real();
    LVARRAY_LOG_VAR( energy );

    if( nIter - iter <= 10 )
    {
      energies.emplace_back( energy );
    }

    alpha = std::sqrt( -2 * hfCalculator.eigenvalues[ 0 ] );

    createBasisFunctions( nMax, lMax, alpha, basisFunctions, params );
  }

  auto [mean, std] = meanAndStd( energies.toViewConst() );
  LVARRAY_LOG( "energy = " << mean << " +/- " << std << ", alpha = " << alpha );
}

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
