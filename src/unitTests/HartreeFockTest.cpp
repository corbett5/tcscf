#include "../setup.hpp"
#include "../caliperInterface.hpp"
#include  "../HartreeFock.hpp"
#include "../HydrogenLikeBasis.hpp"
#include "../OchiBasis.hpp"
#include "../jastrowFunctions.hpp"

#include "testingCommon.hpp"

namespace tcscf::testing
{

CommandLineOptions clo;

int createBasisFunctions(
  int const nMax,
  int const lMax,
  double const alpha,
  std::vector< OchiBasisFunction< double > > & basisFunctions )
{
  basisFunctions.clear();

  for( int n = 0; n <= nMax; ++n )
  {
    for( int l = 0; l <= lMax; ++l )
    {
      for( int m = -l; m <= l; ++m )
      {
        basisFunctions.emplace_back( alpha, n, l, m );
      }
    }
  }

  return basisFunctions.size();
}

template< typename HF_CALCULATOR >
void ochiHF(
  int const nMax,
  int const lMax,
  double const initialAlpha,
  int const r1GridSize,
  int const r2GridSize )
{
  constexpr double HF_LIMIT = -2.861679995612;

  int const Z = 2;
  int const nSpinUp = 1;
  int const nSpinDown = 1;

  double alpha = initialAlpha;

  std::vector< OchiBasisFunction< double > > basisFunctions;
  int const nBasis = createBasisFunctions( nMax, lMax, alpha, basisFunctions );

  LVARRAY_LOG( "nMax = " << nMax << ", lMax = " << lMax << ", nBasis = " << nBasis <<
               ", r1 grid size = " << r1GridSize << ", r2 grid size = " << r2GridSize << ", alpha = " << alpha );

  Array2d< double > const coreGrid = integration::createGrid(
    integration::ChebyshevGauss< double >( 1000 ),
    integration::changeOfVariables::TreutlerAhlrichsM4< double >( 1, 0.9 ) );

  HF_CALCULATOR hfCalculator( nSpinUp, nSpinDown, basisFunctions.size() );

  Array1d< double > energies;

  int const nIter = 30;
  for( int iter = 0; iter < nIter; ++iter )
  {
    Array2d< double > const coreMatrix( nBasis, nBasis );
    fillCoreMatrix( coreGrid, Z, basisFunctions, coreMatrix );
    
    Array4d< std::complex< double > > coulombIntegrals = integration::integrateAllR1R12< double >( r1GridSize, r2GridSize, basisFunctions,
      [] ( double const r1, Cartesian< double > const & r1C, double const r12, Cartesian< double > const & r12C, double const r2 )
      {
        LVARRAY_UNUSED_VARIABLE( r1 );
        LVARRAY_UNUSED_VARIABLE( r1C );
        LVARRAY_UNUSED_VARIABLE( r12C );
        LVARRAY_UNUSED_VARIABLE( r2 );
        return 1 / r12;
      }
    );

    // Array4d< std::complex< double > > coulombIntegrals( nBasis, nBasis, nBasis, nBasis );
    // fillAtomicR12Array( basisFunctions, coulombIntegrals );

    std::cout << std::setprecision( 10 );

    double energy = 0;
    if constexpr ( std::is_same_v< HF_CALCULATOR, TCHartreeFock< std::complex< double > > > )
    {
      double const a = 1.5;
      double const a12 = a;
      Array2d< int > S( 1, 3 );
      S( 0, 0 ) = 1;

      Array2d< double > c( 1, 2 );
      c( 0, false ) = a12 / 2;
      c( 0, true ) = a12 / 4;

      jastrowFunctions::Ochi< double > const u { a, a12, c, S };

      Array4d< std::complex< double > > laplacianOpposite = integration::integrateAllR1R12< double >( r1GridSize, r2GridSize, basisFunctions,
        [&u] ( double const r1, Cartesian< double > const & r1C, double const r12, Cartesian< double > const & r12C, double const r2 )
        {
          Cartesian< double > const r2C = r1C + r12C;
          Cartesian< double > r21C { -r12C.x(), -r12C.y(), -r12C.z() };

          Cartesian< double > const grad1 = u.gradient( r1, r1C, r12, r12C, r2, false );
          Cartesian< double > const grad2 = u.gradient( r2, r2C, r12, r21C, r1, false );

          return u.laplacian( r1, r1C, r12, r12C, r2, false ) + u.laplacian( r2, r2C, r12, r21C, r1, false ) - dot( grad1, grad1 ) - dot( grad2, grad2 );
        }
      );

      Array4d< std::complex< double > > gradOpposite = integration::integrateAllR1R12< double >( r1GridSize, r2GridSize, basisFunctions,
        [&u] ( double const r1, Cartesian< double > const & r1C, double const r12, Cartesian< double > const & r12C, double const r2 )
        {
          Cartesian< double > const r2C = r1C + r12C;
          Cartesian< double > r21C { -r12C.x(), -r12C.y(), -r12C.z() };

          Cartesian< double > const grad1 = u.gradient( r1, r1C, r12, r12C, r2, false );
          Cartesian< double > const grad2 = u.gradient( r2, r2C, r12, r21C, r1, false );

          return dot( grad1, grad1 ) + dot( grad2, grad2 );
        }
      );

      Array4d< std::complex< double > > const twoElectronOppositeSpin( nBasis, nBasis, nBasis, nBasis );
      Array4d< std::complex< double > > const twoElectronSameSpin( nBasis, nBasis, nBasis, nBasis );
      
      for( int b1 = 0; b1 < nBasis; ++b1 )
      {
        for( int b2 = 0; b2 < nBasis; ++b2 )
        {
          for( int b3 = 0; b3 < nBasis; ++b3 )
          {
            for( int b4 = 0; b4 < nBasis; ++b4 )
            {
              twoElectronOppositeSpin( b1, b2, b3, b4 ) = 2 * coulombIntegrals( b1, b2, b3, b4 ) + laplacianOpposite( b1, b2, b3, b4 ) - gradOpposite( b1, b2, b3, b4 );
              twoElectronSameSpin( b1, b2, b3, b4 ) = 2 * coulombIntegrals( b1, b2, b3, b4 ) + (laplacianOpposite( b1, b2, b3, b4 ) / 2 - gradOpposite( b1, b2, b3, b4 ) / 4);
            }
          }
        }
      }

      energy = hfCalculator.compute( true, {}, coreMatrix, twoElectronSameSpin, twoElectronOppositeSpin );
    }
    else
    {
      energy = hfCalculator.compute( true, {}, coreMatrix, coulombIntegrals );
    }

    // TODO: zero out the two electron electron terms that don't share the same angular coords.

    if( nIter - iter <= 10 )
    {
      energies.emplace_back( energy );
    }

    alpha = std::sqrt( -2 * hfCalculator.highestOccupiedOrbitalEnergy() );

    printf( "\r    iteration = %4d, energy = %10.6F, error = %e, alpha = %10.6F", iter, energy, std::abs( HF_LIMIT - energy ), alpha );
    fflush( stdout );

    createBasisFunctions( nMax, lMax, alpha, basisFunctions );
  }

  std::cout << std::endl;

  auto [mean, standardDev] = meanAndStd( energies.toViewConst() );
  LVARRAY_LOG( "energy = " << mean << " +/- " << standardDev <<
               " Ht, error = " << std::abs( HF_LIMIT - mean ) << " Ht, alpha = " << alpha );
}


TEST( HartreeFock, RestrictedClosedShell )
{
  ochiHF< RCSHartreeFock< std::complex< double > > >( clo.nMax, clo.lMax, clo.initialAlpha, clo.r1GridSize, clo.r2GridSize );
}

TEST( HartreeFock, UnrestrictedOpenShell )
{
  ochiHF< UOSHartreeFock< std::complex< double > > >( clo.nMax, clo.lMax, clo.initialAlpha, clo.r1GridSize, clo.r2GridSize );
}

TEST( HartreeFock, Transcorrelated )
{
  ochiHF< TCHartreeFock< std::complex< double > > >( clo.nMax, clo.lMax, clo.initialAlpha, clo.r1GridSize, clo.r2GridSize );
}

} // namespace tcscf::testing


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );

  tcscf::testing::clo = tcscf::parseCommandLineOptions( argc, argv );
  tcscf::CaliperWrapper caliperWrapper( tcscf::testing::clo.caliperArgs );

  int const result = RUN_ALL_TESTS();
  tcscf::printHighWaterMarks();

  return result;
}

// clear; ninja HartreeFockTest && ./tests/HartreeFockTest -n2 -l0 -a 1.31 --r1 1000 --r2 1000