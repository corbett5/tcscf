#include "../setup.hpp"
#include "../caliperInterface.hpp"
#include "../HartreeFock.hpp"
#include "../HydrogenLikeBasis.hpp"
#include "../OchiBasis.hpp"
#include "../jastrowFunctions.hpp"
#include "../integration/integrateAll.hpp"
#include "../integration/ChebyshevGauss.hpp"
#include "../integration/changeOfVariables.hpp"

#include "testingCommon.hpp"

namespace tcscf::testing
{

CommandLineOptions clo;

/**
 * 
 */
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

/**
 * 
 */
Array4d< std::complex< double > > computeLOppositeSpin(
  int const r1GridSize,
  int const r2GridSize,
  std::vector< OchiBasisFunction< double > > const & basisFunctions,
  jastrowFunctions::Ochi< double > const & u )
{
  return integration::integrateAll< double >( r1GridSize, r2GridSize, basisFunctions,
    [&u] ( double const r1, Cartesian< double > const & r1C, double const r12, Cartesian< double > const & r12C, double const r2 )
    {
      return u.laplacian( r1, r1C, r12, r12C, r2, false );
    }
  );
}

/**
 * 
 */
Array4d< std::complex< double > > computeGOppositeSpin(
  int const r1GridSize,
  int const r2GridSize,
  std::vector< OchiBasisFunction< double > > const & basisFunctions,
  jastrowFunctions::Ochi< double > const & u )
{
  return integration::integrateAll< double >( r1GridSize, r2GridSize, basisFunctions,
    [&u] ( double const r1, Cartesian< double > const & r1C, double const r12, Cartesian< double > const & r12C, double const r2 )
    {
      Cartesian< double > const grad1 = u.gradient( r1, r1C, r12, r12C, r2, false );
      return dot( grad1, grad1 );
    }
  );
}

/**
 * 
 */
void computeH2Prime(
  ArrayView4d< std::complex< double > > const & h2PrimeOppo,
  ArrayView4d< std::complex< double > > const & h2PrimeSame,
  ArrayView4d< std::complex< double > const > const & R,
  ArrayView4d< std::complex< double > const > const & LOppo,
  ArrayView4d< std::complex< double > const > const & GOppo )
{
  int const nBasis = h2PrimeOppo.size( 0 );
  for( int dim = 0; dim < 4; ++dim )
  {
    LVARRAY_ERROR_IF_NE( h2PrimeOppo.size( dim ), nBasis );
    LVARRAY_ERROR_IF_NE( h2PrimeSame.size( dim ), nBasis );
    LVARRAY_ERROR_IF_NE( R.size( dim ), nBasis );
    LVARRAY_ERROR_IF_NE( LOppo.size( dim ), nBasis );
    LVARRAY_ERROR_IF_NE( GOppo.size( dim ), nBasis );
  }

  for( int j = 0; j < nBasis; ++j )
  {
    for( int ell = 0; ell < nBasis; ++ell )
    {
      for( int i = 0; i < nBasis; ++i )
      {
        for( int m = 0; m < nBasis; ++m )
        {
          std::complex< double > const h2_oppo_jl_im = R( j, ell, i, m ) + LOppo( j, ell, i, m ) - GOppo( j, ell, i, m );
          std::complex< double > const h2_oppo_lj_mi = R( ell, j, m, i ) + LOppo( ell, j, m, i ) - GOppo( ell, j, m, i );
          h2PrimeOppo( j, ell, i, m ) = h2_oppo_jl_im + h2_oppo_lj_mi;

          std::complex< double > const h2_same_jl_im = R( j, ell, i, m ) + LOppo( j, ell, i, m ) / 2 - GOppo( j, ell, i, m ) / 4;
          std::complex< double > const h2_same_lj_mi = R( ell, j, m, i ) + LOppo( ell, j, m, i ) / 2 - GOppo( ell, j, m, i ) / 4;
          h2PrimeSame( j, ell, i, m ) = h2_same_jl_im + h2_same_lj_mi;
        }
      }
    }
  }
}

/**
 * 
 */
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
    
    Array4d< std::complex< double > > R = integration::integrateAll< double >( r1GridSize, r2GridSize, basisFunctions,
      [] ( double const r1, Cartesian< double > const & r1C, double const r12, Cartesian< double > const & r12C, double const r2 )
      {
        LVARRAY_UNUSED_VARIABLE( r1 );
        LVARRAY_UNUSED_VARIABLE( r1C );
        LVARRAY_UNUSED_VARIABLE( r12C );
        LVARRAY_UNUSED_VARIABLE( r2 );
        return 1 / r12;
      }
    );

    // Array4d< std::complex< double > > R( nBasis, nBasis, nBasis, nBasis );
    // fillAtomicR12Array( basisFunctions, R );

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

      Array4d< std::complex< double > > const LOppo = computeLOppositeSpin( r1GridSize, r2GridSize, basisFunctions, u );
      Array4d< std::complex< double > > const GOppo = computeGOppositeSpin( r1GridSize, r2GridSize, basisFunctions, u );

      Array4d< std::complex< double > > const h2PrimeOppo( nBasis, nBasis, nBasis, nBasis );
      Array4d< std::complex< double > > const h2PrimeSame( nBasis, nBasis, nBasis, nBasis );

      computeH2Prime( h2PrimeOppo, h2PrimeSame, R, LOppo, GOppo );

      energy = hfCalculator.compute( true, {}, coreMatrix, h2PrimeSame, h2PrimeOppo );
    }
    else
    {
      energy = hfCalculator.compute( true, {}, coreMatrix, R );
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