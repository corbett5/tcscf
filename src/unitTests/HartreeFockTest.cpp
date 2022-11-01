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




template< typename POLICY_TYPE >
class DiffChecker
{
  using PAIR = RAJA::tuple< double, double >;

public:

  DiffChecker( double const rtol, double const atol ):
    _rtol( rtol ),
    _atol( atol )
  {}

  LVARRAY_HOST_DEVICE bool witness( double const baseV, double const newV ) const
  {
    double const diff = LvArray::math::abs( baseV - newV );
    double const rdiff = diff <= 0 ? 0 : diff / LvArray::math::abs( baseV );

    bool compareEqual = true;
    if( rdiff > _rtol && diff > _atol )
    {
      _numViolators += 1;

      if( rdiff > _rtol )
      {
        _maxRDiff.maxloc( rdiff, PAIR( baseV, newV ) );
      }
      if( diff > _atol )
      {
        _maxDiff.maxloc( diff, PAIR( baseV, newV ) );
      }

      compareEqual = false;
    }

    _squaredError += diff * diff;
    _baseNormSquared += baseV * baseV;

    _numWitnessed += 1;

    return compareEqual;
  }

  IndexType numViolators()
  { return _numViolators.get(); }

  double getMaximumRelativeViolation()
  {
    return _maxRDiff.get();
  }

  double getMaximumRelativeViolationAbsoluteDiff()
  {
    return std::abs( RAJA::get< 0 >( _maxRDiff.getLoc() ) - RAJA::get< 1 >( _maxRDiff.getLoc() ) );
  }

  std::string report()
  {
    std::ostringstream oss;

    double const rse = _squaredError.get() / _baseNormSquared.get();
    double const maxDiff = _maxDiff.get();
    double const maxDiff0 = RAJA::get< 0 >( _maxDiff.getLoc() );
    double const maxDiff1 = RAJA::get< 1 >( _maxDiff.getLoc() );
    double const maxRDiff = _maxRDiff.get();
    double const maxRDiff0 = RAJA::get< 0 >( _maxRDiff.getLoc() );
    double const maxRDiff1 = RAJA::get< 1 >( _maxRDiff.getLoc() );

    oss << "With a relative tolerance of " << _rtol << " and an absolute tolerance of " << _atol << " " << _numViolators.get()
        << " / " << _numWitnessed.get() << " failed the check.\n"
        << "    Relative squared error of " << rse << "\n"
        << "    max violating error of " << maxDiff << " with values of " << maxDiff0 << " and " << maxDiff1 << "\n"
        << "    max violating relative error of " << maxRDiff << " with values of " << maxRDiff0 << " and " << maxRDiff1 << std::endl;

    return oss.str();
  }

private:
  double const _rtol;
  double const _atol;
  RAJA::ReduceSum< Reduce< POLICY_TYPE >, IndexType > _numWitnessed{ 0 };
  RAJA::ReduceSum< Reduce< POLICY_TYPE >, IndexType > _numViolators{ 0 };

  RAJA::ReduceMaxLoc< Reduce< POLICY_TYPE >, double, PAIR > _maxDiff{ 0, PAIR( 0, 0 ) };
  RAJA::ReduceMaxLoc< Reduce< POLICY_TYPE >, double, PAIR > _maxRDiff{ 0, PAIR( 0, 0 ) };
  RAJA::ReduceSum< Reduce< POLICY_TYPE >, double > _squaredError{ 0 };
  RAJA::ReduceSum< Reduce< POLICY_TYPE >, double > _baseNormSquared{ 0 };
};


template< typename POLICY_TYPE, typename T, typename U, int NDIM, int USD >
void checkDiffs(
  ArrayView< T const, NDIM, USD > const & base,
  ArrayView< U const, NDIM, USD > const & newValues,
  double const rtol,
  double const atol )
{
  for( int dim = 0; dim < NDIM; ++dim )
  {
    LVARRAY_ERROR_IF_NE( base.size( dim ), newValues.size( dim ) );
    LVARRAY_ERROR_IF_NE( base.strides()[ dim ], newValues.strides()[ dim ] );
  }

  DiffChecker< POLICY_TYPE > realChecker( rtol, atol );
  DiffChecker< POLICY_TYPE > imagChecker( rtol, atol );

  forAll< DefaultPolicy< POLICY_TYPE > >( base.size(), [=] LVARRAY_HOST_DEVICE ( IndexType const idx )
  {
    realChecker.witness( std::real( base.data()[ idx ] ), std::real( newValues.data()[ idx ] ) );
    imagChecker.witness( std::imag( base.data()[ idx ] ), std::imag( newValues.data()[ idx ] ) );
  } );

  LVARRAY_LOG( "Real part: " << realChecker.report() );
  LVARRAY_LOG( "Imag part: " << imagChecker.report() );
}












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
Array4d< std::complex< double > > computeR(
  int const r1GridSize,
  int const r2GridSize,
  std::vector< OchiBasisFunction< double > > const & basisFunctions )
{
  TCSCF_MARK_FUNCTION;

  return integration::integrateAllR1R2< false, double >( r1GridSize, r2GridSize, basisFunctions,
    [] ( Cartesian< double > const & r1, Cartesian< double > const & r2 )
    {
      double const r12 = (r1 - r2).r();
      return 1 / r12;
    }
  );
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
  TCSCF_MARK_FUNCTION;

  // LVARRAY_UNUSED_VARIABLE( r1GridSize );
  // LVARRAY_UNUSED_VARIABLE( r2GridSize );
  // LVARRAY_UNUSED_VARIABLE( basisFunctions );
  // LVARRAY_UNUSED_VARIABLE( u );

  // return {};

  return integration::integrateAllR1R2< false, double >( r1GridSize, r2GridSize, basisFunctions,
    [&u] ( Cartesian< double > const & r1, Cartesian< double > const & r2 )
    {
      return u.laplacian( r1, r2, false );
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
  TCSCF_MARK_FUNCTION;

  return integration::integrateAllR1R2< false, double >( r1GridSize, r2GridSize, basisFunctions,
    [&u] ( Cartesian< double > const & r1, Cartesian< double > const & r2 )
    {
      Cartesian< double > const grad1 = u.gradient( r1, r2, false );
      return dot( grad1, grad1 );
    }
  );
}

/**
 * 
 */
Array4d< std::complex< double > > computeDOppositeSpin(
  int const r1GridSize,
  int const r2GridSize,
  std::vector< OchiBasisFunction< double > > const & basisFunctions,
  jastrowFunctions::Ochi< double > const & u )
{
  TCSCF_MARK_FUNCTION;
  
  return integration::integrateAllR1R2< true, double >( r1GridSize, r2GridSize, basisFunctions,
    [&u] ( Cartesian< double > const & r1, Cartesian< double > const & r2 )
    {
      return u.gradient( r2, r1, false );
    }
  );
}


void compareLAndD(
  ArrayView4d< std::complex< double > const > const & LOppo,
  ArrayView4d< std::complex< double > const > const & DPOppo )
{
  TCSCF_MARK_FUNCTION;

  IndexType const nBasis = LOppo.size( 0 );
  Array4d< std::complex< double > > LFromD( nBasis, nBasis, nBasis, nBasis );

  for( int j = 0; j < nBasis; ++j )
  {
    for( int ell = 0; ell < nBasis; ++ell )
    {
      for( int i = 0; i < nBasis; ++i )
      {
        for( int m = 0; m < nBasis; ++m )
        {
          LFromD( j, ell, i, m ) = - conj( DPOppo( m, i, ell, j ) ) - DPOppo( ell, j, m, i );
        }
      }
    }
  }

  checkDiffs< Serial >( LOppo, LFromD.toViewConst(), 0, 1e-4 );
  abort();
}



/**
 * 
 */
void computeH2Prime(
  ArrayView4d< std::complex< double > > const & h2PrimeOppo,
  ArrayView4d< std::complex< double > > const & h2PrimeSame,
  ArrayView4d< std::complex< double > const > const & R,
  ArrayView4d< std::complex< double > const > const & GOppo,
  ArrayView4d< std::complex< double > const > const & DPOppo )
{
  TCSCF_MARK_FUNCTION;

  int const nBasis = h2PrimeOppo.size( 0 );
  for( int dim = 0; dim < 4; ++dim )
  {
    LVARRAY_ERROR_IF_NE( h2PrimeOppo.size( dim ), nBasis );
    LVARRAY_ERROR_IF_NE( h2PrimeSame.size( dim ), nBasis );
    LVARRAY_ERROR_IF_NE( R.size( dim ), nBasis );
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
          std::complex< double > const h2_oppo_jl_im =
            R( j, ell, i, m ) - GOppo( j, ell, i, m ) + DPOppo( ell, j, m, i ) - conj( DPOppo( m, i, ell, j ) );
          std::complex< double > const h2_oppo_lj_mi =
            R( ell, j, m, i ) - GOppo( ell, j, m, i ) + DPOppo( j, ell, i, m ) - conj( DPOppo( i, m, j, ell ) );

          h2PrimeOppo( j, ell, i, m ) = h2_oppo_jl_im + h2_oppo_lj_mi;

          std::complex< double > const h2_same_jl_im =
            R( j, ell, i, m ) - GOppo( j, ell, i, m ) / 4 + DPOppo( ell, j, m, i ) / 2 - conj( DPOppo( m, i, ell, j ) ) / 2;
          std::complex< double > const h2_same_lj_mi =
            R( ell, j, m, i ) - GOppo( ell, j, m, i ) / 4 + DPOppo( j, ell, i, m ) / 2 - conj( DPOppo( i, m, j, ell ) ) / 2;
          
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
  TCSCF_MARK_FUNCTION;

  constexpr double HF_LIMIT = -2.861679995612;

  int const Z = 2;
  int const nSpinUp = 1;
  int const nSpinDown = 1;

  double alpha = initialAlpha;

  std::vector< OchiBasisFunction< double > > basisFunctions;
  int const nBasis = createBasisFunctions( nMax, lMax, alpha, basisFunctions );

  LVARRAY_LOG( "nMax = " << nMax << ", lMax = " << lMax << ", nBasis = " << nBasis <<
               ", r1 grid size = " << r1GridSize << ", r2 grid size = " << r2GridSize << ", alpha = " << alpha );

  integration::QuadratureGrid< double > const coreGrid = integration::createGrid(
    integration::ChebyshevGauss< double >( 1000 ),
    integration::changeOfVariables::TreutlerAhlrichsM4< double >( 1, 0.9 ) );

  HF_CALCULATOR hfCalculator( nSpinUp, nSpinDown, basisFunctions.size() );

  Array1d< double > energies;

  int const nIter = 30;
  for( int iter = 0; iter < nIter; ++iter )
  {
    Array2d< double > const coreMatrix( nBasis, nBasis );
    fillCoreMatrix( coreGrid, Z, basisFunctions, coreMatrix );
    
    Array4d< std::complex< double > > R = computeR( r1GridSize, r2GridSize, basisFunctions );
  
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

      // Array4d< std::complex< double > > const LOppo = computeLOppositeSpin( r1GridSize, r2GridSize, basisFunctions, u );
      Array4d< std::complex< double > > const GOppo = computeGOppositeSpin( r1GridSize, r2GridSize, basisFunctions, u );
      Array4d< std::complex< double > > const DOppo = computeDOppositeSpin( r1GridSize, r2GridSize, basisFunctions, u );

      Array4d< std::complex< double > > const h2PrimeOppo( nBasis, nBasis, nBasis, nBasis );
      Array4d< std::complex< double > > const h2PrimeSame( nBasis, nBasis, nBasis, nBasis );

      computeH2Prime( h2PrimeOppo, h2PrimeSame, R, GOppo, DOppo );

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

  auto const [mean, standardDev] = meanAndStd( energies.toViewConst() );
  double const error = std::abs( HF_LIMIT - mean );
  printf( "\renergy = %.6F +/- %.2e Ht, error = %.2e Ht, alpha = %.6F                     \n", mean, standardDev, error, alpha );
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