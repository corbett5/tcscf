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

int LOG_LEVEL = 0;


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
  integration::QMCGrid< double, 3 > const & r1Grid,
  integration::QMCGrid< double, 2 > const & r2Grid )
{
  TCSCF_MARK_FUNCTION;

  return integration::integrateAllR1R2< false, double >( r1Grid, r2Grid,
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
  integration::QMCGrid< double, 3 > const & r1Grid,
  integration::QMCGrid< double, 2 > const & r2Grid,
  jastrowFunctions::Ochi< double > const & u )
{
  TCSCF_MARK_FUNCTION;

  return integration::integrateAllR1R2< false, double >( r1Grid, r2Grid,
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
  integration::QMCGrid< double, 3 > const & r1Grid,
  integration::QMCGrid< double, 2 > const & r2Grid,
  jastrowFunctions::Ochi< double > const & u )
{
  TCSCF_MARK_FUNCTION;

  return integration::integrateAllR1R2< false, double >( r1Grid, r2Grid,
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
  integration::QMCGrid< double, 3 > const & r1Grid,
  integration::QMCGrid< double, 2 > const & r2Grid,
  jastrowFunctions::Ochi< double > const & u )
{
  TCSCF_MARK_FUNCTION;
  
  return integration::integrateAllR1R2< true, double >( r1Grid, r2Grid,
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
  LVARRAY_ERROR( "No longer using DPOppo, need to fix." );

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
  ArrayView4d< std::complex< double > const > const & DOppo )
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
            R( j, ell, i, m ) - GOppo( j, ell, i, m ) + DOppo( j, ell, i, m ) - conj( DOppo( i, m, j, ell ) );
          std::complex< double > const h2_oppo_lj_mi =
            R( ell, j, m, i ) - GOppo( ell, j, m, i ) + DOppo( ell, j, m, i ) - conj( DOppo( m, i, ell, j ) );

          h2PrimeOppo( j, ell, i, m ) = h2_oppo_jl_im + h2_oppo_lj_mi;

          std::complex< double > const h2_same_jl_im =
            R( j, ell, i, m ) - GOppo( j, ell, i, m ) / 4 + DOppo( j, ell, i, m ) / 2 - conj( DOppo( i, m, j, ell ) ) / 2;
          std::complex< double > const h2_same_lj_mi =
            R( ell, j, m, i ) - GOppo( ell, j, m, i ) / 4 + DOppo( ell, j, m, i ) / 2 - conj( DOppo( m, i, ell, j ) ) / 2;
          
          h2PrimeSame( j, ell, i, m ) = h2_same_jl_im + h2_same_lj_mi;
        }
      }
    }
  }
}

void computeH2PrimeXXX(
  ArrayView4d< std::complex< double > > const & h2PrimeOppo,
  ArrayView4d< std::complex< double > > const & h2PrimeSame,
  ArrayView4d< std::complex< double > const > const & R,
  ArrayView4d< std::complex< double > const > const & GOppo,
  ArrayView4d< std::complex< double > const > const & LOppo )
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
            R( j, ell, i, m ) + LOppo( j, ell, i, m ) - GOppo( j, ell, i, m );
          std::complex< double > const h2_oppo_lj_mi =
            R( ell, j, m, i ) + LOppo( ell, j, m, i ) - GOppo( ell, j, m, i );

          h2PrimeOppo( j, ell, i, m ) = h2_oppo_jl_im + h2_oppo_lj_mi;

          std::complex< double > const h2_same_jl_im =
            R( j, ell, i, m ) + LOppo( j, ell, i, m ) / 2 - GOppo( j, ell, i, m ) / 4;
          std::complex< double > const h2_same_lj_mi =
            R( ell, j, m, i ) + LOppo( ell, j, m, i ) / 2 - GOppo( ell, j, m, i ) / 4;
          
          h2PrimeSame( j, ell, i, m ) = h2_same_jl_im + h2_same_lj_mi;
        }
      }
    }
  }
}

// void computeH3Prime(
//   ArrayView6d< std::complex< double > > const & h3PrimeUpUpUp,
//   ArrayView6d< std::complex< double > > const & hePrimeUpDownDown,
//   ArrayView6d< std::complex< double > > const & h3PrimeUpUpDown,
//   ArrayView6d< std::complex< double > > const & h3PrimeUpDownUp,
//   ArrayView6d< std::complex< double > const > const & integrals )
// {

// }

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
  using Real = double;
  using Complex = std::complex< Real >;

  constexpr double HF_LIMIT = -2.861679995612;

  int const Z = 2;
  int const nSpinUp = 1;
  int const nSpinDown = 1;

  Real alpha = initialAlpha;

  std::vector< OchiBasisFunction< Real > > basisFunctions;
  int const nBasis = createBasisFunctions( nMax, lMax, alpha, basisFunctions );

  LVARRAY_LOG( "nMax = " << nMax << ", lMax = " << lMax << ", nBasis = " << nBasis <<
               ", r1 grid size = " << r1GridSize << ", r2 grid size = " << r2GridSize << ", alpha = " << alpha );

  integration::QuadratureGrid< Real > const coreGrid = integration::createGrid(
    integration::ChebyshevGauss< Real >( 1000 ),
    integration::changeOfVariables::TreutlerAhlrichsM4< Real >( 1, 0.9 ) );

  HF_CALCULATOR hfCalculator( nSpinUp, nSpinDown, basisFunctions.size() );

  Array1d< Real > energies;

  int const nIter = 30;
  for( int iter = 0; iter < nIter; ++iter )
  {
    Array2d< Real > const coreMatrix( nBasis, nBasis );
    fillCoreMatrix( coreGrid, Z, basisFunctions, coreMatrix );

    integration::QMCGrid< Real, 3 > r1Grid( r1GridSize );
    r1Grid.setBasisFunctions( basisFunctions, false );

    integration::QMCGrid< Real, 2 > r2Grid( r2GridSize );
    r2Grid.setBasisFunctions( basisFunctions, std::is_same_v< HF_CALCULATOR, TCHartreeFock< Complex > > );
    
    Array4d< Complex > R = computeR( r1Grid, r2Grid );
  
    std::cout << std::setprecision( 10 );

    Real energy = 0;
    if constexpr ( std::is_same_v< HF_CALCULATOR, TCHartreeFock< Complex > > )
    {
      Real const a = 1.5;
      Real const a12 = a;
      Array2d< int > S( 1, 3 );
      S( 0, 0 ) = 1;

      Array2d< Real > c( 1, 2 );
      c( 0, false ) = a12 / 2;
      c( 0, true ) = a12 / 4;

      jastrowFunctions::Ochi< Real > const u { a, a12, c, S };

      // Array4d< Complex > const LOppo = computeLOppositeSpin( r1Grid, r2Grid, u );
      Array4d< Complex > const GOppo = computeGOppositeSpin( r1Grid, r2Grid, u );
      Array4d< Complex > const DOppo = computeDOppositeSpin( r1Grid, r2Grid, u );

      // threeElectronIntegrals( r1Grid, r1Grid, u );

      Array4d< Complex > const h2PrimeOppo( nBasis, nBasis, nBasis, nBasis );
      Array4d< Complex > const h2PrimeSame( nBasis, nBasis, nBasis, nBasis );

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

template< typename REAL >
void precompute(
  ArrayView2d< REAL > const & fSame,
  integration::QMCGrid< REAL, 3 > const & r1Grid,
  integration::QMCGrid< REAL, 2 > const & r2Grid )
{
  using Real = REAL;

  precomputeIntegrand( fSame, r1Grid, r2Grid,
    [] (Cartesian< Real > const & r1, Cartesian< Real > const & r2 )
    {
      return 1 / (r1 - r2).r();
    }
  );
}

template< typename REAL >
void precomputeTranscorrelated(
  ArrayView2d< REAL > const & fSame,
  ArrayView2d< REAL > const & fOppo,
  ArrayView2d< Cartesian< REAL > > const & vSame21,
  ArrayView2d< Cartesian< REAL > > const & vOppo21,
  ArrayView2d< Cartesian< REAL > > const & vSame12,
  ArrayView2d< Cartesian< REAL > > const & vOppo12,
  jastrowFunctions::Ochi< REAL > const & u,
  integration::QMCGrid< REAL, 3 > const & r1Grid,
  integration::QMCGrid< REAL, 2 > const & r2Grid )
{
  using Real = REAL;

  precomputeIntegrand( fSame, r1Grid, r2Grid,
    [&u] (Cartesian< Real > const & r1, Cartesian< Real > const & r2 )
    {
      Cartesian< Real > const grad = u.gradient( r1, r2, true );
      return 1 / (r1 - r2).r() + u.laplacian( r1, r2, true ) - dot( grad, grad );
    }
  );

  precomputeIntegrand( fOppo, r1Grid, r2Grid,
    [&u] (Cartesian< Real > const & r1, Cartesian< Real > const & r2 )
    {
      Cartesian< Real > const grad = u.gradient( r1, r2, false );
      return 1 / (r1 - r2).r() + u.laplacian( r1, r2, false ) - dot( grad, grad );
    }
  );

  precomputeIntegrand( vSame21, r1Grid, r2Grid,
    [&u] (Cartesian< Real > const & r1, Cartesian< Real > const & r2 )
    {
      // LVARRAY_UNUSED_VARIABLE( u );
      // LVARRAY_UNUSED_VARIABLE( r1 );
      // LVARRAY_UNUSED_VARIABLE( r2 );
      // return Cartesian< Real > {};

      auto const grad = u.gradient( r1, r2, true );
      return grad + grad;
    }
  );

  precomputeIntegrand( vOppo21, r1Grid, r2Grid,
    [&u] (Cartesian< Real > const & r1, Cartesian< Real > const & r2 )
    {
      // LVARRAY_UNUSED_VARIABLE( u );
      // LVARRAY_UNUSED_VARIABLE( r1 );
      // LVARRAY_UNUSED_VARIABLE( r2 );
      // return Cartesian< Real > {};

      auto const grad = u.gradient( r1, r2, false );
      return grad + grad;
    }
  );

  precomputeIntegrand( vSame12, r2Grid, r1Grid,
    [&u] (Cartesian< Real > const & r1, Cartesian< Real > const & r2 )
    {
      // LVARRAY_UNUSED_VARIABLE( u );
      // LVARRAY_UNUSED_VARIABLE( r1 );
      // LVARRAY_UNUSED_VARIABLE( r2 );
      // return Cartesian< Real > {};

      auto const grad = u.gradient( r1, r2, true );
      return grad + grad;
    }
  );

  precomputeIntegrand( vOppo12, r2Grid, r1Grid,
    [&u] (Cartesian< Real > const & r1, Cartesian< Real > const & r2 )
    {
      // LVARRAY_UNUSED_VARIABLE( u );
      // LVARRAY_UNUSED_VARIABLE( r1 );
      // LVARRAY_UNUSED_VARIABLE( r2 );
      // return Cartesian< Real > {};

      auto const grad = u.gradient( r1, r2, false );
      return grad + grad;
    }
  );
}


template< typename HF_CALCULATOR >
void ochiNewHF(
  int const nMax,
  int const lMax,
  double const initialAlpha,
  int const r1GridSize,
  int const r2GridSize )
{
  using Real = double;
  using Complex = std::complex< Real >;

  int const Z = 2;
  int const nSpinUp = 1;
  int const nSpinDown = 1;

  std::vector< OchiBasisFunction< Real > > basisFunctions;
  int const nBasis = createBasisFunctions( nMax, lMax, initialAlpha, basisFunctions );

  LVARRAY_LOG( "nMax = " << nMax << ", lMax = " << lMax << ", nBasis = " << nBasis <<
                ", r1 grid size = " << r1GridSize << ", r2 grid size = " << r2GridSize << ", alpha = " << initialAlpha );

  HF_CALCULATOR hfCalculator( nSpinUp, nSpinDown, nBasis );

  integration::QuadratureGrid< Real > const coreGrid = integration::createGrid(
    integration::ChebyshevGauss< Real >( 1000 ),
    integration::changeOfVariables::TreutlerAhlrichsM4< Real >( 1, 0.9 ) );

  integration::QMCGrid< Real, 3 > r1Grid( r1GridSize );
  r1Grid.setBasisFunctions( basisFunctions, hfCalculator.needsGradients() );
  
  integration::QMCGrid< Real, 2 > r2Grid( r2GridSize );
  r2Grid.setBasisFunctions( basisFunctions, hfCalculator.needsGradients() );

  Array2d< Real > scalarSame( r1Grid.nGrid(), r2Grid.nGrid() );
  Array2d< Real > scalarOppo( r1Grid.nGrid(), r2Grid.nGrid() );
  Array2d< Cartesian< Real > > vectorSame21( r1Grid.nGrid(), r2Grid.nGrid() );
  Array2d< Cartesian< Real > > vectorOppo21( r1Grid.nGrid(), r2Grid.nGrid() );

  Array2d< Cartesian< Real > > vectorSame12( r2Grid.nGrid(), r1Grid.nGrid() );
  Array2d< Cartesian< Real > > vectorOppo12( r2Grid.nGrid(), r1Grid.nGrid() );
  
  Real const a = 1.5;
  Real const a12 = a;
  Array2d< int > S( 1, 3 );
  S( 0, 0 ) = 1;

  Array2d< Real > c( 1, 2 );
  c( 0, false ) = a12 / 2;
  c( 0, true ) = a12 / 4;

  jastrowFunctions::Ochi< Real > const u { a, a12, c, S };

  if constexpr ( hfCalculator.needsGradients() )
  {
    precomputeTranscorrelated( scalarSame, scalarOppo, vectorSame21, vectorOppo21, vectorSame12, vectorOppo12, u, r1Grid, r2Grid );
  }
  else
  {
    precompute( scalarSame, r1Grid, r2Grid );
  }

  Real alpha = initialAlpha;
  bool converged = false;
  int const maxIter = 30;
  int iter = 0;
  for( iter = 0; iter < maxIter; ++iter )
  {
    createBasisFunctions( nMax, lMax, alpha, basisFunctions );

    Array2d< Real > const coreMatrix( nBasis, nBasis );
    fillCoreMatrix( coreGrid, Z, basisFunctions, coreMatrix );

    if constexpr ( hfCalculator.needsGradients() )
    {
      Array4d< Complex > R = computeR( r1Grid, r2Grid );

      Array4d< Complex > const GOppo = computeGOppositeSpin( r1Grid, r2Grid, u );
      Array4d< Complex > const DOppo = computeDOppositeSpin( r1Grid, r2Grid, u );

      // threeElectronIntegrals( r1Grid, r1Grid, u );

      Array4d< Complex > const h2PrimeOppo( nBasis, nBasis, nBasis, nBasis );
      Array4d< Complex > const h2PrimeSame( nBasis, nBasis, nBasis, nBasis );

      // Array4d< Complex > const LOppo = computeLOppositeSpin( r1Grid, r2Grid, u );
      // computeH2PrimeXXX( h2PrimeOppo, h2PrimeSame, R, GOppo, LOppo );

      computeH2Prime( h2PrimeOppo, h2PrimeSame, R, GOppo, DOppo );

      LVARRAY_LOG_VAR( hfCalculator.compute( true, {}, coreMatrix, h2PrimeSame, h2PrimeOppo, r1Grid, r2Grid, scalarSame, scalarOppo, vectorSame21, vectorOppo21, vectorSame12, vectorOppo12 ) );
    }
    else
    {
      LVARRAY_LOG_VAR( hfCalculator.compute( true, {}, coreMatrix, r1Grid, r2Grid, scalarSame ) );
    }

    Real const newAlpha = std::sqrt( -2 * hfCalculator.highestOccupiedOrbitalEnergy() );
    LVARRAY_LOG_VAR( newAlpha );

    if( LOG_LEVEL > 1 )
    {
      LVARRAY_LOG( "\t\t number of loops to SCF convergence = " << hfCalculator.numberOfConvergenceLoops() );
    }

    if( std::abs( newAlpha - alpha ) < 1e-6 )
    {
      converged = true;
      break;
    }

    alpha = newAlpha;
  }

  LVARRAY_ERROR_IF( !converged, "Did not converge." );

  // createBasisFunctions( nMax, lMax, alpha, basisFunctions );

  // Array2d< Real > const coreMatrix( nBasis, nBasis );
  // fillCoreMatrix( coreGrid, Z, basisFunctions, coreMatrix );

  // Array1d< Real > energies;
  // for( int errorIter = 0; errorIter < 10; ++errorIter )
  // {
  //   integration::QMCGrid< Real, 3 > newR1Grid( r1GridSize );
  //   newR1Grid.setBasisFunctions( basisFunctions, false );
    
  //   integration::QMCGrid< Real, 2 > newR2Grid( r2GridSize );
  //   newR2Grid.setBasisFunctions( basisFunctions, hfCalculator.needsGradients() );

  //   precomputeIntegrand( scalarIntegrand.toView(), newR1Grid, newR2Grid,
  //     [] (Cartesian< Real > const & r1, Cartesian< Real > const & r2 )
  //     {
  //       return 1 / (r1 - r2).r();
  //     }
  //   );

  //   energies.emplace_back( hfCalculator.compute( true, {}, coreMatrix, newR1Grid, newR2Grid, scalarIntegrand ) );
  // }

  // constexpr Real HF_LIMIT = -2.861679995612;
  // auto const [mean, standardDev] = meanAndStd( energies.toViewConst() );
  // double const error = std::abs( HF_LIMIT - mean );
  // printf( "energy = %.6F +/- %.2e Ht, error = %.2e Ht, alpha = %.6F, number of alpha iterations = %d\n", mean, standardDev, error, alpha, iter + 1 );
}


TEST( HartreeFock, RestrictedClosedShell )
{
  TCSCF_MARK_SCOPE( "Restricted closed shell" );

  ochiHF< RCSHartreeFock< std::complex< double > > >( clo.nMax, clo.lMax, clo.initialAlpha, clo.r1GridSize, clo.r2GridSize );
}

TEST( HartreeFock, UnrestrictedOpenShell )
{
  TCSCF_MARK_SCOPE( "Unrestricted Open Shell" );

  ochiHF< UOSHartreeFock< std::complex< double > > >( clo.nMax, clo.lMax, clo.initialAlpha, clo.r1GridSize, clo.r2GridSize );
}

TEST( HartreeFock, Transcorrelated )
{
  TCSCF_MARK_SCOPE( "Transcorrelated" );

  ochiHF< TCHartreeFock< std::complex< double > > >( clo.nMax, clo.lMax, clo.initialAlpha, clo.r1GridSize, clo.r2GridSize );
}

TEST( NewHartreeFock, RestrictedClosedShell )
{
  TCSCF_MARK_SCOPE( "New restricted closed shell" );

  ochiNewHF< RCSHartreeFock< std::complex< double > > >( clo.nMax, clo.lMax, clo.initialAlpha, clo.r1GridSize, clo.r2GridSize );
}

TEST( NewHartreeFock, UnrestrictedOpenShell )
{
  TCSCF_MARK_SCOPE( "New unrestricted open shell" );

  ochiNewHF< UOSHartreeFock< std::complex< double > > >( clo.nMax, clo.lMax, clo.initialAlpha, clo.r1GridSize, clo.r2GridSize );
}

TEST( NewHartreeFock, Transcorrelated )
{
  TCSCF_MARK_SCOPE( "New transcorrelated" );

  ochiNewHF< TCHartreeFock< std::complex< double > > >( clo.nMax, clo.lMax, clo.initialAlpha, clo.r1GridSize, clo.r2GridSize );
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

// clear; ninja HartreeFockTest && ./tests/HartreeFockTest -n9 -l0 -a 1.355 --r1 1000 --r2 2000 -c runtime-report,max_column_width=200