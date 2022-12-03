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
int createBasisFunctions(
  int const nMax,
  int const lMax,
  int const Z,
  std::vector< HydrogenLikeBasisFunction< double > > & basisFunctions )
{
  basisFunctions.clear();

  for( int n = 1; n <= nMax; ++n )
  {
    for( int l = 0; l <= std::min( lMax, n ); ++l )
    {
      for( int m = -l; m <= l; ++m )
      {
        basisFunctions.emplace_back( Z, n, l, m );
      }
    }
  }

  return basisFunctions.size();
}

template< typename REAL >
void precompute(
  Array3d< REAL > & FjiSame,
  integration::QMCGrid< REAL, 3 > const & r1Grid,
  integration::QMCGrid< REAL, 2 > const & r2Grid )
{
  using Real = REAL;

  Array2d< Real > scalarIntegrand( r1Grid.nGrid(), r2Grid.nGrid() );

  precomputeIntegrand( scalarIntegrand, r1Grid, r2Grid,
    [] (Cartesian< Real > const & r1, Cartesian< Real > const & r2 )
    {
      return 1 / (r1 - r2).r();
    }
  );

  FjiSame = computeF( r1Grid, r2Grid, scalarIntegrand.toViewConst() );
}

template< typename REAL >
void precomputeTranscorrelated(
  Array3d< REAL > & FjiSame,
  Array3d< REAL > & FjiOppo,
  Array3d< Cartesian< std::complex< REAL > > > & VjiSame,
  Array3d< Cartesian< std::complex< REAL > > > & VjiOppo,
  jastrowFunctions::Ochi< REAL > const & u,
  integration::QMCGrid< REAL, 3 > const & r1Grid,
  integration::QMCGrid< REAL, 2 > const & r2Grid,
  integration::QMCGrid< REAL, 3 > const & r3Grid )
{
  using Real = REAL;

  Array2d< Real > scalarIntegrand( r1Grid.nGrid(), r2Grid.nGrid() );

  precomputeIntegrand( scalarIntegrand, r1Grid, r2Grid,
    [&u] ( Cartesian< Real > const & r1, Cartesian< Real > const & r2 )
    {
      Cartesian< Real > const grad12 = u.gradient( r1, r2, true );
      Cartesian< Real > const grad21 = u.gradient( r2, r1, true );
      return 2 / (r1 - r2).r() - dot( grad12, grad12 ) - dot( grad21, grad21 );
    }
  );

  FjiSame = computeF( r1Grid, r2Grid, scalarIntegrand.toViewConst() );

  precomputeIntegrand( scalarIntegrand, r1Grid, r2Grid,
    [&u] ( Cartesian< Real > const & r1, Cartesian< Real > const & r2 )
    {
      Cartesian< Real > const grad12 = u.gradient( r1, r2, false );
      Cartesian< Real > const grad21 = u.gradient( r2, r1, false );
      return 2 / (r1 - r2).r() - dot( grad12, grad12 ) - dot( grad21, grad21 );
    }
  );

  FjiOppo = computeF( r1Grid, r2Grid, scalarIntegrand.toViewConst() );

  Array2d< Cartesian< Real > > vectorIntegrand( r1Grid.nGrid(), r3Grid.nGrid() );

  precomputeIntegrand( vectorIntegrand, r1Grid, r3Grid,
    [&u] ( Cartesian< Real > const & r1, Cartesian< Real > const & r2 )
    {
      auto const grad = u.gradient( r1, r2, true );
      return grad;
    }
  );

  VjiSame = computeV( r1Grid, r3Grid, vectorIntegrand.toViewConst() );

  precomputeIntegrand( vectorIntegrand, r1Grid, r3Grid,
    [&u] ( Cartesian< Real > const & r1, Cartesian< Real > const & r2 )
    {
      auto const grad = u.gradient( r1, r2, false );
      return grad;
    }
  );

  VjiOppo = computeV( r1Grid, r3Grid, vectorIntegrand.toViewConst() );
}


template< typename HF_CALCULATOR >
void ochiNewHF(
  int const Z,
  double const hfEnergy,
  double const energy,
  int const nMax,
  int const lMax,
  double const alpha,
  int const r1GridSize,
  int const r2GridSize )
{
  using Real = double;
  using Complex = std::complex< Real >;

  int const nSpinUp = Z / 2;
  int const nSpinDown = Z - nSpinUp;

  std::vector< OchiBasisFunction< Real > > basisFunctions;
  int const nBasis = createBasisFunctions( nMax, lMax, alpha, basisFunctions );

  LVARRAY_LOG( "Z = " << Z << ", nMax = " << nMax << ", lMax = " << lMax << ", nBasis = " << nBasis <<
                ", r1 grid size = " << r1GridSize << ", r2 grid size = " << r2GridSize << ", alpha = " << alpha );

  HF_CALCULATOR hfCalculator( nSpinUp, nSpinDown, nBasis );

  integration::QuadratureGrid< Real > const coreGrid = integration::createGrid(
    integration::ChebyshevGauss< Real >( 1000 ),
    integration::changeOfVariables::TreutlerAhlrichsM4< Real >( 1, 0.9 ) );
  
  Array3d< Real > FjiSame;
  Array3d< Real > FjiOppo;
  Array3d< Cartesian< Complex > > VjiSame;
  Array3d< Cartesian< Complex > > VjiOppo;
  
  Real const a = 1.5;
  Real const a12 = a;
  Array2d< int > S( 1, 3 );
  S( 0, 0 ) = 1;

  Array2d< Real > c( 1, 2 );
  c( 0, false ) = a12 / 2;
  c( 0, true ) = a12 / 4;

  jastrowFunctions::Ochi< Real > const u { a, a12, c, S };
  
  Array2d< Real > const coreMatrix( nBasis, nBasis );
  fillCoreMatrix( coreGrid, Z, basisFunctions, coreMatrix );

  Array1d< Real > energies;
  for( int errorIter = 0; errorIter < 10; ++errorIter )
  {
    integration::QMCGrid< Real, 3 > r1Grid( r1GridSize );
    integration::QMCGrid< Real, 3 > r3Grid( r1GridSize );
    integration::QMCGrid< Real, 2 > r2Grid( r2GridSize );
    
    r1Grid.setBasisFunctions( basisFunctions, hfCalculator.needsGradients() );
    r3Grid.setBasisFunctions( basisFunctions, hfCalculator.needsGradients() );
    r2Grid.setBasisFunctions( basisFunctions, hfCalculator.needsGradients() );

    if constexpr ( hfCalculator.needsGradients() )
    {
      precomputeTranscorrelated( FjiSame, FjiOppo, VjiSame, VjiOppo, u, r1Grid, r2Grid, r3Grid );
      energies.emplace_back( hfCalculator.compute( true, {}, coreMatrix, r1Grid, FjiSame, FjiOppo, VjiSame, VjiOppo, basisFunctions ) );
    }
    else
    {
      precompute( FjiSame, r1Grid, r2Grid );
      energies.emplace_back( hfCalculator.compute( true, {}, coreMatrix, r1Grid, FjiSame, basisFunctions ) );
    }
  }

  auto const [mean, standardDev] = meanAndStd( energies.toViewConst() );
  double const error = std::abs( energy - mean );
  double const correlationEnergy = energy - hfEnergy;
  double const percentOfCorrelation = (mean - hfEnergy) / correlationEnergy * 100;
  printf( "\tenergy = %.6F +/- %.2e Ht, error = %.2e Ht, percent of correlation = %.1F%%\n", mean, standardDev, error, percentOfCorrelation );
}

// Taken from https://aip.scitation.org/doi/pdf/10.1063/1.458750
std::vector< std::tuple< int, double, double, double > > atoms {
    { 2, 1.355,  -2.861700, -2.903700 }
  // , { 3,  -7.432700, -7.478100 }
  // , { 4,  -14.57300, -14.66730 }
  // , { 5, 2.400,  -24.52910, -24.65390 }
  // , { 6,  -37.68860, -37.84510 }
  // , { 7,  -54.40090, -54.58950 }
  // , { 8,  -74.80940, -75.06730 }
  // , { 9,  -99.40930, -99.73130 }
  // , { 10, -128.5471, -128.9370 }
};


TEST( NewHartreeFock, RestrictedClosedShell )
{
  TCSCF_MARK_SCOPE( "New restricted closed shell" );

  for( auto const & [Z, alpha, hfEnergy, energy] : atoms )
  {
    if( Z % 2 != 0 )
    {
      continue;
    }

    ochiNewHF< RCSHartreeFock< std::complex< double > > >( Z, hfEnergy, energy, clo.nMax, clo.lMax, alpha, clo.r1GridSize, clo.r2GridSize );
  }
}

TEST( NewHartreeFock, UnrestrictedOpenShell )
{
  TCSCF_MARK_SCOPE( "New unrestricted open shell" );

  for( auto const & [Z, alpha, hfEnergy, energy] : atoms )
  {
    ochiNewHF< UOSHartreeFock< std::complex< double > > >( Z, hfEnergy, energy, clo.nMax, clo.lMax, alpha, clo.r1GridSize, clo.r2GridSize );
  }
}

TEST( NewHartreeFock, Transcorrelated )
{
  TCSCF_MARK_SCOPE( "New transcorrelated" );

  for( auto const & [Z, alpha, hfEnergy, energy] : atoms )
  {
    ochiNewHF< TCHartreeFock< std::complex< double > > >( Z, hfEnergy, energy, clo.nMax, clo.lMax, alpha, clo.r1GridSize, clo.r2GridSize );
  }
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