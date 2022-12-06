#include "../setup.hpp"
#include "../caliperInterface.hpp"
#include "../HartreeFock.hpp"
#include "../HydrogenLikeBasis.hpp"
#include "../OchiBasis.hpp"
#include "../SlaterTypeOrbital.hpp"
#include "../jastrowFunctions.hpp"
#include "../integration/integrateAll.hpp"
#include "../integration/ChebyshevGauss.hpp"
#include "../integration/changeOfVariables.hpp"

#include "testingCommon.hpp"

#include "gsl/gsl_multimin.h"

namespace tcscf::testing
{

CommandLineOptions clo {};

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

/**
 * 
 */
int createBasisFunctions(
  int const nMax,
  int const lMax,
  double const alpha,
  std::vector< SlaterTypeOrbital< double > > & basisFunctions )
{
  basisFunctions.clear();

  // for( int n = 1; n <= nMax; ++n )
  // {
  //   for( int l = 0; l <= std::min( lMax, n ); ++l )
  //   {
  //     for( int m = -l; m <= l; ++m )
  //     {
  //       basisFunctions.emplace_back( alpha, n, l, m );
  //     }
  //   }
  // }

  LVARRAY_UNUSED_VARIABLE( nMax );
  LVARRAY_UNUSED_VARIABLE( lMax );
  LVARRAY_UNUSED_VARIABLE( alpha );
  basisFunctions.emplace_back( 1.75, 1, 0, 0 );
  basisFunctions.emplace_back( 0.70, 1, 0, 0 );
  basisFunctions.emplace_back( 3.7, 1, 0, 0 );
  
  basisFunctions.emplace_back( 2.7, 2, 0, 0 );
  basisFunctions.emplace_back( 1.0, 2, 0, 0 );

  return basisFunctions.size();
}

/**
 * 
 */
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

/**
 * 
 */
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

/**
 * 
 */
template< typename HF_CALCULATOR >
double ochiNewHF(
  int const Z,
  double const hfEnergy,
  double const energy,
  std::vector< SlaterTypeOrbital< double > > const & basisFunctions,
  int const r1GridSize,
  int const r2GridSize )
{
  using Real = double;
  using Complex = std::complex< Real >;

  int const nSpinUp = Z / 2;
  int const nSpinDown = Z - nSpinUp;

  // std::vector< OchiBasisFunction< double > > basisFunctions;
  // createBasisFunctions( 9, 0, 1.355, basisFunctions );

  int const nBasis = basisFunctions.size();

  LVARRAY_LOG( "Z = " << Z << ", nBasis = " << nBasis <<
                ", r1 grid size = " << r1GridSize << ", r2 grid size = " << r2GridSize );

  HF_CALCULATOR hfCalculator( nSpinUp, nSpinDown, nBasis );
  
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
  
  Array2d< Complex, RAJA::PERM_JI > const overlapMatrix( nBasis, nBasis );
  Array2d< Real > const coreMatrix( nBasis, nBasis );
  {
    integration::QuadratureGrid< Real > const coreGrid = integration::createGrid(
      integration::ChebyshevGauss< Real >( 1000 ),
      integration::changeOfVariables::TreutlerAhlrichsM4< Real >( 1, 0.9 ) );
    for( int i = 0; i < nBasis; ++i )
    {
      for( int j = 0; j < nBasis; ++j )
      {
        overlapMatrix( i, j ) = overlap( basisFunctions[ i ], basisFunctions[ j ] );
        coreMatrix( i, j ) = coreMatrixElement( coreGrid, Z, basisFunctions[ i ], basisFunctions[ j ] );
      }
    }
  }

  int convergenceRepeats = 0;
  Array1d< Real > energies;
  for( int errorIter = 0; errorIter < 10; ++errorIter )
  {
    integration::QMCGrid< Real, 3 > r1Grid( r1GridSize );
    integration::QMCGrid< Real, 3 > r3Grid( r1GridSize );
    integration::QMCGrid< Real, 2 > r2Grid( r2GridSize );

    r1Grid.setBasisFunctions( basisFunctions, hfCalculator.needsGradients() );
    r3Grid.setBasisFunctions( basisFunctions, hfCalculator.needsGradients() );
    r2Grid.setBasisFunctions( basisFunctions, hfCalculator.needsGradients() );

    try
    {
      if constexpr ( hfCalculator.needsGradients() )
      {
        precomputeTranscorrelated( FjiSame, FjiOppo, VjiSame, VjiOppo, u, r1Grid, r2Grid, r3Grid );
        energies.emplace_back( hfCalculator.compute( overlapMatrix, coreMatrix, r1Grid, FjiSame, FjiOppo, VjiSame, VjiOppo, true ) );
      }
      else
      {
        precompute( FjiSame, r1Grid, r2Grid );
        energies.emplace_back( hfCalculator.compute( overlapMatrix, coreMatrix, r1Grid, FjiSame, true ) );
      }

      convergenceRepeats = 0;
    }
    catch( LvArray::dense::Error const & e )
    {
      LVARRAY_LOG( "Caught a linear algebra exception: " << e.what() );
      return std::numeric_limits< double >::max();
    }
    catch( ConvergenceError const & e )
    {
      --errorIter;
      ++convergenceRepeats;

      if( convergenceRepeats == 3 )
      {
        LVARRAY_LOG( "Could not converge for the third time, exiting." );
        return std::numeric_limits< double >::max();
      }

      LVARRAY_LOG( "Convergence error" );
    }
  }

  auto const [mean, standardDev] = meanAndStd( energies.toViewConst() );
  double const error = std::abs( energy - mean );
  double const correlationEnergy = energy - hfEnergy;
  double const percentOfCorrelation = (mean - hfEnergy) / correlationEnergy * 100;
  printf( "\tenergy = %.6F +/- %.2e Ht, error = %.2e Ht, percent of correlation = %.1F%%\n", mean, standardDev, error, percentOfCorrelation );

  return mean;
}

/**
 */
template< typename HF_SOLVER >
void optimizeOrbitalExponents(
  int const Z,
  std::vector< SlaterTypeOrbital< double > > & basisFunctions,
  double const hfEnergy,
  double const energy )
{
  /* Starting point */
  gsl_vector * alphas = gsl_vector_alloc( basisFunctions.size() );
  for( std::size_t i = 0; i < basisFunctions.size(); ++i )
  {
    gsl_vector_set( alphas, i, basisFunctions[ i ].alpha );
  }

  /* Set initial step sizes to 1 */
  gsl_vector * ss = gsl_vector_alloc( basisFunctions.size() );
  gsl_vector_set_all( ss, 1.0 );

  /* Initialize method and iterate */
  using ParamType = std::tuple< std::vector< SlaterTypeOrbital< double > > &, int, double, double >;
  ParamType params{ basisFunctions, Z, hfEnergy, energy };
  gsl_multimin_function minex_func;
  minex_func.n = basisFunctions.size();
  minex_func.params = &params;

  minex_func.f = [] ( gsl_vector const * x, void * params )
  {
    auto * paramsP = reinterpret_cast< ParamType * >( params );
    
    std::vector< SlaterTypeOrbital< double > > & basisFunctions = std::get< 0 >( *paramsP );
    int const Z = std::get< 1 >( *paramsP );
    double const hfEnergy = std::get< 2 >( *paramsP );
    double const energy = std::get< 3 >( *paramsP );
    
    for( std::size_t i = 0; i < basisFunctions.size(); ++i )
    {
      basisFunctions[ i ].resetOrbitalExponent( gsl_vector_get( x, i ) );
    }

    return ochiNewHF< HF_SOLVER >( Z, hfEnergy, energy, basisFunctions, clo.r1GridSize, clo.r2GridSize );
  };

  gsl_multimin_fminimizer * s = gsl_multimin_fminimizer_alloc( gsl_multimin_fminimizer_nmsimplex2, basisFunctions.size() );
  gsl_multimin_fminimizer_set( s, &minex_func, alphas, ss );
 
  int status;
  for( int iter = 0; iter < 100; ++iter )
  {
    status = gsl_multimin_fminimizer_iterate( s );

    if( status )
    {
      break;
    }

    double size = gsl_multimin_fminimizer_size( s );
    status = gsl_multimin_test_size( size, 1e-2 );

    std::cout << "Best so far: " << std::setprecision(10);
    for( std::size_t i = 0; i < basisFunctions.size(); ++i )
    {
      basisFunctions[ i ].resetOrbitalExponent( gsl_vector_get( s->x, i ) );
      std::cout << basisFunctions[ i ].alpha << ", ";
    }

    std::cout << std::endl;

    if (status == GSL_SUCCESS)
    {
      printf ("converged to minimum\n");
    }
    if( status != GSL_CONTINUE )
    {
      break;
    }
  }

  for( std::size_t i = 0; i < basisFunctions.size(); ++i )
  {
    basisFunctions[ i ].resetOrbitalExponent( gsl_vector_get( s->x, i ) );
    LVARRAY_LOG( basisFunctions[ i ].alpha );
  }

  gsl_vector_free( alphas );
  gsl_vector_free( ss );
  gsl_multimin_fminimizer_free( s );
}

/**
 */
std::vector< SlaterTypeOrbital< double > > getBasisFunctions( int const Z )
{
  std::unordered_map< int, std::vector< SlaterTypeOrbital< double > > > const basisFunctions
  {
    { 2, { { 2.822085428, 1, 0, 0 },
           { 1.440559397, 1, 0, 0 } } },
    { 3, { { 4.204771337, 1, 0, 0 },
           { 2.202175264, 1, 0, 0 },
           { 2.257048049, 2, 0, 0 },
           { 0.669288846, 2, 0, 0 } } },
    { 4, { { 5.328935439, 1, 0, 0 },
           { 2.786911325, 1, 0, 0 },
           { 2.638473675, 2, 0, 0 },
           { 0.927608881, 2, 0, 0 },
           { 1.242131697, 3, 0, 0 },
           { 1.862040748, 3, 0, 0 } } },
    { 5, { { 5.085032820, 1, 0, +0 },
           { 2.878504560, 1, 0, +0 },
           { 1.789508005, 2, 0, +0 },
           { 1.101417655, 2, 0, +0 },
           { 2.385883417, 2, 1, -1 },
           { 1.061275537, 2, 1, -1 },
           { 2.385883417, 2, 1, +0 },
           { 1.061275537, 2, 1, +0 },
           { 2.385883417, 2, 1, +1 },
           { 1.061275537, 2, 1, +1 } } },
    { 6, { { 6.739069556, 1, 0, 0 },
           { 2.631220383, 1, 0, 0 },
           { 3.898095485, 1, 0, 0 },
           { 2.008361494, 2, 0, 0 },
           { 1.242563282, 2, 0, 0 },
           { 2.458624422, 2, 0, 0 },
           { 2.796921480, 2, 1, -1 },
           { 1.179855443, 2, 1, -1 },
           { 2.407663971, 2, 1, -1 },
           { 2.796921480, 2, 1, +0 },
           { 1.179855443, 2, 1, +0 },
           { 2.407663971, 2, 1, +0 },
           { 2.796921480, 2, 1, +1 },
           { 1.179855443, 2, 1, +1 },
           { 2.407663971, 2, 1, +1 } } }
  };

  return basisFunctions.at( Z );
}


// Energies taken from https://aip.scitation.org/doi/pdf/10.1063/1.458750
std::vector< std::tuple< int, double, double > > atoms {
  //   { 2, -2.861700, -2.903700 }
  // , { 3, -7.432700, -7.478100 }
  { 4,  -14.57300, -14.66730 }
  // , { 5, -24.52910, -24.65390 }
  // , { 6,  -37.68860, -37.84510 }
  // , { 7,  -54.40090, -54.58950 }
  // , { 8,  -74.80940, -75.06730 }
  // , { 9,  -99.40930, -99.73130 }
  // , { 10, -128.5471, -128.9370 }
};


TEST( NewHartreeFock, OptimizeOrbitalExponents )
{
  TCSCF_MARK_SCOPE( "New restricted closed shell" );

  for( auto const & [Z, hfEnergy, energy] : atoms )
  {
    std::vector< SlaterTypeOrbital< double > > basisFunctions = getBasisFunctions( Z );

    optimizeOrbitalExponents< UOSHartreeFock< std::complex< double > > >( Z, basisFunctions, hfEnergy, energy );

    std::cout << std::setprecision( 10 );
    for( auto const & basis : basisFunctions )
    {
      LVARRAY_LOG( "{ " << basis.alpha << ", " << basis.n << ", " << basis.l << ", " << basis.m << " }"  );
    }
  }
}


TEST( NewHartreeFock, RestrictedClosedShell )
{
  TCSCF_MARK_SCOPE( "New restricted closed shell" );

  for( auto const & [Z, hfEnergy, energy] : atoms )
  {
    if( Z % 2 != 0 )
    {
      continue;
    }

    std::vector< SlaterTypeOrbital< double > > basisFunctions = getBasisFunctions( Z );
    ochiNewHF< RCSHartreeFock< std::complex< double > > >( Z, hfEnergy, energy, basisFunctions, clo.r1GridSize, clo.r2GridSize );
  }
}

TEST( NewHartreeFock, UnrestrictedOpenShell )
{
  TCSCF_MARK_SCOPE( "New unrestricted open shell" );

  for( auto const & [Z, hfEnergy, energy] : atoms )
  {
    std::vector< SlaterTypeOrbital< double > > basisFunctions = getBasisFunctions( Z );
    ochiNewHF< UOSHartreeFock< std::complex< double > > >( Z, hfEnergy, energy, basisFunctions, clo.r1GridSize, clo.r2GridSize );
  }
}

TEST( NewHartreeFock, Transcorrelated )
{
  TCSCF_MARK_SCOPE( "New unrestricted open shell" );

  for( auto const & [Z, hfEnergy, energy] : atoms )
  {
    std::vector< SlaterTypeOrbital< double > > basisFunctions = getBasisFunctions( Z );
    ochiNewHF< TCHartreeFock< std::complex< double > > >( Z, hfEnergy, energy, basisFunctions, clo.r1GridSize, clo.r2GridSize );
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