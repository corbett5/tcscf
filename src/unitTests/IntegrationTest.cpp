#include "../integration/quadrature.hpp"
#include "../integration/ChebyshevGauss.hpp"
#include "../integration/Lebedev.hpp"
#include "../integration/TreutlerAhlrichsLebedev.hpp"

#include "../setup.hpp"
#include "../caliperInterface.hpp"
#include "../RAJAInterface.hpp"

#include "../HydrogenLikeBasis.hpp"

#include "testingCommon.hpp"

#include <chrono>

namespace tcscf::integration::testing
{

TEST( ChebyshevGauss, linear )
{
  ChebyshevGauss< double > integrator( 100 );
  double const value = integrate( integrator,
    [] ( CArray< double, 1 > const & x )
    {
      return x[ 0 ];
    }
  );

  EXPECT_NEAR( value, 0, 3e-16 );
}


TEST( ChebyshevGauss, quadratic )
{
  ChebyshevGauss< double > integrator( 1000 );
  double const value = integrate( integrator, 
    [] ( CArray< double, 1 > const & x )
    {
      return x[ 0 ] * x[ 0 ];
    }
  );

  EXPECT_NEAR( value, 2.0 / 3.0, 1e-5 );
}


TEST( ChebyshevGauss, cubic )
{
  ChebyshevGauss< double > integrator( 1000 );
  double const value = integrate( integrator,
    [] ( CArray< double, 1 > const & x )
    {
      return std::pow( x[ 0 ], 3 ) + 3 * std::pow( x[ 0 ], 2 ) - 4 * x[ 0 ];
    }
  );

  EXPECT_NEAR( value, 2.0, 1e-5 );
}


TEST( ChebyshevGauss, linearGrid )
{
  Array2d< double > const grid = createGrid( ChebyshevGauss< double >( 100 ) );

  double const value = integrate< 1 >( grid.toViewConst(),
    [] ( CArray< double, 1 > const & x )
    {
      return x[ 0 ];
    }
  );

  EXPECT_NEAR( value, 0, 3e-16 );
}


TEST( TreutlerAhlrichs, exponential )
{
  ChebyshevGauss< double > integrator( 100 );
  changeOfVariables::TreutlerAhlrichs< double > changeOfVariables( 1.0 );

  double const value = integrate( integrator, changeOfVariables, 
    [] ( CArray< double, 1 > const & r )
    {
      return std::exp( -r[ 0 ] );
    }
  );

  EXPECT_NEAR( value, 1.0, 1e-7 );
}


TEST( TreutlerAhlrichs, exponentialPolynomial )
{
  ChebyshevGauss< double > integrator( 100 );
  changeOfVariables::TreutlerAhlrichs< double > changeOfVariables( 1.0 );

  double const value = integrate( integrator, changeOfVariables, 
    [] ( CArray< double, 1 > const & r )
    {
      return (std::pow( r[ 0 ], 2 ) - 3 * r[ 0 ]) * std::exp( -3 * r[ 0 ] / 2 );
    }
  );

  EXPECT_NEAR( value, -20.0 / 27.0, 1e-10 );
}

TEST( TreutlerAhlrichs, exponentialGrid )
{
  ChebyshevGauss< double > integrator( 100 );
  changeOfVariables::TreutlerAhlrichs< double > changeOfVariables( 1.0 );

  Array2d< double > const grid = createGrid( integrator, changeOfVariables );

  double const value = integrate< 1 >( grid.toViewConst(), 
    [] ( CArray< double, 1 > const & r )
    {
      return std::exp( -r[ 0 ] );
    }
  );

  EXPECT_NEAR( value, 1.0, 1e-7 );
}

TEST( TreutlerAhlrichs, exponentialPolynomialGrid )
{
  ChebyshevGauss< double > integrator( 100 );
  changeOfVariables::TreutlerAhlrichs< double > changeOfVariables( 1.0 );

  Array2d< double > const grid = createGrid( integrator, changeOfVariables );

  double const value = integrate< 1 >( grid.toViewConst(), 
    [] ( CArray< double, 1 > const & r )
    {
      return (std::pow( r[ 0 ], 2 ) - 3 * r[ 0 ]) * std::exp( -3 * r[ 0 ] / 2 );
    }
  );

  EXPECT_NEAR( value, -20.0 / 27.0, 1e-10 );
}


TEST( Lebedev, constant )
{
  Lebedev< double > integrator( 3 );

  double const value = integrate( integrator,
    [] ( CArray< double, 2 > const & )
    {
      return 1;
    }
  );

  EXPECT_NEAR( value, 4 * pi< double >, 3e-14 );
}


TEST( Lebedev, SphericalHarmonics )
{
  int const numSamples = 10;
  int const numRandomSamples = 50;

  std::mt19937_64 gen;

  for( int order : { 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25,
    27, 29, 31, 35, 41, 47, 53, 59, 65, 71, 77,
    83, 89, 95, 101, 107, 113, 119, 125, 131 } )
  {
    Lebedev< double > integrator( order );

    for( int i = 0; i < numSamples + numRandomSamples; ++i )
    {
      int const l1 = std::uniform_int_distribution< int >( 0, order )( gen );
      int const l2 = std::uniform_int_distribution< int >( 0, order - l1 )( gen );

      int m1 = l1;
      int m2 = l2;
      
      if( i >= numSamples )
      {
        m1 = std::uniform_int_distribution< int >( -l1, l1 )( gen );
        m2 = std::uniform_int_distribution< int >( -l2, l2 )( gen );
      }

      std::complex< double > const value = integrate( integrator,
        [l1, l2, m1, m2] ( CArray< double, 2 > const & thetaAndPhi )
        {
          double const theta = thetaAndPhi[ 0 ];
          double const phi = thetaAndPhi[ 1 ];
          return conj( sphericalHarmonic( l1, m1, theta, phi ) ) * sphericalHarmonic( l2, m2, theta, phi );
        }
      );

      bool const delta = (l1 == l2) && (m1 == m2);
      EXPECT_COMPLEX_NEAR( value, delta, (l1 + l2 + order) * 3e-15 );
    }
  }
}

std::complex< double > integrate(
  TreutlerAhlrichsLebedev< double > const & integrator,
  HydrogenLikeBasisFunction< double > const & b1,
  HydrogenLikeBasisFunction< double > const & b2 )
{
  ArrayView2d< double const > const & radialGrid = integrator.m_radialGrid;
  ArrayView2d< double const > const & angularGrid = integrator.m_angularGrid;

  std::complex< double > answer = 0;
  for( IndexType i = 0; i < angularGrid.size( 1 ); ++i )
  {
    double const theta = angularGrid( 0, i );
    double const phi = angularGrid( 1, i );

    double tmp = 0;
    for( IndexType j = 0; j < radialGrid.size( 1 ); ++j )
    {
      double const r = radialGrid( 0, j );
      double const weight = radialGrid( 1, j );
      tmp = tmp + weight * b1.radialComponent( r ) * b2.radialComponent( r ) * std::pow( r, 2 );
    }

    answer = answer + tmp * conj( sphericalHarmonic( b1.l, b1.m, theta, phi ) ) * sphericalHarmonic( b2.l, b2.m, theta, phi ) * angularGrid( 2, i );
  }

  return answer;
}


TEST( TreutlerAhlrichsLebedev, orthogonal )
{
  HydrogenLikeBasisFunction< double > b1 { 1, 2, 1, 0 };
  HydrogenLikeBasisFunction< double > b2 { 1, 2, 1, 0 };

  int order = 35;
  TreutlerAhlrichsLebedev< double > integrator( 1.0, 1000, order );

  {
    std::complex< double > value = integrate( integrator,
      [b1, b2] ( CArray< double, 3 > const & rThetaPhi )
      {
        double const r = rThetaPhi[ 0 ];
        double const theta = rThetaPhi[ 1 ];
        double const phi = rThetaPhi[ 2 ];
        return std::conj( b1( r, theta, phi ) ) * b2( r, theta, phi ) * std::pow( r, 2 );
      }
    );

    LVARRAY_LOG_VAR( value );
  }

  std::complex< double > const value = integrate( integrator, b1, b2 );
  LVARRAY_LOG_VAR( value );
}


TEST( TreutlerAhlrichsLebedev, orthongonal2 )
{
  HydrogenLikeBasisFunction< double > b1 { 1, 2, 1, 0 };
  HydrogenLikeBasisFunction< double > b2 { 1, 2, 1, 1 };
  HydrogenLikeBasisFunction< double > b3 { 1, 2, 1, 0 };
  HydrogenLikeBasisFunction< double > b4 { 1, 2, 1, 1 };

  auto identity = [] ( CArray< double, 6 > const & )
  {
    return 1;
  };

  int orders[] = { 15 };
  int radialSizes[] = { 50 };

  for( int order : orders )
  {
    Array1d< std::complex< double > > results;
    for( int nRadial : radialSizes )
    {
      TreutlerAhlrichsLebedev< double > integrator( 0.8, nRadial, order );
      std::complex< double > const value = integrateR1R2( integrator, b1, b2, b3, b4, identity );
      results.emplace_back( value );
    }

    LVARRAY_LOG( "Order = " << order << ": " << results );
  }
}



void testDifferentMethods(
  HydrogenLikeBasisFunction< double > const & b1,
  HydrogenLikeBasisFunction< double > const & b2,
  HydrogenLikeBasisFunction< double > const & b3,
  HydrogenLikeBasisFunction< double > const & b4,
  std::complex< double > const expectedValue,
  double const error,
  std::vector< CArray< int, 2 > > standardParams,
  std::vector< CArray< int, 2 > > dualParams )
{
  printf( "Expected value is %.3e + %.3e I \\pm %.3e\n", expectedValue.real(), expectedValue.imag(), error );

  auto r12Inv = [] ( CArray< double, 6 > const & R1R2 )
  {
    double const r12 = calculateR12( R1R2[ 0 ], R1R2[ 1 ], R1R2[ 2 ], R1R2[ 3 ], R1R2[ 4 ], R1R2[ 5 ] );
    if( r12 <= 0 )
    {
      return 0.0;
    }
    else
    {
      return 1.0 / r12;
    }
  };
  
  char const * const formatString = "%20s, order = %2d, nRadial = %4d: %.3e + %.3e I, diff = %.3e, inside error bars = %5s, relative error = %.2e, time = %6.2f s\n";

  for( CArray< int, 2 > const & params : standardParams )
  {
    int order = params[ 0 ];
    int nRadial = params[ 1 ];

    auto const t0 = std::chrono::steady_clock::now();
    TreutlerAhlrichsLebedev< double > integrator( 0.8, nRadial, order );
    std::complex< double > const value = integrateR1R2( integrator, b1, b2, b3, b4, r12Inv );
    std::chrono::duration< double > const tElapsed = std::chrono::steady_clock::now() - t0;

    double const diff = std::abs( value - expectedValue );
    double const rdiff = diff / std::abs( expectedValue );
    char const * const inBounds = diff < error ? "true" : "false";
    printf( formatString, "r1r2", order, nRadial, value.real(), value.imag(), diff, inBounds, rdiff, tElapsed.count() );
  }

  for( CArray< int, 2 > const & params : dualParams )
  {
    int order = params[ 0 ];
    int nRadial = params[ 1 ];

    auto const t0 = std::chrono::steady_clock::now();
    TreutlerAhlrichsLebedev< double > integrator( 0.8, nRadial, order );
    std::complex< double > const value = integrateR1R12( integrator, b1, b2, b3, b4, [] ( double, double, double const r12 ) { return 1.0 / r12; } );
    std::chrono::duration< double > const tElapsed = std::chrono::steady_clock::now() - t0;

    double const diff = std::abs( value - expectedValue );
    double const rdiff = diff / std::abs( expectedValue );
    char const * const inBounds = diff < error ? "true" : "false";
    printf( formatString, "r1r12", order, nRadial, value.real(), value.imag(), diff, inBounds, rdiff, tElapsed.count() );
  }

  std::complex< double > value = 0;
  
  auto const t0 = std::chrono::steady_clock::now();
  if( b1.l == b3.l && b1.m == b3.m && b2.l == b4.l && b2.m == b4.m )
  {
    value = atomicCoulombOperator( b1, b2, b3, b4 );
  }
  else if( b1.l == b4.l && b1.m == b4.m && b2.l == b3.l && b2.m == b3.m )
  {
    value = atomicCoulombOperator( b1, b2, b3, b4 );
  }
  else
  {
    LVARRAY_ERROR( "This isn't a coulomb or exchange integral." );
  }

  std::chrono::duration< double > const tElapsed = std::chrono::steady_clock::now() - t0;
  double const diff = std::abs( value - expectedValue );
    double const rdiff = diff / std::abs( expectedValue );
  char const * const inBounds = diff < error ? "true" : "false";
  printf( formatString, "QMC", 0, 0, value.real(), value.imag(), diff, inBounds, rdiff, tElapsed.count() );
}


TEST( r12, first )
{
  std::vector< CArray< int, 2 > > standardParams = {
    { 9, 50 },
    { 9, 100 },
    { 11, 50 },
    { 11, 100 },
    { 13, 50 },
    { 13, 100 },
    { 15, 50 },
    { 15, 100 },
    { 17, 50 },
    { 17, 100 },
  };

  std::vector< CArray< int, 2 > > dualParams = {
    { 9, 50 },
    { 9, 100 },
    { 11, 50 },
    { 11, 100 },
    { 13, 50 },
    { 13, 100 },
    { 15, 50 },
    { 15, 100 },
    { 17, 50 },
    { 17, 100 },
  };

  HydrogenLikeBasisFunction< double > b1 { 1, 2, 1, 0 };
  testDifferentMethods( b1, b1, b1, b1, 0.196207, 0.0018, standardParams, dualParams );
}


TEST( r12, second )
{
  std::vector< CArray< int, 2 > > standardParams = {
    // { 9, 50 },
    // { 9, 100 },
    // { 11, 50 },
    // { 11, 100 },
    // { 13, 50 },
    // { 13, 100 },
    // { 15, 50 },
    // { 15, 100 },
    // { 17, 50 },
    // { 17, 100 },
  };

  std::vector< CArray< int, 2 > > dualParams = {
    { 17, 10 },
    { 17, 25 },
    { 17, 50 },
    { 17, 100 },
    { 19, 10 },
    { 19, 25 },
    { 19, 50 },
    { 19, 100 },
    { 21, 10 },
    { 21, 25 },
    { 21, 50 },
    { 21, 100 },
    { 23, 10 },
    { 23, 25 },
    { 23, 50 },
    { 23, 100 },
    { 25, 10 },
    { 25, 25 },
    { 25, 50 },
    { 25, 100 },
    // { 15, 50 },
    // { 15, 100 },
    // { 17, 50 },
    // { 17, 100 },
  };

  HydrogenLikeBasisFunction< double > b1 { 1, 2, 1, 0 };
  HydrogenLikeBasisFunction< double > b2 { 1, 2, 1, 1 };
  testDifferentMethods( b1, b2, b1, b2, 0.174825, 0.000862, standardParams, dualParams );
}


TEST( r12, third )
{
  std::vector< CArray< int, 2 > > standardParams = {
    { 9, 50 },
    { 9, 100 },
    { 11, 50 },
    { 11, 100 },
    { 13, 50 },
    { 13, 100 },
    { 15, 50 },
    { 15, 100 },
    { 17, 50 },
    { 17, 100 },
  };

  std::vector< CArray< int, 2 > > dualParams = {
    { 9, 50 },
    { 9, 100 },
    { 11, 50 },
    { 11, 100 },
    { 13, 50 },
    { 13, 100 },
    { 15, 50 },
    { 15, 100 },
    { 17, 50 },
    { 17, 100 },
  };

  HydrogenLikeBasisFunction< double > b1 { 1, 4, 3, 2 };
  HydrogenLikeBasisFunction< double > b2 { 1, 3, 2, -1 };
  HydrogenLikeBasisFunction< double > b3 { 1, 5, 3, 2 };
  HydrogenLikeBasisFunction< double > b4 { 1, 6, 2, -1 };
  testDifferentMethods( b1, b2, b1, b2, 0.000451565, 0.000064, standardParams, dualParams );
}

TEST( r12, fourth )
{
  std::vector< CArray< int, 2 > > standardParams = {
    { 7, 50 },
    { 7, 100 },
    { 9, 50 },
    { 9, 100 },
    { 11, 50 },
    { 11, 100 },
    { 13, 50 },
    { 13, 100 },
    { 15, 50 },
    { 15, 100 },
    { 17, 50 },
    { 17, 100 },
  };

  std::vector< CArray< int, 2 > > dualParams = {
    { 7, 50 },
    { 7, 100 },
    { 9, 50 },
    { 9, 100 },
    { 11, 50 },
    { 11, 100 },
    { 13, 50 },
    { 13, 100 },
    { 15, 50 },
    { 15, 100 },
    { 17, 50 },
    { 17, 100 },
  };

  HydrogenLikeBasisFunction< double > b1 { 1, 4, 2, -2 };
  HydrogenLikeBasisFunction< double > b2 { 1, 5, 3, 1 };
  testDifferentMethods( b1, b2, b1, b2, 0.0317012, 0.0001, standardParams, dualParams );
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
