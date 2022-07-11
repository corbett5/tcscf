#include "../integration/quadrature.hpp"
#include "../integration/ChebyshevGauss.hpp"
#include "../integration/Lebedev.hpp"
#include "../integration/TreutlerAhlrichsLebedev.hpp"

#include "../setup.hpp"
#include "../caliperInterface.hpp"

#include "../HydrogenLikeBasis.hpp"

#include "testingCommon.hpp"


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

template< typename F >
std::complex< double > integrate(
  TreutlerAhlrichsLebedev< double > const & integrator,
  HydrogenLikeBasisFunction< double > const & b1,
  HydrogenLikeBasisFunction< double > const & b2,
  F && f,
  HydrogenLikeBasisFunction< double > const & b3,
  HydrogenLikeBasisFunction< double > const & b4 )
{
  ArrayView2d< double const > const & radialGrid = integrator.m_radialGrid;
  ArrayView2d< double const > const & radialGrid2 = integrator.m_radialGrid2;
  ArrayView2d< double const > const & angularGrid = integrator.m_angularGrid;

  std::complex< double > answer = 0;
  for( IndexType i = 0; i < angularGrid.size( 1 ); ++i )
  {
    double const theta1 = angularGrid( 0, i );
    double const phi1 = angularGrid( 1, i );

    std::complex< double > tmp = 0;
    for( IndexType j = 0; j < radialGrid.size( 1 ); ++j )
    {
      double const r1 = radialGrid( 0, j );

      std::complex< double > innerIntegral = 0;
      for( IndexType k = 0; k < angularGrid.size( 1 ); ++k )
      {
        double const theta2 = angularGrid( 0, k );
        double const phi2 = angularGrid( 1, k );

        double r2Tmp = 0;
        for( IndexType l = 0; l < radialGrid2.size( 1 ); ++l )
        {
          double const r2 = radialGrid2( 0, l );
          CArray< double, 6 > const R1R2 { r1, theta1, phi1, r2, theta2, phi2 };
          r2Tmp = r2Tmp + radialGrid2( 1, l ) * f( R1R2 ) * b3.radialComponent( r2 ) * b4.radialComponent( r2 ) * std::pow( r2, 2 );
        }

        innerIntegral = innerIntegral + angularGrid( 2, k ) * r2Tmp * sphericalHarmonic( b3.l, b3.m, theta2, phi2 ) * sphericalHarmonic( b4.l, b4.m, theta2, phi2 );
      }

      tmp = tmp + radialGrid( 1, j ) * innerIntegral * b1.radialComponent( r1 ) * b2.radialComponent( r1 ) * std::pow( r1, 2 );
    }

    answer = answer + angularGrid( 2, i ) * tmp * conj( sphericalHarmonic( b1.l, b1.m, theta1, phi1 ) * sphericalHarmonic( b2.l, b2.m, theta1, phi1 ) );
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

TEST( r12, TreutlerAhlrichsLebedev )
{
  HydrogenLikeBasisFunction< double > b1 { 1, 2, 1, 0 };
  HydrogenLikeBasisFunction< double > b2 { 1, 2, 1, 0 };


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

  // int orders[] = {7, 9, 11, 13, 15, 17, 19};
  // int radialSizes[] = {10, 20, 40, 80, 160, 320};

  int orders[] = {19};
  int radialSizes[] = {100, 200};

  for( int order : orders )
  {
    Array1d< std::complex< double > > results;
    for( int nRadial : radialSizes )
    {
      TreutlerAhlrichsLebedev< double > integrator( 1.0, nRadial, order );
      std::complex< double > const value = integrate( integrator, b1, b2, r12Inv, b1, b2 );
      results.emplace_back( value );
    }

    LVARRAY_LOG( "Order = " << order << ": " << results );
  }

  for( int order : orders )
  {
    Array1d< std::complex< double > > results;
    for( int nRadial : radialSizes )
    {
      TreutlerAhlrichsLebedev< double > integrator( 1.0, nRadial, nRadial + 5, order );
      std::complex< double > const value = integrate( integrator, b1, b2, r12Inv, b1, b2 );
      results.emplace_back( value );
    }

    LVARRAY_LOG( "Order = " << order << ": " << results );
  }
}

TEST( r12, qmc )
{
  HydrogenLikeBasisFunction< double > b1 { 1, 2, 1, 0 };
  HydrogenLikeBasisFunction< double > b2 { 1, 2, 1, 0 };

  double const value = atomicCoulombOperator( b1, b2, b1, b2 );
  LVARRAY_LOG_VAR( value );
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
