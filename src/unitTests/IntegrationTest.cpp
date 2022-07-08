#include "../integration/quadrature.hpp"
#include "../integration/ChebyshevGauss.hpp"
#include "../integration/Lebedev.hpp"

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
  for( int order : { 3, 5, 7 } )
  {
    Lebedev< double > integrator( order );

    for( int l1 = 0; l1 <= order; ++l1 )
    {
      for( int l2 = l1; l2 <= order - l1; ++l2 )
      {
        for( int m1 = -l1; m1 <= l1; ++m1 )
        {
          for( int m2 = -l2; m2 <= l2; ++m2 )
          {
            std::complex< double > const value = integrate( integrator,
              [l1, l2, m1, m2] ( CArray< double, 2 > const & thetaAndPhi )
              {
                double const theta = thetaAndPhi[ 0 ];
                double const phi = thetaAndPhi[ 1 ];
                return conj( sphericalHarmonic( l1, m1, theta, phi ) ) * sphericalHarmonic( l2, m2, theta, phi );
              }
            );

            bool const delta = (l1 == l2) && (m1 == m2);
            EXPECT_COMPLEX_NEAR( value, delta, (l1 + l2 + 2) * 1e-15 );
          }
        }
      }
    }
  }
}

#if 0

TEST( TreutlerAhlrichsLebedev, orthogonal )
{
  ChebyshevGauss< double > radialIntegrator( 10000 );
  changeOfVariables::TreutlerAhlrichs< double > changeOfVariables( 1.0 );

  Array2d< double > const grid = createGrid( radialIntegrator, changeOfVariables );

  HydrogenLikeBasisFunction< double > b1 { 1, 3, 1, 1 };
  HydrogenLikeBasisFunction< double > b2 { 1, 3, 2, 1 };

  Lebedev< double > angleIntegrator( 3 );

  std::complex< double > const value = integrate( angleIntegrator,
    [&] ( CArray< double, 2 > const & thetaAndPhi )
    {
      double const theta = thetaAndPhi[ 0 ];
      double const phi = thetaAndPhi[ 1 ];

      auto const integrand = [&] ( CArray< double, 1 > const & r )
      {
        return std::conj( b1( r[ 0 ], theta, phi ) ) * b2( r[ 0 ], theta, phi ) * std::pow( r[ 0 ], 2 );
      };

      return integrate< 1 >( grid.toViewConst(), integrand );
    }
  );

  LVARRAY_LOG_VAR( value );
}

#endif

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
