#include "../quadrature.hpp"
#include "../setup.hpp"
#include "../caliperInterface.hpp"

#include "../HydrogenLikeBasis.hpp"

#include "testingCommon.hpp"

namespace tcscf::testing
{

TEST( ChebyshevGauss, linear )
{
  quadrature::ChebyshevGauss< double > integrator( 100 );
  double const value = integrator.integrate(
    [] ( double const x )
    {
      return x;
    }
  );

  EXPECT_NEAR( value, 0, 3e-16 );
}

TEST( ChebyshevGauss, quadratic )
{
  quadrature::ChebyshevGauss< double > integrator( 1000 );
  double const value = integrator.integrate(
    [] ( double const x )
    {
      return x * x;
    }
  );

  EXPECT_NEAR( value, 2.0 / 3.0, 1e-5 );
}

TEST( ChebyshevGauss, cubic )
{
  quadrature::ChebyshevGauss< double > integrator( 1000 );
  double const value = integrator.integrate(
    [] ( double const x )
    {
      return std::pow( x, 3 ) + 3 * std::pow( x, 2 ) - 4 * x;
    }
  );

  EXPECT_NEAR( value, 2.0, 1e-5 );
}

TEST( TreutlerAhlrichs, exponential )
{
  quadrature::ChebyshevGauss< double > integrator( 100 );
  changeOfVariables::TreutlerAhlrichs< double > changeOfVariables( 1.0 );

  double const value = quadrature::integrate( integrator, changeOfVariables, 
    [] ( double const r )
    {
      return std::exp( -r );
    }
  );

  EXPECT_NEAR( value, 1.0, 1e-7 );
}

TEST( TreutlerAhlrichs, exponentialPolynomial )
{
  quadrature::ChebyshevGauss< double > integrator( 100 );
  changeOfVariables::TreutlerAhlrichs< double > changeOfVariables( 1.0 );

  double const value = quadrature::integrate( integrator, changeOfVariables, 
    [] ( double const r )
    {
      return (std::pow( r, 2 ) - 3 * r) * std::exp( -3 * r / 2 );
    }
  );

  EXPECT_NEAR( value, -20.0 / 27.0, 1e-10 );
}

TEST( TreutlerAhlrichs, exponentialGrid )
{
  quadrature::ChebyshevGauss< double > integrator( 100 );
  changeOfVariables::TreutlerAhlrichs< double > changeOfVariables( 1.0 );

  Array2d< double > const grid = quadrature::createGrid( integrator, changeOfVariables );

  double const value = quadrature::integrate( grid.toViewConst(), 
    [] ( double const r )
    {
      return std::exp( -r );
    }
  );

  EXPECT_NEAR( value, 1.0, 1e-7 );
}

TEST( TreutlerAhlrichs, exponentialPolynomialGrid )
{
  quadrature::ChebyshevGauss< double > integrator( 100 );
  changeOfVariables::TreutlerAhlrichs< double > changeOfVariables( 1.0 );

  Array2d< double > const grid = quadrature::createGrid( integrator, changeOfVariables );

  double const value = quadrature::integrate( grid.toViewConst(), 
    [] ( double const r )
    {
      return (std::pow( r, 2 ) - 3 * r) * std::exp( -3 * r / 2 );
    }
  );

  EXPECT_NEAR( value, -20.0 / 27.0, 1e-10 );
}


TEST( Lebedev, constant )
{
  quadrature::Lebedev< double > integrator( 6 );

  double const value = integrator.integrate(
    [] ( double const, double const )
    {
      return 1;
    }
  );

  EXPECT_NEAR( value, 4 * pi< double >, 3e-14 );
}

TEST( TreutlerAhlrichsLebedev, orthogonal )
{
  quadrature::Lebedev< double > integrator( 6 );

  quadrature::ChebyshevGauss< double > radialIntegrator( 100 );
  changeOfVariables::TreutlerAhlrichs< double > changeOfVariables( 1.0 );

  Array2d< double > const grid = quadrature::createGrid( radialIntegrator, changeOfVariables );

  HydrogenLikeBasisFunction< double > b1 { 1, 5, 4, 3 };
  HydrogenLikeBasisFunction< double > b2 { 1, 5, 4, 3 };

  quadrature::Lebedev< double > angleIntegrator( 6 );

  std::complex< double > const value = angleIntegrator.integrate(
    [&] ( double const theta, double const phi )
    {
      auto const integrand = [&] ( double const r )
      {
        return std::conj( b1( r, theta, phi ) ) * b2( r, theta, phi ) * std::pow( r, 2 );
      };

      return quadrature::integrate( grid.toViewConst(), integrand );
    }
  );

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
