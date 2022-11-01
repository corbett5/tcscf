#include  "../jastrowFunctions.hpp"

#include "testingCommon.hpp"

#include <random>

namespace tcscf::testing
{

template< typename F >
Cartesian< decltype( std::declval< F >()( Cartesian< double > {} ) ) > finiteDifferenceGrad( Cartesian< double > const & r, F && f )
{
  double const dh = 1.0 / std::pow( 2, 15 );

  auto dfdx = (f( r + Cartesian< double >{ dh, 0, 0 } ) - f( r - Cartesian< double >{ dh, 0, 0 } )) / (2 * dh);
  auto dfdy = (f( r + Cartesian< double >{ 0, dh, 0 } ) - f( r - Cartesian< double >{ 0, dh, 0 } )) / (2 * dh);
  auto dfdz = (f( r + Cartesian< double >{ 0, 0, dh } ) - f( r - Cartesian< double >{ 0, 0, dh } )) / (2 * dh);

  return { dfdx, dfdy, dfdz };
}

template< typename F >
Cartesian< decltype( std::declval< F >()( Spherical< double > {} ) ) > finiteDifferenceGradSpherical(
  Cartesian< double > const & r,
  F && f )
{
  return finiteDifferenceGrad( r, [f] ( Cartesian< double > const & evalPoint )
  {
    return f( evalPoint.toSpherical() );
  } );
}

TEST( finiteDifferenceGrad, halfRSquared )
{
  auto f = [] ( Cartesian< double > const & r )
  {
    return 0.5 * std::pow( r.r(), 2 );
  };

  Cartesian< double > grad = finiteDifferenceGrad( Cartesian< double >{ 1, 2, 3 }, f );
  EXPECT_NEAR( grad.x(), 1, 1e-10 );
  EXPECT_NEAR( grad.y(), 2, 1e-10 );
  EXPECT_NEAR( grad.z(), 3, 1e-10 );
}

TEST( finiteDifferenceGrad, halfRSquaredSpherical )
{
  auto f = [] ( Spherical< double > const & r )
  {
    return 0.5 * std::pow( r.r(), 2 );
  };

  Cartesian< double > const grad = finiteDifferenceGradSpherical( Cartesian< double >{ 1, 2, 3 }, f );
  EXPECT_NEAR( grad.x(), 1, 1e-10 );
  EXPECT_NEAR( grad.y(), 2, 1e-10 );
  EXPECT_NEAR( grad.z(), 3, 1e-10 );
}

TEST( finiteDifferenceGrad, ochi )
{
  double const a = 1.5;
  double const a12 = a;
  Array2d< int > S( 1, 3 );
  S( 0, 0 ) = 1;

  Array2d< double > c( 1, 2 );
  c( 0, false ) = a12 / 2;
  c( 0, true ) = a12 / 4;

  jastrowFunctions::Ochi< double > const u { a, a12, c, S };

  std::mt19937 gen;
  std::uniform_real_distribution< double > dis(-5, 5);

  for( IndexType i = 0; i < 100; ++i )
  {
    Cartesian< double > const r2 { dis( gen ), dis( gen ), dis( gen ) };

    auto f = [&u, &r2] ( Cartesian< double > const & r1 )
    {
      return u( r1.r(), r2.r(), (r1 - r2).r(), false );
    };

    Cartesian< double > const r1 { 1, 1, 1 };
    
    Cartesian< double > const numericalGrad = finiteDifferenceGrad( r1, f );
    Cartesian< double > const analyticGrad = u.gradient( r1, r2, false );

    EXPECT_NEAR( analyticGrad.x(), numericalGrad.x(), 1e-8 );
    EXPECT_NEAR( analyticGrad.y(), numericalGrad.y(), 1e-8 );
    EXPECT_NEAR( analyticGrad.z(), numericalGrad.z(), 1e-8 );
  }
}

} // namespace tcscf::testing
