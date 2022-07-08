#include "../setup.hpp"
#include "../caliperInterface.hpp"
#include "../mathFunctions.hpp"
#include "../LvArrayInterface.hpp"

#include "testingCommon.hpp"

namespace tcscf::testing
{

// Integrate from -1 to 1 using Chebyhev quadrature
template< typename F >
double chebyshevGaussQuadrature( int const n, F && f )
{
  double const freq = pi< double > / (n + 1);

  double answer = 0;
  for( int i = 1; i <= n; ++i )
  {
    double const theta = i * freq;
    double sinTheta, cosTheta;
    LvArray::math::sincos( theta, sinTheta, cosTheta );

    double const weight = std::pow( sinTheta, 2 );
    answer += weight * f( cosTheta ) / sinTheta;
  }

  return freq * answer;
}

// Integrate from 0 to inf using Chebyhev quadrature and Treutler-Ahlrichs mapping
template< typename F >
double treutler( int const n, double const epsilon, F && f )
{
  constexpr double alpha = 0.6;
  double const epsilonOverLog2 = epsilon / ln2< double >;

  return chebyshevGaussQuadrature( n, 
    [=] ( double const x )
    {
      double const logPart = std::log( 2 / (1 - x) );
      double const xPlus1ToTheAlphaMinus1 = std::pow( x + 1, alpha - 1 );
      double const xPlus1ToTheAlpha = xPlus1ToTheAlphaMinus1 * (x + 1);

      double const r = epsilonOverLog2 * xPlus1ToTheAlpha * logPart;
      double const dr = epsilonOverLog2 * ( alpha * xPlus1ToTheAlphaMinus1 * logPart - xPlus1ToTheAlpha / (x - 1) );
      return f( r ) * dr;
    }
  );
}

template< typename T >
Array2d< T > getTreutlerAhlrichsRadialGrid( int const n, double const epsilon )
{
  Array2d< T > grid( 2, n );

  constexpr double alpha = 0.6;
  double const constant = pi< double > / (n + 1);

  for( int i = 1; i <= n; ++i )
  {
    double const weight = std::pow( std::sin( i * constant ), 2 );
    double const x = std::cos( i * constant );
    
    double const r = epsilon / ln2< double > * std::pow( 1 + x, alpha ) * std::log( 2 / (1 - x) );
    double const dr = epsilon / ln2< double > * ( alpha * std::pow( x + 1, alpha - 1 ) * std::log( 2 / (1 - x) ) - std::pow( x + 1, alpha ) / (x - 1) );
    
    grid( 0, i - 1 ) = r;
    grid( 1, i - 1 ) = weight * dr / std::sqrt( 1 - x * x );
  }

  return grid;
}

template< typename T, typename F >
T evaluateQuadrature( ArrayView2d< T const > const & grid, F && f )
{
  LVARRAY_ERROR_IF_NE( grid.size( 0 ), 2 );

  double answer = 0;
  for( IndexType i = 0; i < grid.size( 1 ); ++i )
  {
    double const r = grid( 0, i );
    double const weight = grid( 1, i );
    answer += f( r ) * weight;
  }

  return pi< double > / (grid.size( 1 ) + 1) * answer;
}


TEST( foo, chebyshevGaussQuadrature )
{
  double const a1 = chebyshevGaussQuadrature( 100,
    [] ( double const x )
    {
      return x;
    }
  );

  EXPECT_NEAR( a1, 0, 3e-16 );

  double const a2 = chebyshevGaussQuadrature( 1000,
    [] ( double const x )
    {
      return x * x;
    }
  );

  EXPECT_NEAR( a2, 2.0 / 3.0, 1e-5 );

  double const a3 = chebyshevGaussQuadrature( 1000,
    [] ( double const x )
    {
      return std::pow( x, 3 ) + 3 * std::pow( x, 2 ) - 4 * x;
    }
  );

  EXPECT_NEAR( a3, 2.0, 1e-5 );
}

TEST( foo, treutler )
{
  double const a1 = treutler( 100, 1.0,
    [] ( double const r )
    {
      return std::exp( -r );
    }
  );

  EXPECT_NEAR( a1, 1.0, 1e-7 );

  Array2d< double > const grid = getTreutlerAhlrichsRadialGrid< double >( 100, 1.0 );

  double const a2 = evaluateQuadrature( grid.toViewConst(),
    [] ( double const r )
    {
      return std::exp( -r );
    }
  );

  EXPECT_NEAR( a2, 1.0, 1e-7 );
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
