#pragma once

#include "mathFunctions.hpp"

#include <qmc.hpp>

namespace tcscf::qmc
{

namespace internal
{

/**
 * 
 */
template< typename ARG0, typename ... ARGS >
constexpr bool allSameTypes()
{ return ( std::is_same_v<ARG0, ARGS> && ... ); }

/**
 * 
 */
template< typename >
struct FunctionInfo
{};

/**
 * 
 */
template< typename RET, typename ARG0, typename ... ARGS >
struct FunctionInfo< RET ( ARG0, ARGS ...) >
{
  static constexpr int NARGS = 1 + sizeof ... ( ARGS );
  using RETURN_TYPE = RET;
  using ARG_TYPE = ARG0;
  static_assert( allSameTypes< ARG0, ARGS... >() );
};

/**
 * 
 */
template< typename CLASS, typename RET, typename ARG0, typename ... ARGS >
struct FunctionInfo< RET ( CLASS:: * ) ( ARG0, ARGS ... ) const >
{
  static constexpr int NARGS = 1 + sizeof ... (ARGS);
  using RETURN_TYPE = RET;
  using ARG_TYPE = ARG0;
  static_assert( allSameTypes< ARG0, ARGS... >() );
};

/**
 * 
 */
template< typename F >
struct Functor
{
  using RETURN_TYPE = typename FunctionInfo< decltype( &F::operator() )>::RETURN_TYPE;
  using ARG_TYPE = typename FunctionInfo< decltype( &F::operator() )>::ARG_TYPE;
  static constexpr int NARGS = FunctionInfo< decltype( &F::operator() )>::NARGS;
  static constexpr unsigned long long int number_of_integration_variables = NARGS;

  template< int _NARGS = NARGS >
  constexpr std::enable_if_t< _NARGS == 3, RETURN_TYPE >
  operator()( ARG_TYPE const * const x ) const
  { return f( x[ 0 ], x[ 1 ], x[ 2 ] ); }

  template< int _NARGS = NARGS >
  constexpr std::enable_if_t< _NARGS == 5, RETURN_TYPE >
  operator()( ARG_TYPE const * const x ) const
  { return f( x[ 0 ], x[ 1 ], x[ 2 ], x[ 3 ], x[ 4 ] ); }

  template< int _NARGS = NARGS >
  constexpr std::enable_if_t< _NARGS == 6, RETURN_TYPE >
  operator()( ARG_TYPE const * const x ) const
  { return f( x[ 0 ], x[ 1 ], x[ 2 ], x[ 3 ], x[ 4 ], x[ 5 ] ); }

  F f;
};

/**
 * 
 */
template< typename F >
auto createFunctor( F && f )
{ return Functor< F >{ std::forward< F >( f ) }; }

} // namespace internal

/**
 * 
 */
template< typename F >
auto sphericalCoordinates3DIntegral( F && f )
{
  using FMetaData = internal::FunctionInfo< decltype( &F::operator() ) >;
  using RETURN_TYPE = typename FMetaData::RETURN_TYPE;
  using COORD_TYPE = typename FMetaData::ARG_TYPE;

  static_assert( FMetaData::NARGS == 3 );

  auto integrand = [f = std::forward< F >( f )] ( COORD_TYPE const xR, COORD_TYPE const xTheta, COORD_TYPE const xPhi )
  {
    COORD_TYPE const r = xR / (1 - xR);
    COORD_TYPE const theta = pi< COORD_TYPE > * xTheta;
    COORD_TYPE const phi = 2 * pi< COORD_TYPE > * xPhi;
    return f( r, theta, phi ) * std::pow( r, 2 ) * std::sin( theta ) / std::pow( 1 - xR, 2 );
  };

  auto const functor = internal::createFunctor( std::move( integrand ) );

  integrators::Qmc< RETURN_TYPE, COORD_TYPE, 3, integrators::transforms::None::type > integrator;
  auto const result = integrator.integrate( functor );
  return 2 * std::pow( pi< COORD_TYPE >, 2 ) * result.integral;
}

/**
 * phi2 is fixed to zero
 */
template< typename F >
auto sphericalCoordinates5DIntegral( F && f )
{
  using FMetaData = internal::FunctionInfo< decltype( &F::operator() )>;
  using RETURN_TYPE = typename FMetaData::RETURN_TYPE;
  using COORD_TYPE = typename FMetaData::ARG_TYPE;

  static_assert( FMetaData::NARGS == 5 );

  auto const functor = internal::createFunctor( [f = std::forward< F >( f )]
    ( COORD_TYPE const xR1, COORD_TYPE const xTheta1, COORD_TYPE const xPhi1,
      COORD_TYPE const xR2, COORD_TYPE const xTheta2 )
    {
      COORD_TYPE const r1 = xR1 / (1 - xR1);
      COORD_TYPE const theta1 = pi< COORD_TYPE > * xTheta1;
      COORD_TYPE const phi1 = 2 * pi< COORD_TYPE > * xPhi1;

      COORD_TYPE const r2 = xR2 / (1 - xR2);
      COORD_TYPE const theta2 = pi< COORD_TYPE > * xTheta2;

      return f( r1, theta1, phi1, r2, theta2 ) * std::pow( r1, 2 ) * std::sin( theta1 ) / std::pow( 1 - xR1, 2 ) * 
             std::pow( r2, 2 ) * std::sin( theta2 ) / std::pow( 1 - xR2, 2 );
    }
  );

  integrators::Qmc< RETURN_TYPE, COORD_TYPE, 5, integrators::transforms::None::type > integrator;
  // integrator.epsrel = 1e-6;
  // integrator.epsabs = 1e-10;
  // integrator.maxeval = 1e7;
  auto const result = integrator.integrate( functor );

  // LVARRAY_LOG( result.evaluations << ", " << std::abs( result.error / result.integral ) );

  return 4 * std::pow( pi< COORD_TYPE >, 4 ) * result.integral;
}

/**
 * 
 */
template< typename F >
auto sphericalCoordinates6DIntegral( F && f )
{
  using FMetaData = internal::FunctionInfo< decltype( &F::operator() )>;
  using RETURN_TYPE = typename FMetaData::RETURN_TYPE;
  using COORD_TYPE = typename FMetaData::ARG_TYPE;

  static_assert( FMetaData::NARGS == 6 );

  auto const functor = internal::createFunctor( [f = std::forward< F >( f )]
    ( COORD_TYPE const xR1, COORD_TYPE const xTheta1, COORD_TYPE const xPhi1,
      COORD_TYPE const xR2, COORD_TYPE const xTheta2, COORD_TYPE const xPhi2 )
    {
      COORD_TYPE const r1 = xR1 / (1 - xR1);
      COORD_TYPE const theta1 = pi< COORD_TYPE > * xTheta1;
      COORD_TYPE const phi1 = 2 * pi< COORD_TYPE > * xPhi1;

      COORD_TYPE const r2 = xR2 / (1 - xR2);
      COORD_TYPE const theta2 = pi< COORD_TYPE > * xTheta2;
      COORD_TYPE const phi2 = 2 * pi< COORD_TYPE > * xPhi2;

      return f( r1, theta1, phi1, r2, theta2, phi2 ) * std::pow( r1, 2 ) * std::sin( theta1 ) / std::pow( 1 - xR1, 2 ) * 
             std::pow( r2, 2 ) * std::sin( theta2 ) / std::pow( 1 - xR2, 2 );
    }
  );

  integrators::Qmc< RETURN_TYPE, COORD_TYPE, 6, integrators::transforms::None::type > integrator;
  auto const result = integrator.integrate( functor );
  return 4 * std::pow( pi< COORD_TYPE >, 4 ) * result.integral;
}

} // namespace qmc::scf