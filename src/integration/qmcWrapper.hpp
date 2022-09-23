#pragma once

#include "quadrature.hpp"

#include "../RAJAInterface.hpp"

#include <qmc.hpp>

namespace tcscf::integration
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
template< typename FUNCTOR >
using FunctorInfo = FunctionInfo< decltype( &std::remove_reference_t< FUNCTOR >::operator() ) >;

/**
 * 
 */
template< typename F >
struct Functor
{
  using RETURN_TYPE = typename FunctorInfo< F >::RETURN_TYPE;
  using CARRAY_TYPE = std::remove_reference_t< typename FunctorInfo< F >::ARG_TYPE >;
  using ARG_TYPE = typename CARRAY_TYPE::value_type;
  static constexpr unsigned long long int number_of_integration_variables = CARRAY_TYPE {}.size();

  constexpr RETURN_TYPE operator()( ARG_TYPE const * const x ) const
  {
    CArray< ARG_TYPE, number_of_integration_variables > xArr;
    for( unsigned long long int i = 0; i < number_of_integration_variables; ++i )
    {
      xArr[ i ] = x[ i ];
    }

    return f( xArr );
  }

  F f;
};

/**
 * 
 */
template< typename F >
auto createFunctor( F && f )
{ return Functor< F >{ std::forward< F >( f ) }; }

} // namespace internal


template< int NDIM_P, typename REAL >
class QMCCache
{
public:
  static constexpr int NDIM = NDIM_P;
  using Real = REAL;

  QMCCache( IndexType const minGridSize ):
    _points( 1 )
  {
    integrators::Qmc< Real, Real, NDIM, integrators::transforms::None::type > integrator;
    integrator.maxeval = 1;
    integrator.minn = minGridSize;
    integrator.minm = 2;
    
    _points.reserveValues( 2 * integrator.get_next_n( minGridSize ) );

    auto functor = internal::createFunctor( [this] ( CArray< Real, NDIM > const & x )
    {
      _points.template emplaceBackAtomic< Atomic< ParallelHost > >( 0, x );
      return Real{ 1 };
    } );

    auto result = integrator.integrate( functor );
    LVARRAY_ERROR_IF_GT( result.evaluations, 2 * integrator.get_next_n( minGridSize ) );
  }

  constexpr IndexType numPoints() const
  {
    return _points.sizeOfArray( 0 );
  }

  constexpr Real gridWeight() const
  {
    return Real{ 1 } / _points.sizeOfArray( 0 );
  }

  constexpr GridPoint< Real, NDIM > gridPoint( IndexType const i ) const
  {
    return { _points( 0, i ), 1 };
  }

private:

  ArrayOfArrays< CArray< Real, NDIM > > _points;
};

/**
 * 
 */
template< typename INTEGRATOR, typename F >
auto sphericalCoordinates3DIntegral( INTEGRATOR & integrator, F && f )
{
  using COORD_TYPE = typename internal::FunctorInfo< F >::ARG_TYPE;

  static_assert( internal::FunctorInfo< F >::NARGS == 3 );

  auto integrand = [f = std::forward< F >( f )] ( CArray< COORD_TYPE, 3 > const & x )
  {
    COORD_TYPE const r = x[ 0 ] / (1 - x[ 0 ]);
    COORD_TYPE const theta = pi< COORD_TYPE > * x[ 1 ];
    COORD_TYPE const phi = 2 * pi< COORD_TYPE > * x[ 2 ];
    return f( r, theta, phi ) * std::pow( r, 2 ) * std::sin( theta ) / std::pow( 1 - x[ 0 ], 2 );
  };

  auto const functor = internal::createFunctor( std::move( integrand ) );

  auto const result = integrator.integrate( functor );
  return 2 * std::pow( pi< COORD_TYPE >, 2 ) * result.integral;
}

template< typename F >
auto sphericalCoordinates2DIntegral( F && f )
{
  using RETURN_TYPE = typename internal::FunctorInfo< F >::RETURN_TYPE;
  using COORD_TYPE = typename internal::FunctorInfo< F >::ARG_TYPE;

  static_assert( internal::FunctorInfo< F >::NARGS == 2 );

  auto const functor = internal::createFunctor( [f = std::forward< F >( f )]
    ( CArray< COORD_TYPE, 2 > const & x )
    {
      COORD_TYPE const r = x[ 0 ] / (1 - x[ 0 ]);
      COORD_TYPE const theta = pi< COORD_TYPE > * x[ 1 ];

      return f( r, theta ) * std::pow( r, 2 ) * std::sin( theta ) / std::pow( 1 - x[ 0 ], 2 );
    }
  );

  integrators::Qmc< RETURN_TYPE, COORD_TYPE, 2, integrators::transforms::None::type > integrator;
  integrator.maxeval = 1;
  integrator.minn = 10000;
  auto const result = integrator.integrate( functor );

  return 2 * std::pow( pi< COORD_TYPE >, 2 ) * result.integral;
}

/**
 * 
 */
template< typename F >
auto sphericalCoordinates3DIntegral( F && f )
{
  using RETURN_TYPE = typename internal::FunctorInfo< F >::RETURN_TYPE;
  using COORD_TYPE = typename internal::FunctorInfo< F >::ARG_TYPE;

  integrators::Qmc< RETURN_TYPE, COORD_TYPE, 3, integrators::transforms::None::type > integrator;
  return sphericalCoordinates3DIntegral( integrator, std::forward< F >( f ) );
}

/**
 * phi2 is fixed to zero
 */
template< typename F >
auto sphericalCoordinates5DIntegral( F && f )
{
  using RETURN_TYPE = typename internal::FunctorInfo< F >::RETURN_TYPE;
  using COORD_TYPE = typename internal::FunctorInfo< F >::ARG_TYPE;

  static_assert( internal::FunctorInfo< F >::NARGS == 5 );

  auto const functor = internal::createFunctor( [f = std::forward< F >( f )]
    ( CArray< COORD_TYPE, 5 > const & x )
    {
      COORD_TYPE const r1 = x[ 0 ] / (1 - x[ 0 ]);
      COORD_TYPE const theta1 = pi< COORD_TYPE > * x[ 1 ];
      COORD_TYPE const phi1 = 2 * pi< COORD_TYPE > * x[ 2 ];

      COORD_TYPE const r2 = x[ 3 ] / (1 - x[ 3 ]);
      COORD_TYPE const theta2 = pi< COORD_TYPE > * x[ 4 ];

      return f( r1, theta1, phi1, r2, theta2 ) * std::pow( r1, 2 ) * std::sin( theta1 ) / std::pow( 1 - x[ 0 ], 2 ) * 
              std::pow( r2, 2 ) * std::sin( theta2 ) / std::pow( 1 - x[ 3 ], 2 );
    }
  );

  integrators::Qmc< RETURN_TYPE, COORD_TYPE, 5, integrators::transforms::None::type > integrator;
  integrator.maxeval = 1;
  integrator.minn = 310577;
  auto const result = integrator.integrate( functor );

  return 4 * std::pow( pi< COORD_TYPE >, 4 ) * result.integral;
}

/**
 * 
 */
template< typename F >
auto sphericalCoordinates6DIntegral( F && f )
{
  using RETURN_TYPE = typename internal::FunctorInfo< F >::RETURN_TYPE;
  using COORD_TYPE = typename internal::FunctorInfo< F >::ARG_TYPE;

  static_assert( internal::FunctorInfo< F >::NARGS == 6 );

  auto const functor = internal::createFunctor( [f = std::forward< F >( f )]
    ( CArray< COORD_TYPE, 5 > const & x )
    {
      COORD_TYPE const r1 = x[ 0 ] / (1 - x[ 0 ]);
      COORD_TYPE const theta1 = pi< COORD_TYPE > * x[ 1 ];
      COORD_TYPE const phi1 = 2 * pi< COORD_TYPE > * x[ 2 ];

      COORD_TYPE const r2 = x[ 3 ] / (1 - x[ 3 ]);
      COORD_TYPE const theta2 = pi< COORD_TYPE > * x[ 4 ];
      COORD_TYPE const phi2 = 2 * pi< COORD_TYPE > * x[ 5 ];

      return f( r1, theta1, phi1, r2, theta2, phi2 ) * std::pow( r1, 2 ) * std::sin( theta1 ) / std::pow( 1 - x[ 0 ], 2 ) * 
             std::pow( r2, 2 ) * std::sin( theta2 ) / std::pow( 1 - x[ 3 ], 2 );
    }
  );

  integrators::Qmc< RETURN_TYPE, COORD_TYPE, 6, integrators::transforms::None::type > integrator;
  auto const result = integrator.integrate( functor );
  return 4 * std::pow( pi< COORD_TYPE >, 4 ) * result.integral;
}


} // namespace tcscf::integration