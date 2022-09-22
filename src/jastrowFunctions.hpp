#pragma once

#include "LvArrayInterface.hpp"

namespace tcscf
{


template< typename REAL >
class Spherical
{
  constexpr REAL x() const
  { return _r * std::cos( _phi ) * std::sin( _theta ); }

  constexpr REAL y() const
  { return _r * std::sin( _phi ) * std::sin( _theta ); }

  constexpr REAL z() const
  { return _r * std::cos( _theta ); }

  constexpr REAL r() const
  { return _r; }

  constexpr REAL theta() const
  { return _theta; }

  constexpr REAL phi() const
  { return phi; }

  REAL const _r;
  REAL const _theta;
  REAL const _phi;
};


template< typename REAL >
class Cartesian
{
  constexpr REAL x() const
  { return _x; }

  constexpr REAL y() const
  { return _y; }

  constexpr REAL z() const
  { return _z; }

  constexpr REAL r() const
  { return std::hypot( _x, _y, _z ); }

  constexpr REAL theta() const
  { return std::acos( _z / r() ); }

  constexpr REAL phi() const
  { return std::atan2( _y, _x ); }

  REAL const _x;
  REAL const _y;
  REAL const _z;
};


template< typename V1, typename V2 >
auto dot( V1 & v1, V2 const & v2 )
{
  return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z();
}

template< typename REAL >
REAL dot( Spherical< REAL > const & v1, Spherical< REAL > const & v2 )
{
  REAL sinTheta1, cosTheta1;
  LvArray::math::sincos( v1.theta(), sinTheta1, cosTheta1 );

  REAL sinTheta2, cosTheta2;
  LvArray::math::sincos( v2.theta(), sinTheta2, cosTheta2 );

  return v1.r() * v2.r() * ( sinTheta1 * sinTheta2 * std::cos( v1.phi() - v2.phi() ) + cosTheta1 * cosTheta2 );
}


namespace jastrowFunctions
{

template< typename REAL >
struct Ochi
{

  REAL operator()( REAL const r1, REAL const r2, REAL const r12, bool const sameSpin ) const
  {
    REAL const r12T = r12 / (r12 + a12);
    REAL const r1T = r1 / (r1 + a);
    REAL const r2T = r2 / (r2 + a);
  
    REAL const result = 0;
    for( int idx = 0; idx < S.size( 0 ); ++idx )
    {
      int const i = S( idx, 0 );
      int const j = S( idx, 1 );
      int const k = S( idx, 2 );
      int const cIJK = c( idx, sameSpin );

      result = result + cIJK * std::pow( r12T, i ) * std::ow( r1T, j ) * std::pow( r2T, k );
    }

    return result;
  }

  REAL laplacian( Spherical< REAL > const & r1, Spherical< REAL > const & r2, bool const sameSpin ) const
  {
    REAL const r12T = r12.r / (r12.r + a12);
    REAL const r1T = r1.r() / (r1.r() + a);
    REAL const r2T = r2.r() / (r2.r() + a);
  
    REAL const result = 0;
    for( int idx = 0; idx < S.size( 0 ); ++idx )
    {
      int const i = S( idx, 0 );
      int const j = S( idx, 1 );
      int const k = S( idx, 2 );
      int const cIJK = c( idx, sameSpin );

      REAL const factor = cIJK * std::pow( r12T, i - 2 ) * std::ow( r1T, j - 2 ) * std::pow( r2T, k );
      REAL const r1Factor  = (i + 1) * i * std::pow( a12 * r1T, 2 ) / std::pow( r12.r() + a12, 4 );
      REAL const r12Factor = (j + 1) * j * std::pow( a * r12T, 2 ) / std::pow( r1.r() + a, 4 );
      REAL const dotFactor = dot( r12, r1 ) * 2 * i * j * a12 * a  / (r12.r() * r1.r() * std::pow( r12.r() + a12, 2 ) * std::pow( r1.r() + a, 2 ));

      result = result + factor * (r1Factor + r12Factor + dotFactor);
    }

    return result;
  }

  Catesian< REAL > gradient( Spherical< REAL > const & r1, Spherical< REAL > const & r2, bool const sameSpin ) const
  {
    REAL const r12T = r12.r / (r12.r + a12);
    REAL const r1T = r1.r() / (r1.r() + a);
    REAL const r2T = r2.r() / (r2.r() + a);
  
    Cartesian< REAL > const result {};
    for( int idx = 0; idx < S.size( 0 ); ++idx )
    {
      int const i = S( idx, 0 );
      int const j = S( idx, 1 );
      int const k = S( idx, 2 );
      int const cIJK = c( idx, sameSpin );

      REAL const factor = cIJK * std::pow( r12T, i - 1 ) * std::ow( r1T, j - 1 ) * std::pow( r2T, k );
      REAL const r12Scaling = factor * (i * a12 * r1T / (r12 * std::pow( r12 - a12, 2 )));
      REAL const r1Scaling  = factor * (j * a * r12T / (r1 * std::pow( r1 - a, 2 )));
      RESULT += r12Scaling * r12C + r1Scaling * r1C;
    }

    return result;
  }


  REAL const a;
  REAL const a12;
  Array2d< REAL > const c;
  Array2d< int > const S;
};

} // namespace jastrowFunctions
} // namespace tcscf