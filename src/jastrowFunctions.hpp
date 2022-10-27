#pragma once

#include "LvArrayInterface.hpp"

namespace tcscf::jastrowFunctions
{

template< typename REAL >
struct Ochi
{
  using Real = REAL;

  Real operator()( Real const r1, Real const r2, Real const r12, bool const sameSpin ) const
  {
    Real const r12T = r12 / (r12 + a);
    Real const r1T = r1 / (r1 + a);
    Real const r2T = r2 / (r2 + a);
  
    Real const result = 0;
    for( int idx = 0; idx < S.size( 0 ); ++idx )
    {
      int const i = S( idx, 0 );
      int const j = S( idx, 1 );
      int const k = S( idx, 2 );
      int const cIJK = c( idx, sameSpin );

      result = result + cIJK * std::pow( r12T, i ) * std::pow( r1T, j ) * std::pow( r2T, k );
    }

    return result;
  }

  Real laplacian(
    Cartesian< Real > const & r1C,
    Cartesian< Real > const & r2C,
    bool const sameSpin ) const
  {
    Cartesian< Real > const r12C = r1C - r2C;

    Real const r1 = r1C.r();
    Real const r2 = r1C.r();
    Real const r12 = r12C.r();

    Real const r1T = r1 / (r1 + a);
    Real const r12T = r12 / (r12 + a);
    Real const r2T = r2 / (r2 + a);

    Real const r1Scale = std::pow( a / (r1 * (r1 + a)), 2 );
    Real const r12Scale = std::pow( a12 / (r12 * (r12 + a12)), 2 );
    Real const r1DotR12Scale = 2 * dot( r1C, r12C ) * a12 * a / (std::pow( r1 * r12, 2 ) * (r1 + a) * (r12 + a12));
  
    Real result = 0;
    for( int idx = 0; idx < S.size( 0 ); ++idx )
    {
      int const i = S( idx, 0 );
      int const j = S( idx, 1 );
      int const k = S( idx, 2 );
      Real const cIJK = c( idx, sameSpin );

      Real const factor = cIJK * std::pow( r12T, i ) * std::pow( r1T, j ) * std::pow( r2T, k );
      Real const r1Factor = (j + 1) * j * r1Scale;
      Real const r12Factor = (i + 1) * i * r12Scale;
      Real const dotFactor = i * j * r1DotR12Scale;

      result = result + factor * (r1Factor + r12Factor + dotFactor);
    }

    return result;
  }

  Cartesian< Real > gradient(
    Cartesian< Real > const & r1C,
    Cartesian< Real > const & r2C,
    bool const sameSpin ) const
  {
    Cartesian< Real > const r12C = r1C - r2C;

    Real const r1 = r1C.r();
    Real const r2 = r1C.r();
    Real const r12 = r12C.r();

    Real const r1T = r1 / (r1 + a);
    Real const r12T = r12 / (r12 + a);
    Real const r2T = r2 / (r2 + a);

    Real const r1Scale = a / (std::pow( r1, 2 ) * (r1 + a));
    Real const r2Scale = a12 / (std::pow( r12, 2 ) * (r12 + a12));
  
    Real r1Length = 0;
    Real r12Length = 0;
    for( int idx = 0; idx < S.size( 0 ); ++idx )
    {
      int const i = S( idx, 0 );
      int const j = S( idx, 1 );
      int const k = S( idx, 2 );
      Real const cIJK = c( idx, sameSpin );

      Real const factor = cIJK * std::pow( r12T, i ) * std::pow( r1T, j ) * std::pow( r2T, k );
      r1Length += factor * j * r1Scale;
      r12Length += factor * i * r2Scale;
    }

    return {
      r12Length * r12C.x() + r1Length * r1C.x(),
      r12Length * r12C.y() + r1Length * r1C.y(),
      r12Length * r12C.y() + r1Length * r1C.y() };
  }


  Real const a;
  Real const a12;
  Array2d< Real > const c;
  Array2d< int > const S;
};

} // namespace tcscf::jastrowFunctions
