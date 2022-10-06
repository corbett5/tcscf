#pragma once

#include "LvArrayInterface.hpp"

namespace tcscf::jastrowFunctions
{

template< typename REAL >
struct Ochi
{

  REAL operator()( REAL const r1, REAL const r2, REAL const r12, bool const sameSpin ) const
  {
    REAL const r12T = r12 / (r12 + a);
    REAL const r1T = r1 / (r1 + a);
    REAL const r2T = r2 / (r2 + a);
  
    REAL const result = 0;
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

  REAL laplacian( double const r1,
    Cartesian< double > const & r1C,
    double const r12,
    Cartesian< double > const & r12C,
    double const r2,
    bool const sameSpin ) const
  {
    REAL const r1T = r1 / (r1 + a);
    REAL const r12T = r12 / (r12 + a);
    REAL const r2T = r2 / (r2 + a);
  
    REAL result = 0;
    for( int idx = 0; idx < S.size( 0 ); ++idx )
    {
      int const i = S( idx, 0 );
      int const j = S( idx, 1 );
      int const k = S( idx, 2 );
      REAL const cIJK = c( idx, sameSpin );

      REAL const factor = cIJK * std::pow( r12T, i ) * std::pow( r1T, j ) * std::pow( r2T, k );
      REAL const r12Factor = (i + 1) * i * std::pow( a12 / (r12 * (r12 + a12)), 2 );
      REAL const r1Factor = (j + 1) * j * std::pow( a / (r1 * (r1 + a)), 2 );
      REAL const dotFactor = 2 * dot( r12C, r1C ) * i * j * a12 * a / (std::pow( r1 * r12, 2 ) * (r1 + a) * (r12 + a12));

      result = result + factor * (r1Factor + r12Factor + dotFactor);
    }

    return result;
  }

  Cartesian< REAL > gradient(
    double const r1,
    Cartesian< double > const & r1C,
    double const r12,
    Cartesian< double > const & r12C,
    double const r2,
    bool const sameSpin ) const
  {
    REAL const r1T = r1 / (r1 + a);
    REAL const r12T = r12 / (r12 + a);
    REAL const r2T = r2 / (r2 + a);
  
    Cartesian< REAL > result {};
    for( int idx = 0; idx < S.size( 0 ); ++idx )
    {
      int const i = S( idx, 0 );
      int const j = S( idx, 1 );
      int const k = S( idx, 2 );
      REAL const cIJK = c( idx, sameSpin );

      REAL const factor = cIJK * std::pow( r12T, i ) * std::pow( r1T, j ) * std::pow( r2T, k );
      REAL const r12Scaling = factor * i * a12 / (std::pow( r12, 2 ) * (r12 + a12));
      result.scaledAdd( r12Scaling, r12C );

      REAL const r1Scaling  = factor * j * a / (std::pow( r1, 2 ) * (r1 + a));
      result.scaledAdd( r1Scaling, r1C );
    }

    return result;
  }


  REAL const a;
  REAL const a12;
  Array2d< REAL > const c;
  Array2d< int > const S;
};

} // namespace tcscf::jastrowFunctions