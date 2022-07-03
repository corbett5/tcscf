#pragma once

#include  "LvArrayInterface.hpp"

#include <complex>

namespace tcscf
{

void eigenvalues(
  ArrayView2d< double, 0 > const & A,
  ArrayView1d< std::complex< double > > const & lambda );


template< typename T >
void hermitianEigendecomposition(
    ArrayView2d< std::complex< T >, 0 > const & A,
    ArrayView1d< T > const & eigenValues );

} // namespace tcscf