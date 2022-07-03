#pragma once

#include "blasLapackInterface.hpp"

namespace tcscf
{

template< typename T >
using RealVersion = decltype( std::real( T {} ) );

template< typename T >
struct RCSHartreeFock
{
  using Real = RealVersion< T >;

  RCSHartreeFock( IndexType const numElectrons, IndexType const numBasisFunctions ):
    nElectrons{ numElectrons },
    nBasis{ numBasisFunctions },
    density( nBasis, nBasis ),
    fockOperator( nBasis, nBasis ),
    eigenvalues( nBasis )
  {}
  
  void compute(
    bool const orthogonal,
    ArrayView2d< T const > const & overlap,
    ArrayView2d< T const > const & oneElectronTerms,
    ArrayView4d< T const > const & twoElectronTerms );

  IndexType const nElectrons;
  IndexType const nBasis;
  Array2d< T > const density;
  Array2d< T, RAJA::PERM_JI > const fockOperator;
  Array1d< Real > const eigenvalues;
};

} // namespace tcscf
