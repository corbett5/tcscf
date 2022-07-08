#pragma once

#include "blasLapackInterface.hpp"

#include "AtomicBasis.hpp"

namespace tcscf
{

template< typename T >
struct RCSHartreeFock
{
  using Real = RealType< T >;

  RCSHartreeFock( int const numElectrons, int const numBasisFunctions ):
    nElectrons{ numElectrons },
    basisSize{ numBasisFunctions },
    density( basisSize, basisSize ),
    fockOperator( basisSize, basisSize ),
    eigenvalues( basisSize )
  {}
  
  void compute(
    bool const orthogonal,
    ArrayView2d< T const > const & overlap,
    ArrayView2d< T const > const & oneElectronTerms,
    ArrayView4d< T const > const & twoElectronTerms );

  int const nElectrons;
  int const basisSize;
  Array2d< T > const density;
  Array2d< T, RAJA::PERM_JI > const fockOperator;
  Array1d< Real > const eigenvalues;
};



template< typename REAL >
struct AtomicRCSHartreeFock
{
  using Real = REAL;

  AtomicRCSHartreeFock(
    int const numElectrons,
    std::vector< AtomicParams > const & atomicParams ):
    nElectrons{ numElectrons },
    params( atomicParams ),
    density( params.size(), params.size() ),
    fockOperator( params.size(), params.size() ),
    eigenvalues( params.size() )
  {}

  void compute(
    ArrayView2d< Real const > const & oneElectronTerms,
    ArrayView4d< std::complex< Real > const > const & twoElectronTerms );

  int const nElectrons;
  std::vector< AtomicParams > const params;
  Array2d< std::complex< Real > > const density;
  Array2d< std::complex< Real >, RAJA::PERM_JI > const fockOperator;
  Array1d< Real > const eigenvalues;
};

} // namespace tcscf
