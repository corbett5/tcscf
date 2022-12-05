#pragma once

#include <RAJA/RAJA.hpp>

#include <complex>

namespace tcscf
{

struct Serial
{
  template< int=0 >
  using DefaultPolicy = RAJA::loop_exec;

  template< int=0 >
  using DefaultAsyncPolicy = RAJA::loop_exec;

  using Resource = RAJA::resources::Host;
  using Atomic = RAJA::seq_atomic;
  using Reduce = RAJA::seq_reduce;
  static constexpr LvArray::MemorySpace SPACE = LvArray::MemorySpace::host;
  static constexpr int ID = 0;
};

struct ParallelHost
{
  template< int=0 >
  using DefaultPolicy = RAJA::omp_parallel_for_exec;

  template< int=0 >
  using DefaultAsyncPolicy = RAJA::omp_parallel_for_exec;

  using Resource = RAJA::resources::Host;
  using Atomic = RAJA::builtin_atomic;
  using Reduce = RAJA::omp_reduce;
  static constexpr LvArray::MemorySpace SPACE = LvArray::MemorySpace::host;
  static constexpr int ID = Serial::ID + 1;
};

#if defined( __CUDACC__ )
  struct ParallelDevice
  {

    template< int BLOCK_DIM=32 >
    using DefaultPolicy = RAJA::cuda_exec< BLOCK_DIM >;

    template< int BLOCK_DIM=32 >
    using DefaultAsyncPolicy = RAJA::cuda_exec_async< BLOCK_DIM >;
    
    using Resource = RAJA::resources::Cuda;
    using Atomic = RAJA::cuda_atomic;
    using Reduce = RAJA::cuda_reduce;

    static constexpr LvArray::MemorySpace SPACE = LvArray::MemorySpace::cuda;
    static constexpr int ID = ParallelHost::ID + 1;
  };
#else
  using ParallelDevice = ParallelHost;
#endif

template< typename POLICY_TYPE, int BLOCK_DIM=32 >
using DefaultPolicy = typename POLICY_TYPE::template DefaultPolicy< BLOCK_DIM >;

template< typename POLICY_TYPE >
using Resource = typename POLICY_TYPE::Resource;

template< typename POLICY_TYPE >
using Atomic = typename POLICY_TYPE::Atomic;

template< typename POLICY_TYPE >
using Reduce = typename POLICY_TYPE::Reduce;

template< typename POLICY_TYPE >
constexpr LvArray::MemorySpace SPACE = POLICY_TYPE::SPACE;

/**
 * @brief Iterate over the space [0, end) using @p POLICY, applying @p body at each iteration point.
 */
template< typename POLICY, typename LAMBDA >
inline void forAll( IndexType const end, LAMBDA && body )
{
  RAJA::forall< POLICY >( RAJA::TypedRangeSegment< IndexType >( 0, end ), std::forward< LAMBDA >( body ) );
}


/**
 * 
 */
template< typename POLICY_TYPE, typename T >
void atomicAdd( T * const ptr, T const & value )
{
  RAJA::atomicAdd< Atomic< POLICY_TYPE > >( ptr, value );
}

/**
 * 
 */
template< typename POLICY_TYPE, typename T >
void atomicAdd( std::complex< T > * const ptr, std::complex< T > const & value )
{
  RAJA::atomicAdd< Atomic< POLICY_TYPE > >( reinterpret_cast< T * >( ptr ), value.real() );
  RAJA::atomicAdd< Atomic< POLICY_TYPE > >( reinterpret_cast< T * >( ptr ) + 1, value.imag() );
}

} // namespace tcscf
