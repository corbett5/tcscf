#include  "../orthogonalization.hpp"

#include "testingCommon.hpp"

#include <random>

namespace tcscf::testing
{

template< typename T >
void verifyOrthogonal( ArrayView2d< T const > const & vectors )
{
  for( std::size_t i = 0; i < vectors.size(); ++i )
  {
    for( std::size_t j = 0; j < i; ++j )
    {
      T const dot = orthogonalization::internal::innerProduct( vectors[ i ], vectors[ j ] );
      EXPECT_COMPLEX_NEAR( dot, (i == j), vectors.size( 1 ) * 1e-15 );
    }
  }
}

template< typename T >
void testOrthogonalization( IndexType const k, IndexType const n )
{
  Array2d< T > vectors( k, n );

  std::mt19937 gen;
  std::uniform_real_distribution< T > dis( -10, 10 );

  for( auto & value : vectors )
  {
    value = dis( gen );
  }

  std::vector< ArraySlice1d< T > > vectorOfVectors;
  for( IndexType i = 0; i < k; ++i )
  {
    vectorOfVectors.emplace_back( vectors[ i ] );
  }

  orthogonalization::modifiedGramSchmidt( vectorOfVectors );

  verifyOrthogonal( vectors.toViewConst() );
  // verifySpan( vectors.toViewConst(), originalVectors.toViewConst() );
}

TEST( orthogonalization, RandomVectorsFullSubspace )
{
  testOrthogonalization< double >( 5, 5 );
  testOrthogonalization< double >( 10, 10 );
  testOrthogonalization< double >( 20, 20 );
  testOrthogonalization< double >( 40, 40 );
}

TEST( orthogonalization, RandomVectorsPartialSubspace )
{
  testOrthogonalization< double >( 2, 5 );
  testOrthogonalization< double >( 5, 10 );
  testOrthogonalization< double >( 8, 20 );
  testOrthogonalization< double >( 10, 40 );
}

} // namespace tcscf::testing
