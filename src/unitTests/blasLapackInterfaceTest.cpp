#include  "../LvArrayInterface.hpp"
#include "../blasLapackInterface.hpp"

#include "testingCommon.hpp"

#include <random>

namespace tcscf::testing
{

TEST( blasLapack, eigenvalues )
{
  tcscf::Array2d< double, RAJA::PERM_JI > test( 3, 3 );
  test(0, 0) = 3;
  test(1, 1) = 2;
  test(2, 2) = -4;


  tcscf::Array1d< std::complex< double > > eigenvalues( 3 );

  tcscf::eigenvalues( test, eigenvalues );

  EXPECT_DOUBLE_EQ( eigenvalues[ 0 ].real(), 3 );
  EXPECT_DOUBLE_EQ( eigenvalues[ 0 ].imag(), 0 );
  
  EXPECT_DOUBLE_EQ( eigenvalues[ 1 ].real(), 2 );
  EXPECT_DOUBLE_EQ( eigenvalues[ 1 ].imag(), 0 );
  
  EXPECT_DOUBLE_EQ( eigenvalues[ 2 ].real(), -4 );
  EXPECT_DOUBLE_EQ( eigenvalues[ 2 ].imag(), 0 );
}

TEST( blasLapack, hermitianEigendecomposition )
{
  tcscf::Array2d< std::complex< double >, RAJA::PERM_JI > matrix( 3, 3 );
  matrix(0, 0) = 3;
  matrix(1, 1) = 2;
  matrix(2, 2) = -4;

  tcscf::Array1d< double > eigenvalues( 3 );

  tcscf::hermitianEigendecomposition( matrix, eigenvalues );

  EXPECT_DOUBLE_EQ( eigenvalues[ 0 ], -4 );
  EXPECT_COMPLEX_EQ( matrix(0, 0), 0 );
  EXPECT_COMPLEX_EQ( matrix(0, 1), 0 );
  EXPECT_COMPLEX_EQ( matrix(0, 2), 1 );
  
  EXPECT_DOUBLE_EQ( eigenvalues[ 1 ], 2 );
  EXPECT_COMPLEX_EQ( matrix(1, 0), 0 );
  EXPECT_COMPLEX_EQ( matrix(1, 1), 1 );
  EXPECT_COMPLEX_EQ( matrix(1, 2), 0 );
  
  EXPECT_DOUBLE_EQ( eigenvalues[ 2 ], 3 );
  EXPECT_COMPLEX_EQ( matrix(2, 0), 1 );
  EXPECT_COMPLEX_EQ( matrix(2, 1), 0 );
  EXPECT_COMPLEX_EQ( matrix(2, 2), 0 );
}

template< typename T >
struct SymmetricEigendecomposition : public ::testing::Test
{
  using Real = RealType< T >;

  SymmetricEigendecomposition():
    m_gen( std::random_device {}() ),
    m_dist( -10, 10 )
  {}

  void test( int const size )
  {
    Array2d< T, RAJA::PERM_JI > matrix( size, size );

    for( IndexType i = 0; i < size; ++i )
    {
      matrix( i, i ) = std::real( randomNumber() );
      for( IndexType j = i + 1; j < size; ++j )
      {
        matrix( i, j ) = randomNumber();
        matrix( j, i ) = conj( matrix( i, j ) );
      }
    }

    // symmetricEigendecomposition overwrites the matrix.
    Array2d< T, RAJA::PERM_JI > eigenvectors( matrix );

    Array1d< Real > eigenvalues( size );

    hermitianEigendecomposition( eigenvectors.toView(), eigenvalues.toView() );

    for( IndexType i = 0; i < size; ++i )
    {
      if( i > 0 )
      {
        // Verify that the eigenvalues are in ascending order.
        EXPECT_GE( eigenvalues[ i ], eigenvalues[ i - 1 ] );
      }

      for( IndexType j = 0; j < size; ++j )
      {
        T value = 0;
        for( IndexType k = 0; k < size; ++k )
        {
          value += matrix( j, k ) * eigenvectors( k, i );
        }

        EXPECT_COMPLEX_NEAR( value, eigenvalues[ i ] * eigenvectors( j, i ), 500 * std::numeric_limits< Real >::epsilon() );
      }
    }
  }

private:

  // template< typename _T=T >
  // std::enable_if_t< std::is_same_v< Real, T >, T >
  // randomNumber()
  // { return m_dist( m_gen ); }

  template< typename _T=T >
  std::enable_if_t< !std::is_same_v< Real, T >, T >
  randomNumber()
  { return { m_dist( m_gen ), m_dist( m_gen ) }; }

  std::mt19937_64 m_gen;
  std::uniform_real_distribution< Real > m_dist;
};

using SymmetricEigendecompositionTypes = ::testing::Types<
  std::complex< float >,
  std::complex< double >
  >;
TYPED_TEST_SUITE( SymmetricEigendecomposition, SymmetricEigendecompositionTypes, );

TYPED_TEST( SymmetricEigendecomposition, 2x2 )
{
  this->test( 2 );
}

TYPED_TEST( SymmetricEigendecomposition, 3x3 )
{
  this->test( 3 );
}

TYPED_TEST( SymmetricEigendecomposition, 10x10 )
{
  this->test( 10 );
}

TYPED_TEST( SymmetricEigendecomposition, 100x100 )
{
  this->test( 100 );
}

} // namespace tcscf::testing
