#include  "../OchiBasis.hpp"
#include  "../HydrogenLikeBasis.hpp"

#include "testingCommon.hpp"

namespace tcscf::testing
{


TEST( fillOneElectronHermitianMatrix, unity )
{
  Array2d< int > matrix( 10, 10 );
  fillOneElectronHermitianMatrix( matrix,
    [] ( int const LVARRAY_UNUSED_ARG( i ), int const LVARRAY_UNUSED_ARG( j ) )
    {
      return 1;
    }
  );

  for( int i = 0; i < matrix.size( 0 ); ++i )
  {
    for( int j = 0; j < matrix.size( 0 ); ++j )
    {
      EXPECT_EQ( matrix( i, j ), 1 );
    }
  }
}


TEST( fillOneElectronHermitianMatrix, Hermitian )
{
  Array2d< std::complex< int > > matrix( 10, 10 );
  fillOneElectronHermitianMatrix( matrix,
    [] ( int const i, int const j )
    {
      if( i == j ) return std::complex< int >( i, 0 );

      return std::complex< int >( i, j );
    }
  );

  for( int i = 0; i < matrix.size( 0 ); ++i )
  {
    for( int j = 0; j < matrix.size( 0 ); ++j )
    {
      EXPECT_EQ( matrix( i, j ), conj( matrix( j, i ) ) );
      
      if( i < j )
      {
        EXPECT_EQ( matrix( i, j ), std::complex< int >( i, j ) );
      }
      if( i == j )
      {
        EXPECT_EQ( matrix( i, j ), std::complex< int >( i, 0 ) );
      }
      if( i > j )
      {
        EXPECT_EQ( matrix( i, j ), std::complex< int >( j, -i ) );
      }
    }
  }
}

TEST( fillTwoElectronSymmetricHermitianArray, unity )
{
  int const N = 10;
  Array4d< int > array( N, N, N, N );
  fillTwoElectronSymmetricHermitianArray( array, false,
    [] ( int const, int const, int const, int const )
    {
      return 1;
    }
  );

  for( int a = 0; a < N; ++a )
  {
    for( int b = 0; b < N; ++b )
    {
      for( int c = 0; c < N; ++c )
      {
        for( int d = 0; d < N; ++d )
        {
          EXPECT_EQ( array( a, b, c, d ), 1 ) << a << ", " << b << ", " << c << ", " << d;
        }
      }
    }
  }
}

TEST( fillTwoElectronSymmetricHermitianArray, realUnity )
{
  int const N = 10;
  Array4d< int > array( N, N, N, N );
  fillTwoElectronSymmetricHermitianArray( array, true,
    [] ( int const, int const, int const, int const )
    {
      return 1;
    }
  );

  for( int a = 0; a < N; ++a )
  {
    for( int b = 0; b < N; ++b )
    {
      for( int c = 0; c < N; ++c )
      {
        for( int d = 0; d < N; ++d )
        {
          EXPECT_EQ( array( a, b, c, d ), 1 ) << a << ", " << b << ", " << c << ", " << d;
        }
      }
    }
  }
}

TEST( fillTwoElectronSymmetricHermitianArray, Hermitian )
{
  int const N = 10;
  Array4d< std::complex< int > > array( N, N, N, N );
  fillTwoElectronSymmetricHermitianArray( array, false,
    [] ( int const a, int const b, int const c, int const d )
    {
      return std::complex< int >( a + b + c + d, a + b - c - d );
    }
  );

  for( int a = 0; a < N; ++a )
  {
    for( int b = 0; b < N; ++b )
    {
      for( int c = 0; c < N; ++c )
      {
        for( int d = 0; d < N; ++d )
        {
          EXPECT_EQ( array( a, b, c, d ), array( b, a, d, c ) );
          EXPECT_EQ( array( a, b, c, d ), conj( array( c, d, a, b ) ) );
          EXPECT_EQ( array( a, b, c, d ), conj( array( d, c, b, a ) ) );
        }
      }
    }
  }
}

TEST( fillTwoElectronSymmetricHermitianArray, RealHermitian )
{
  int const N = 10;
  Array4d< std::complex< int > > array( N, N, N, N );
  fillTwoElectronSymmetricHermitianArray( array, true,
    [] ( int const a, int const b, int const c, int const d )
    {
      return a + b + c + d;
    }
  );

  for( int a = 0; a < N; ++a )
  {
    for( int b = 0; b < N; ++b )
    {
      for( int c = 0; c < N; ++c )
      {
        for( int d = 0; d < N; ++d )
        {
          EXPECT_EQ( array( a, b, c, d ), array( b, a, d, c ) );
          EXPECT_EQ( array( a, b, c, d ), array( c, d, a, b ) );
          EXPECT_EQ( array( a, b, c, d ), array( d, c, b, a ) );
        }
      }
    }
  }
}

TEST( fillTwoElectronSymmetricHermitianArray, parity )
{
  int const Z = 2;

  std::vector< HydrogenLikeBasisFunction< double > > basisFunctions {
    { Z, 1, 0, 0 },
    { Z, 2, 0, 0 },
    { Z, 3, 0, 0 },
  };

  IndexType const nBasis = basisFunctions.size();

  Array4d< std::complex< double > > array( nBasis, nBasis, nBasis, nBasis );

  for( int a = 0; a < nBasis; ++a )
  {
    for( int b = a; b < nBasis; ++b )
    {
      for( int c = a; c < nBasis; ++c )
      {
        for( int d = a; d < nBasis; ++d )
        {
          array( a, b, c, d ) = r12MatrixElement(
            basisFunctions[ a ],
            basisFunctions[ b ],
            basisFunctions[ c ],
            basisFunctions[ d ] );
        }
      }
    }
  }

  Array4d< std::complex< double > > arrayFromSymmetry( nBasis, nBasis, nBasis, nBasis );
  fillR12Array( basisFunctions, arrayFromSymmetry );

  for( int a = 0; a < nBasis; ++a )
  {
    for( int b = a; b < nBasis; ++b )
    {
      for( int c = a; c < nBasis; ++c )
      {
        for( int d = a; d < nBasis; ++d )
        {
          EXPECT_COMPLEX_NEAR( array( a, b, c, d ), arrayFromSymmetry( a, b, c, d ), 1e-2 );
        }
      }
    }
  }
}

TEST( fillAtomicR12Array, parity )
{
  int const Z = 2;

  std::vector< HydrogenLikeBasisFunction< double > > basisFunctions {
    { Z, 1, 0, 0 },
    { Z, 2, 0, 0 },
    { Z, 2, 1, 0 },
    { Z, 3, 1, 0 },
  };

  IndexType const nBasis = basisFunctions.size();

  Array4d< std::complex< double > > array( nBasis, nBasis, nBasis, nBasis );
  fillR12Array( basisFunctions, array );

  Array4d< std::complex< double > > atomicArray( nBasis, nBasis, nBasis, nBasis );
  fillAtomicR12Array( basisFunctions, atomicArray );

  for( int a = 0; a < nBasis; ++a )
  {
    for( int b = a; b < nBasis; ++b )
    {
      for( int c = a; c < nBasis; ++c )
      {
        for( int d = a; d < nBasis; ++d )
        {
          bool const coulomb = 
            basisFunctions[ a ].l == basisFunctions[ c ].l &&
            basisFunctions[ a ].m == basisFunctions[ c ].m &&
            basisFunctions[ b ].l == basisFunctions[ d ].l &&
            basisFunctions[ b ].m == basisFunctions[ d ].m;
          
          bool const exchange = 
            basisFunctions[ a ].l == basisFunctions[ d ].l &&
            basisFunctions[ a ].m == basisFunctions[ d ].m &&
            basisFunctions[ b ].l == basisFunctions[ c ].l &&
            basisFunctions[ b ].m == basisFunctions[ c ].m;
          
          if( coulomb || exchange )
          {
            EXPECT_COMPLEX_NEAR( array( a, b, c, d ), atomicArray( a, b, c, d ), 5e-3 );
          }
        }
      }
    }
  }
}

// TODO: move the following test into their own files.

// TEST( OchiBasis, orthogonal )
// {
//   double const alpha = 1.0;

//   for( int n1 = 0; n1 < 2; ++n1 )
//   {
//     for( int l1 = 0; l1 < 2; ++l1 )
//     {
//       for( int m1 = -l1; m1 <= l1; ++m1 )
//       {
//         for( int n2 = 0; n2 < 2; ++n2 )
//         {
//           for( int l2 = 0; l2 < 2; ++l2 )
//           {
//             for( int m2 = -l2; m2 <= l2; ++m2 )
//             {
//               OchiBasisFunction< double > b1( alpha, n1, l1, m1 );
//               OchiBasisFunction< double > b2( alpha, n2, l2, m2 );

//               std::complex< double > value = overlap( b1, b2 );
//               bool const delta = (n1 == n2) && (l1 == l2) && (m1 == m2);
//               EXPECT_COMPLEX_NEAR( value, delta, 1e-5 );
//             }
//           }
//         }
//       }
//     }
//   }
// }

// TEST( HydrogenLikeBasisFunction, orthogonal )
// {
//   int const Z = 1;

//   for( int n1 = 0; n1 < 3; ++n1 )
//   {
//     for( int l1 = 0; l1 < n1; ++l1 )
//     {
//       for( int m1 = -l1; m1 <= l1; ++m1 )
//       {
//         for( int n2 = 0; n2 < 3; ++n2 )
//         {
//           for( int l2 = 0; l2 < n2; ++l2 )
//           {
//             for( int m2 = -l2; m2 <= l2; ++m2 )
//             {
//               HydrogenLikeBasisFunction< double > b1( Z, n1, l1, m1 );
//               HydrogenLikeBasisFunction< double > b2( Z, n2, l2, m2 );

//               std::complex< double > value = overlap( b1, b2 );
//               bool const delta = (n1 == n2) && (l1 == l2) && (m1 == m2);
//               EXPECT_COMPLEX_NEAR( value, delta, 1e-5 );
//             }
//           }
//         }
//       }
//     }
//   }
// }

} // namespace tcscf::testing
