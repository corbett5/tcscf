#include "blasLapackInterface.hpp"

/// This macro provide a flexible interface for Fortran naming convention for compiled objects
// #ifdef FORTRAN_MANGLE_NO_UNDERSCORE
#define FORTRAN_MANGLE( name ) name
// #else
// #define FORTRAN_MANGLE( name ) name ## _
// #endif

extern "C"
{

#define GEOSX_dgeev FORTRAN_MANGLE( dgeev )
void GEOSX_dgeev(
  char const * JOBVL,
  char const * JOBVR,
  int const * N,
  double * A,
  int const * LDA,
  double * WR,
  double * WI,
  double * VL,
  int const * LDVL,
  double * VR,
  int const * LDVR,
  double * WORK,
  int * LWORK,
  int * INFO );

#define GEOSX_cheev FORTRAN_MANGLE( cheev )
void GEOSX_cheev(
  char const * JOBZ,
  char const * UPLO,
  int const * N,
  std::complex< float > * A,
  int const * LDA,
  float * W,
  std::complex< float > * WORK,
  int const * LWORK,
  float const * RWORK,
  int * INFO
  );

#define GEOSX_zheev FORTRAN_MANGLE( zheev )
void GEOSX_zheev(
  char const * JOBZ,
  char const * UPLO,
  int const * N,
  std::complex< double > * A,
  int const * LDA,
  double * W,
  std::complex< double > * WORK,
  int const * LWORK,
  double const * RWORK,
  int * INFO
  );

} // extern "C"

namespace tcscf
{

void eigenvalues(
  ArrayView2d< double, 0 > const & A,
  ArrayView1d< std::complex< double > > const & lambda )
{
  LVARRAY_ASSERT_EQ_MSG( A.size( 0 ), A.size( 1 ), "The matrix A must be square." );

  LVARRAY_ASSERT_EQ_MSG( A.size( 0 ), lambda.size(), "The matrix A and lambda have incompatible sizes." );

  // define the arguments of dgesvd
  int const N = LvArray::integerConversion< int >( A.size( 0 ) );
  int const LDA  = N;
  int const LDVL = 1;
  int const LDVR = 1;
  int LWORK = 0;
  int INFO  = 0;
  double WKOPT = 0.0;
  double VL = 0.0;
  double VR = 0.0;

  Array1d< double > WR( N );
  Array1d< double > WI( N );

  // 1) query and allocate the optimal workspace
  LWORK = -1;
  GEOSX_dgeev( "N", "N",
               &N, A.data(), &LDA,
               WR.data(), WI.data(),
               &VL, &LDVL,
               &VR, &LDVR,
               &WKOPT, &LWORK, &INFO );

  LWORK = static_cast< int >( WKOPT );
  Array1d< double > WORK( LWORK );

  // 2) compute eigenvalues
  GEOSX_dgeev( "N", "N",
               &N, A.data(), &LDA,
               WR.data(), WI.data(),
               &VL, &LDVL,
               &VR, &LDVR,
               WORK.data(), &LWORK, &INFO );

  for( int i = 0; i < N; ++i )
  {
    lambda[ i ] = std::complex< double >( WR[ i ], WI[ i ] );
  }

  LVARRAY_ERROR_IF_NE_MSG( INFO, 0, "The algorithm computing eigenvalues failed to converge." );
}

template< typename T >
void hermitianEigendecomposition(
    ArrayView2d< std::complex< T >, 0 > const & A,
    ArrayView1d< T > const & eigenValues )
{
  LVARRAY_ASSERT_EQ_MSG( A.size( 0 ), A.size( 1 ),
    "The matrix A must be square." );

  LVARRAY_ASSERT_EQ_MSG( A.size( 0 ), eigenValues.size(),
    "The matrix A and lambda have incompatible sizes." );

  // define the arguments of zheev
  int const N = LvArray::integerConversion< int >( A.size( 0 ) );
  int const LDA = N;
  std::complex< T > optimalWorkSize{ 0, 0 };
  int LWORK = -1;
  int INFO;

  Array1d< T > rWork( std::max( 1, 3 * N - 2 ) );

  if constexpr ( std::is_same_v< T, float > )
  {
    GEOSX_cheev(
        "V",
        "U",
        &N,
        A.data(),
        &LDA,
        eigenValues.data(),
        &optimalWorkSize,
        &LWORK,
        rWork.data(),
        &INFO );
  }
  else
  {
    GEOSX_zheev(
        "V",
        "U",
        &N,
        A.data(),
        &LDA,
        eigenValues.data(),
        &optimalWorkSize,
        &LWORK,
        rWork.data(),
        &INFO );
  }

  LVARRAY_ERROR_IF_NE_MSG( INFO, 0,
    "Error in computing the optimal workspace size." );

  LWORK = static_cast< int >( optimalWorkSize.real() );
  Array1d< std::complex< T > > work( LWORK );

  if constexpr ( std::is_same_v< T, float > )
  {
    GEOSX_cheev(
    "V",
    "U",
    &N,
    A.data(),
    &LDA,
    eigenValues.data(),
    work.data(),
    &LWORK,
    rWork.data(),
    &INFO );
  }
  else
  {
    GEOSX_zheev(
    "V",
    "U",
    &N,
    A.data(),
    &LDA,
    eigenValues.data(),
    work.data(),
    &LWORK,
    rWork.data(),
    &INFO );
  }
  
  LVARRAY_ERROR_IF_NE_MSG( INFO, 0,
    "Error in computing the decomposition." );
}


// explicit instantiations.
template void hermitianEigendecomposition< float >(
  ArrayView2d< std::complex< float >, 0 > const & A,
  ArrayView1d< float > const & eigenValues );

template void hermitianEigendecomposition< double >(
  ArrayView2d< std::complex< double >, 0 > const & A,
  ArrayView1d< double > const & eigenValues );

} // namespace tcscf
