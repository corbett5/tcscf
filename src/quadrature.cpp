#include "quadrature.hpp"

namespace tcscf::quadrature::internal
{
  
static constexpr double lebedev6[ 6 ][ 3 ] = {
  {   0.000000000000000,    90.000000000000000,     0.166666666666667 },
  { 180.000000000000000,    90.000000000000000,     0.166666666666667 },
  {  90.000000000000000,    90.000000000000000,     0.166666666666667 },
  { -90.000000000000000,    90.000000000000000,     0.166666666666667 },
  {  90.000000000000000,     0.000000000000000,     0.166666666666667 },
  {  90.000000000000000,   180.000000000000000,     0.166666666666667 }
};

template< typename REAL, int ORDER >
Array2d< REAL > convertLebedevData( double const ( &data )[ ORDER ][ 3 ] )
{
  Array2d< REAL > array( 3, ORDER );

  for( IndexType i = 0; i < ORDER; ++i )
  {
    array( 0, i ) = data[ i ][ 1 ] * ( pi< double > / 180 );
    array( 1, i ) = data[ i ][ 0 ] * ( pi< double > / 180 );
    array( 2, i ) = 4 * pi< double > * data[ i ][ 2 ];
  }

  return array;
}

template< typename REAL >
Array2d< REAL > getLebedevGrid( int const order )
{
  if( order == 6 )
  {
    return convertLebedevData< REAL >( lebedev6 );
  }

  LVARRAY_ERROR( "Order " << order << " not supported. Supported orders are:" <<
                 "\t 6" );
  
  return {};
}

// Explicit instantiations.

template Array2d< float > getLebedevGrid< float >( int const order );
template Array2d< double > getLebedevGrid< double >( int const order );

} // namespace tcscf::quadrature::internal