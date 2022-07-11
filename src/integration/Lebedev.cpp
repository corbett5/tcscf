#include "Lebedev.hpp"
#include "data/LebedevData.hpp"

namespace tcscf::integration::internal
{

template< typename REAL >
Array2d< REAL > convertLebedevData( IndexType const nPoints, long double const * const data )
{
  Array2d< REAL > array( 3, nPoints );

  for( IndexType i = 0; i < nPoints; ++i )
  {
    array( 0, i ) = data[ 3 * i + 0 ];
    array( 1, i ) = data[ 3 * i + 1 ];
    array( 2, i ) = data[ 3 * i + 2 ];
  }

  return array;
}

template< typename REAL >
ArrayView2d< REAL const > getLebedevGrid( int const order )
{
  static std::unordered_map< int, Array2d< REAL > > arrays;

  if( !arrays.count( order ) )
  {
    if( !lebedevCoefficients.count( order ) )
    {
      LVARRAY_ERROR( "Order " << order << " not supported." );
    }

    auto const orderAndPointer = lebedevCoefficients[ order ];
    arrays[ order ] = convertLebedevData< REAL >( orderAndPointer.first, orderAndPointer.second );
  }

  return arrays[ order ].toViewConst();
}

// Explicit instantiations.

template ArrayView2d< float const > getLebedevGrid< float >( int const order );
template ArrayView2d< double const > getLebedevGrid< double >( int const order );

} // namespace tcscf::integration::internal