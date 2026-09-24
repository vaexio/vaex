//  (c) Copyright Fernando Luis Cacciola Carballal 2000-2004
//  Use, modification, and distribution is subject to the Boost Software
//  License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//  See library home page at http://www.boost.org/libs/numeric/conversion
//
// Contact the author at: fernando_cacciola@hotmail.com
// 
#ifndef BOOST_NUMERIC_CONVERSION_DETAIL_SIGN_MIXTURE_FLC_12NOV2002_HPP
#define BOOST_NUMERIC_CONVERSION_DETAIL_SIGN_MIXTURE_FLC_12NOV2002_HPP

#include "boost/config.hpp"
#include "boost/limits.hpp"

#include "boost/numeric/conversion/sign_mixture_enum.hpp"
#include "boost/numeric/conversion/detail/meta.hpp"

#include "boost/mpl/integral_c.hpp"
// Use boost::integral_constant instead of mpl::integral_c here: mpl::integral_c eagerly
// instantiates prior/next typedefs via static_cast<EnumType>(value -/+ 1), which produces
// an out-of-range enum value for the first/last enumerator. Modern Clang rejects this as
// "non-type template argument is not a constant expression". boost::integral_constant
// exposes the same ::value/::tag interface without that eager instantiation.
#include "boost/type_traits/integral_constant.hpp"

namespace boost { namespace numeric { namespace convdetail
{
  // Integral Constants for 'SignMixture'
  typedef boost::integral_constant<sign_mixture_enum, unsigned_to_unsigned> unsig2unsig_c ;
  typedef boost::integral_constant<sign_mixture_enum, signed_to_signed>     sig2sig_c ;
  typedef boost::integral_constant<sign_mixture_enum, signed_to_unsigned>   sig2unsig_c ;
  typedef boost::integral_constant<sign_mixture_enum, unsigned_to_signed>   unsig2sig_c ;

  // Metafunction:
  //
  //   get_sign_mixture<T,S>::type
  //
  // Selects the appropriate SignMixture Integral Constant for the combination T,S.
  //
  template<class T,class S>
  struct get_sign_mixture
  {
    typedef mpl::bool_< ::std::numeric_limits<S>::is_signed > S_signed ;
    typedef mpl::bool_< ::std::numeric_limits<T>::is_signed > T_signed ;

    typedef typename
      for_both<S_signed, T_signed, sig2sig_c, sig2unsig_c, unsig2sig_c, unsig2unsig_c>::type
        type ;
  } ;

  // Metafunction:
  //
  //   for_sign_mixture<SignMixture,Sig2Sig,Sig2Unsig,Unsig2Sig,Unsig2Unsig>::type
  //
  // {SignMixture} is one of the Integral Constants for SignMixture, declared above.
  // {Sig2Sig,Sig2Unsig,Unsig2Sig,Unsig2Unsig} are aribtrary types. (not metafunctions)
  //
  // According to the value of 'SignMixture', selects the corresponding type.
  //
  template<class SignMixture, class Sig2Sig, class Sig2Unsig, class Unsig2Sig, class Unsig2Unsig>
  struct for_sign_mixture
  {
    typedef typename
      ct_switch4<SignMixture
                 , sig2sig_c, sig2unsig_c, unsig2sig_c  // default
                 , Sig2Sig  , Sig2Unsig  , Unsig2Sig  , Unsig2Unsig
                >::type
        type ;
  } ;

} } } // namespace boost::numeric::convdetail

#endif
//
///////////////////////////////////////////////////////////////////////////////////////////////


