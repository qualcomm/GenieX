/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_CONCEPTS_HPP
#define XTENSOR_CONCEPTS_HPP

#include <xtl/xcomplex.hpp>
#include <xtl/xtype_traits.hpp>

namespace xtl
{
    template <typename T>
    concept integral_concept = xtl::is_integral<T>::value;
    template <typename T>
    concept non_integral_concept = !xtl::is_integral<T>::value;
    template <typename T>
    concept complex_concept = xtl::is_complex<typename std::decay<T>::type::value_type>::value;
}

#endif  // XTENSOR_CONCEPTS_HPP
