/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_FORWARD_HPP
#define XTENSOR_FORWARD_HPP

// This file contains forward declarations and
// alias types to solve the problem of circular
// includes. It should not contain anything else
// and should not bring additional dependencies to
// the files that include it. So:
// - do NOT define classes of metafunctions here
// - do NOT include other headers
//
// If you need to do so, something is probably
// going wrong (either your change, or xtensor
// needs to be refactored).

#include <memory>
#include <vector>

#include <xtl/xoptional_sequence.hpp>

#include "../core/xlayout.hpp"
#include "../core/xtensor_config.hpp"

namespace xt
{
    struct xtensor_expression_tag;
    struct xoptional_expression_tag;

    template <class C>
    struct xcontainer_inner_types;

    template <class D>
    class xcontainer;

    template <class T, class A>
    class uvector;

    template <class T, std::size_t N, class A, bool Init>
    class svector;

    template <
        class EC,
        layout_type L = XTENSOR_DEFAULT_LAYOUT,
        class SC = XTENSOR_DEFAULT_SHAPE_CONTAINER(typename EC::value_type, typename EC::allocator_type, std::allocator<typename EC::size_type>),
        class Tag = xtensor_expression_tag>
    class xarray_container;

    /**
     * @typedef xarray
     * Alias template on xarray_container with default parameters for data container
     * type and shape / strides container type. This allows to write
     *
     * @code{.cpp}
     * xt::xarray<double> a = {{1., 2.}, {3., 4.}};
     * @endcode
     *
     * instead of the heavier syntax
     *
     * @code{.cpp}
     * xt::xarray_container<std::vector<double>, std::vector<std::size_t>> a = ...
     * @endcode
     *
     * @tparam T The value type of the elements.
     * @tparam L The layout_type of the xarray_container (default: XTENSOR_DEFAULT_LAYOUT).
     * @tparam A The allocator of the container holding the elements.
     * @tparam SA The allocator of the containers holding the shape and the strides.
     */
    template <
        class T,
        layout_type L = XTENSOR_DEFAULT_LAYOUT,
        class A = XTENSOR_DEFAULT_ALLOCATOR(T),
        class SA = std::allocator<typename std::vector<T, A>::size_type>>
    using xarray = xarray_container<XTENSOR_DEFAULT_DATA_CONTAINER(T, A), L, XTENSOR_DEFAULT_SHAPE_CONTAINER(T, A, SA)>;

    template <
        class EC,
        layout_type L = XTENSOR_DEFAULT_LAYOUT,
        class SC = XTENSOR_DEFAULT_SHAPE_CONTAINER(typename EC::value_type, std::allocator<typename EC::size_type>, std::allocator<typename EC::size_type>),
        class Tag = xtensor_expression_tag>
    class xarray_adaptor;

    /**
     * @typedef xarray_optional
     * Alias template on xarray_container for handling missing values
     *
     * @tparam T The value type of the elements.
     * @tparam L The layout_type of the container (default: XTENSOR_DEFAULT_LAYOUT).
     * @tparam A The allocator of the container holding the elements.
     * @tparam BA The allocator of the container holding the missing flags.
     * @tparam SA The allocator of the containers holding the shape and the strides.
     */
    template <
        class T,
        layout_type L = XTENSOR_DEFAULT_LAYOUT,
        class A = XTENSOR_DEFAULT_ALLOCATOR(T),
        class BC = xtl::xdynamic_bitset<std::size_t>,
        class SA = std::allocator<typename std::vector<T, A>::size_type>>
    using xarray_optional = xarray_container<
        xtl::xoptional_vector<T, A, BC>,
        L,
        XTENSOR_DEFAULT_SHAPE_CONTAINER(T, A, SA),
        xoptional_expression_tag>;

    template <class EC, std::size_t N, layout_type L = XTENSOR_DEFAULT_LAYOUT, class Tag = xtensor_expression_tag>
    class xtensor_container;

    /**
     * @typedef xtensor
     * Alias template on xtensor_container with default parameters for data container
     * type. This allows to write
     *
     * @code{.cpp}
     * xt::xtensor<double, 2> a = {{1., 2.}, {3., 4.}};
     * @endcode
     *
     * instead of the heavier syntax
     *
     * @code{.cpp}
     * xt::xtensor_container<std::vector<double>, 2> a = ...
     * @endcode
     *
     * @tparam T The value type of the elements.
     * @tparam N The dimension of the tensor.
     * @tparam L The layout_type of the tensor (default: XTENSOR_DEFAULT_LAYOUT).
     * @tparam A The allocator of the containers holding the elements.
     */
    template <class T, std::size_t N, layout_type L = XTENSOR_DEFAULT_LAYOUT, class A = XTENSOR_DEFAULT_ALLOCATOR(T)>
    using xtensor = xtensor_container<XTENSOR_DEFAULT_DATA_CONTAINER(T, A), N, L>;

    template <class EC, std::size_t N, layout_type L = XTENSOR_DEFAULT_LAYOUT, class Tag = xtensor_expression_tag>
    class xtensor_adaptor;

    template <class EC, std::size_t N, layout_type L = XTENSOR_DEFAULT_LAYOUT, class Tag = xtensor_expression_tag>
    class xtensor_view;

    template <std::size_t... N>
    class fixed_shape;

    /**
     * @typedef xshape
     * Alias template for ``fixed_shape`` allows for a shorter template shape definition in ``xtensor_fixed``.
     */
    template <std::size_t... N>
    using xshape = fixed_shape<N...>;

    template <class ET, class S, layout_type L = XTENSOR_DEFAULT_LAYOUT, bool Sharable = true, class Tag = xtensor_expression_tag>
    class xfixed_container;

    template <class ET, class S, layout_type L = XTENSOR_DEFAULT_LAYOUT, bool Sharable = true, class Tag = xtensor_expression_tag>
    class xfixed_adaptor;

    /**
     * @typedef xtensor_fixed
     * Alias template on xfixed_container with default parameters for layout
     * type. This allows to write
     *
     * @code{.cpp}
     * xt::xtensor_fixed<double, xt::xshape<2, 2>> a = {{1., 2.}, {3., 4.}};
     * @endcode
     *
     * instead of the syntax
     *
     * @code{.cpp}
     * xt::xfixed_container<double, xt::xshape<2, 2>, xt::layout_type::row_major> a = ...
     * @endcode
     *
     * @tparam T The value type of the elements.
     * @tparam FSH A xshape template shape.
     * @tparam L The layout_type of the tensor (default: XTENSOR_DEFAULT_LAYOUT).
     * @tparam Sharable Whether the tensor can be used in a shared expression.
     */
    template <class T, class FSH, layout_type L = XTENSOR_DEFAULT_LAYOUT, bool Sharable = true>
    using xtensor_fixed = xfixed_container<T, FSH, L, Sharable>;

    /**
     * @typedef xtensor_optional
     * Alias template on xtensor_container for handling missing values
     *
     * @tparam T The value type of the elements.
     * @tparam N The dimension of the tensor.
     * @tparam L The layout_type of the container (default: XTENSOR_DEFAULT_LAYOUT).
     * @tparam A The allocator of the containers holding the elements.
     * @tparam BA The allocator of the container holding the missing flags.
     */
    template <
        class T,
        std::size_t N,
        layout_type L = XTENSOR_DEFAULT_LAYOUT,
        class A = XTENSOR_DEFAULT_ALLOCATOR(T),
        class BC = xtl::xdynamic_bitset<std::size_t>>
    using xtensor_optional = xtensor_container<xtl::xoptional_vector<T, A, BC>, N, L, xoptional_expression_tag>;

    template <class CT, class... S>
    class xview;

    template <class F, class... CT>
    class xfunction;
}

#endif
