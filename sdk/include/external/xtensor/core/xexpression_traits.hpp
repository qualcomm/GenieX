/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_EXPRESSION_TRAITS_HPP
#define XTENSOR_EXPRESSION_TRAITS_HPP

#include "../core/xexpression.hpp"

namespace xt
{
    /***************
     * xvalue_type *
     ***************/

    namespace detail
    {
        template <class E, class enable = void>
        struct xvalue_type_impl
        {
            using type = E;
        };

        template <class E>
        struct xvalue_type_impl<E, std::enable_if_t<is_xexpression<E>::value>>
        {
            using type = typename E::value_type;
        };
    }

    template <class E>
    using xvalue_type = detail::xvalue_type_impl<E>;

    template <class E>
    using xvalue_type_t = typename xvalue_type<E>::type;

    /*********************
     * common_value_type *
     *********************/

    template <class... C>
    struct common_value_type
    {
        using type = std::common_type_t<typename std::decay_t<C>::value_type...>;
    };

    template <class... C>
    using common_value_type_t = typename common_value_type<C...>::type;

    /********************
     * common_size_type *
     ********************/

    template <class... Args>
    struct common_size_type
    {
        using type = std::common_type_t<typename Args::size_type...>;
    };

    template <>
    struct common_size_type<>
    {
        using type = std::size_t;
    };

    template <class... Args>
    using common_size_type_t = typename common_size_type<Args...>::type;

    /**************************
     * common_difference type *
     **************************/

    template <class... Args>
    struct common_difference_type
    {
        using type = std::common_type_t<typename Args::difference_type...>;
    };

    template <>
    struct common_difference_type<>
    {
        using type = std::ptrdiff_t;
    };

    template <class... Args>
    using common_difference_type_t = typename common_difference_type<Args...>::type;

    /******************
     * temporary_type *
     ******************/

    namespace detail
    {
        template <class S>
        struct xtype_for_shape
        {
            template <class T, layout_type L>
            using type = xarray<T, L>;
        };

// Workaround for rebind_container problems when C++17 feature is enabled
#ifdef __cpp_template_template_args
        template <template <class, std::size_t, class, bool> class S, class X, std::size_t N, class A, bool Init>
        struct xtype_for_shape<S<X, N, A, Init>>
        {
            template <class T, layout_type L>
            using type = xarray<T, L>;
        };
#endif  // __cpp_template_template_args

        template <template <class, std::size_t> class S, class X, std::size_t N>
        struct xtype_for_shape<S<X, N>>
        {
            template <class T, layout_type L>
            using type = xtensor<T, N, L>;
        };

        template <template <std::size_t...> class S, std::size_t... X>
        struct xtype_for_shape<S<X...>>
        {
            template <class T, layout_type L>
            using type = xtensor_fixed<T, xshape<X...>, L>;
        };
    }

    template <class Tag, class T>
    struct temporary_type_from_tag;

    template <class T>
    struct temporary_type_from_tag<xtensor_expression_tag, T>
    {
        using I = std::decay_t<T>;
        using shape_type = typename I::shape_type;
        using value_type = typename I::value_type;
        static constexpr layout_type static_layout = XTENSOR_DEFAULT_LAYOUT;
        using type = typename detail::xtype_for_shape<shape_type>::template type<value_type, static_layout>;
    };

    template <class T, class = void>
    struct temporary_type
    {
        using type = typename temporary_type_from_tag<xexpression_tag_t<T>, T>::type;
    };

    template <class T>
    struct temporary_type<T, void_t<typename std::decay_t<T>::temporary_type>>
    {
        using type = typename std::decay_t<T>::temporary_type;
    };

    template <class T>
    using temporary_type_t = typename temporary_type<T>::type;

    /**********************
     * common_tensor_type *
     **********************/

    namespace detail
    {
        template <class... C>
        struct common_tensor_type_impl
        {
            static constexpr layout_type static_layout = compute_layout(std::decay_t<C>::static_layout...);
            using value_type = common_value_type_t<C...>;
            using shape_type = promote_shape_t<typename C::shape_type...>;
            using type = typename xtype_for_shape<shape_type>::template type<value_type, static_layout>;
        };
    }

    template <class... C>
    struct common_tensor_type : detail::common_tensor_type_impl<std::decay_t<C>...>
    {
    };

    template <class... C>
    using common_tensor_type_t = typename common_tensor_type<C...>::type;

    /**************************
     * big_promote_value_type *
     **************************/

    template <class E>
    struct big_promote_value_type
    {
        using type = xtl::big_promote_type_t<typename std::decay_t<E>::value_type>;
    };

    template <class E>
    using big_promote_value_type_t = typename big_promote_value_type<E>::type;
}

#endif
