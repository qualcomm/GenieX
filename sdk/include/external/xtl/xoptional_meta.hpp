/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_OPTIONAL_META_HPP
#define XTL_OPTIONAL_META_HPP

#include <type_traits>

#include "xmasked_value_meta.hpp"
#include "xmeta_utils.hpp"
#include "xtype_traits.hpp"

namespace xtl
{
    template <class CT, class CB = bool>
    class xoptional;

    namespace detail
    {
        template <class E>
        struct is_xoptional_impl : std::false_type
        {
        };

        template <class CT, class CB>
        struct is_xoptional_impl<xoptional<CT, CB>> : std::true_type
        {
        };

        template <class CT, class CTO, class CBO>
        using converts_from_xoptional = std::disjunction<
            std::is_constructible<CT, const xoptional<CTO, CBO>&>,
            std::is_constructible<CT, xoptional<CTO, CBO>&>,
            std::is_constructible<CT, const xoptional<CTO, CBO>&&>,
            std::is_constructible<CT, xoptional<CTO, CBO>&&>,
            std::is_convertible<const xoptional<CTO, CBO>&, CT>,
            std::is_convertible<xoptional<CTO, CBO>&, CT>,
            std::is_convertible<const xoptional<CTO, CBO>&&, CT>,
            std::is_convertible<xoptional<CTO, CBO>&&, CT>
        >;

        template <class CT, class CTO, class CBO>
        using assigns_from_xoptional = std::disjunction<
            std::is_assignable<std::add_lvalue_reference_t<CT>, const xoptional<CTO, CBO>&>,
            std::is_assignable<std::add_lvalue_reference_t<CT>, xoptional<CTO, CBO>&>,
            std::is_assignable<std::add_lvalue_reference_t<CT>, const xoptional<CTO, CBO>&&>,
            std::is_assignable<std::add_lvalue_reference_t<CT>, xoptional<CTO, CBO>&&>
        >;

        template <class... Args>
        struct common_optional_impl;

        template <class T>
        struct common_optional_impl<T>
        {
            using type = std::conditional_t<is_xoptional_impl<T>::value, T, xoptional<T>>;
        };

        template <class T>
        struct identity
        {
            using type = T;
        };

        template <class T>
        struct get_value_type
        {
            using type = typename T::value_type;
        };

        template<class T1, class T2>
        struct common_optional_impl<T1, T2>
        {
            using decay_t1 = std::decay_t<T1>;
            using decay_t2 = std::decay_t<T2>;
            using type1 = xtl::mpl::eval_if_t<xtl::is_fundamental<decay_t1>, identity<decay_t1>, get_value_type<decay_t1>>;
            using type2 = xtl::mpl::eval_if_t<xtl::is_fundamental<decay_t2>, identity<decay_t2>, get_value_type<decay_t2>>;
            using type = xoptional<std::common_type_t<type1, type2>>;
        };

        template <class T1, class T2, class B2>
        struct common_optional_impl<T1, xoptional<T2, B2>>
            : common_optional_impl<T1, T2>
        {
        };

        template <class T1, class B1, class T2>
        struct common_optional_impl<xoptional<T1, B1>, T2>
            : common_optional_impl<T1, T2>
        {
        };

        template <class T1, class B1, class T2, class B2>
        struct common_optional_impl<xoptional<T1, B1>, xoptional<T2, B2>>
            : common_optional_impl<T1, T2>
        {
        };

        template <class T1, class T2, class... Args>
        struct common_optional_impl<T1, T2, Args...>
        {
            using type = typename common_optional_impl<
                             typename common_optional_impl<T1, T2>::type,
                             Args...
                         >::type;
        };
    }

    template <class E>
    using is_xoptional = detail::is_xoptional_impl<E>;

    template <class E, class R = void>
    using disable_xoptional = std::enable_if_t<!is_xoptional<E>::value, R>;

    template <class... Args>
    struct at_least_one_xoptional : std::disjunction<is_xoptional<Args>...>
    {
    };

    template <class... Args>
    struct common_optional : detail::common_optional_impl<Args...>
    {
    };

    template <class... Args>
    using common_optional_t = typename common_optional<Args...>::type;

    template <class E>
    struct is_not_xoptional_nor_xmasked_value : std::negation<std::disjunction<is_xoptional<E>, is_xmasked_value<E>>>
    {
    };
}

#endif
