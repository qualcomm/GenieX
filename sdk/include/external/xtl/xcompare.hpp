/***************************************************************************
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_COMPARE_HPP
#define XTL_COMPARE_HPP

#include <type_traits>

namespace xtl
{

    /**
     * @defgroup xtl_xcompare
     *
     * Compare the values of two integers t and u. Unlike builtin comparison operators,
     * negative signed integers always compare less than (and not equal to) unsigned integers.
     */

    namespace detail
    {
        template <class T, class U>
        struct same_signedness :
           std::integral_constant<bool, std::is_signed<T>::value == std::is_signed<U>::value>
        {
        };

        template <
            class T,
            class U,
            std::enable_if_t<same_signedness<T, U>::value, bool> = true
        >
        constexpr bool cmp_equal_impl(T t, U u) noexcept
        {
            return t == u;
        }

        template <
            class T,
            class U,
            std::enable_if_t<
                !same_signedness<T, U>::value && std::is_signed<T>::value,
                bool
            > = true
        >
        constexpr bool cmp_equal_impl(T t, U u) noexcept
        {
            using UT = std::make_unsigned_t<T>;
            return t < 0 ? false : static_cast<UT>(t) == u;
        }

        template <
            class T,
            class U,
            std::enable_if_t<
                !same_signedness<T, U>::value && !std::is_signed<T>::value,
                bool
            > = true
        >
        constexpr bool cmp_equal_impl(T t, U u) noexcept
        {
            using UU = std::make_unsigned_t<U>;
            return u < 0 ? false : t == static_cast<UU>(u);
        }
    }

    /**
     * ``true`` if @p t is equal to @p u.
     *
     * @ingroup xtl_xcompare
     */
    template <class T, class U>
    constexpr bool cmp_equal(T t, U u) noexcept
    {
        return detail::cmp_equal_impl(t, u);
    }

    /**
     * ``true`` if @p t is not equal to @p u.
     *
     * @ingroup xtl_xcompare
     */
    template <class T, class U>
    constexpr bool cmp_not_equal(T t, U u) noexcept
    {
        return !cmp_equal(t, u);
    }

    namespace detail
    {
        template <
            class T,
            class U,
            std::enable_if_t<detail::same_signedness<T, U>::value, bool> = true
        >
        constexpr bool cmp_less_impl(T t, U u) noexcept
        {
            return t < u;
        }

        template <
            class T,
            class U,
            std::enable_if_t<
                !detail::same_signedness<T, U>::value && std::is_signed<T>::value,
                bool
            > = true
        >
        constexpr bool cmp_less_impl(T t, U u) noexcept
        {
            using UT = std::make_unsigned_t<T>;
            return t < 0 ? true : static_cast<UT>(t) < u;
        }

        template <
            class T,
            class U,
            std::enable_if_t<
                !detail::same_signedness<T, U>::value && !std::is_signed<T>::value,
                bool
            > = true
        >
        constexpr bool cmp_less_impl(T t, U u) noexcept
        {
            using UU = std::make_unsigned_t<U>;
            return u < 0 ? false : t < static_cast<UU>(u);
        }
    }

    /**
     * ``true`` if @p t is striclty less than @p u.
     *
     * @ingroup xtl_xcompare
     */
    template <class T, class U>
    constexpr bool cmp_less(T t, U u) noexcept
    {
        return detail::cmp_less_impl(t, u);
    }

    /**
     * ``true`` if @p t is striclty greater than @p u.
     *
     * @ingroup xtl_xcompare
     */
    template <class T, class U>
    constexpr bool cmp_greater(T t, U u) noexcept
    {
        return cmp_less(u, t);
    }

    /**
     * ``true`` if @p t is less or equal to @p u.
     *
     * @ingroup xtl_xcompare
     */
    template <class T, class U>
    constexpr bool cmp_less_equal(T t, U u) noexcept
    {
        return !cmp_greater(t, u);
    }

    /**
     * ``true`` if @p t is greater or equal to @p u.
     *
     * @ingroup xtl_xcompare
     */
    template <class T, class U>
    constexpr bool cmp_greater_equal(T t, U u) noexcept
    {
        return !cmp_less(t, u);
    }
}

#endif
