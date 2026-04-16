/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_EVAL_HPP
#define XTENSOR_EVAL_HPP

#include "../core/xexpression_traits.hpp"
#include "../core/xshape.hpp"
#include "../core/xtensor_forward.hpp"

namespace xt
{

    /**
     * @defgroup xt_xeval
     *
     * Evaluation functions.
     * Defined in ``xtensor/xeval.hpp``
     */

    namespace detail
    {
        template <class T>
        using is_container = std::is_base_of<xcontainer<std::remove_const_t<T>>, T>;
    }

    /**
     * Force evaluation of xexpression.
     *
     * @code{.cpp}
     * xt::xarray<double> a = {1, 2, 3, 4};
     * auto&& b = xt::eval(a); // b is a reference to a, no copy!
     * auto&& c = xt::eval(a + b); // c is xarray<double>, not an xexpression
     * @endcode
     *
     * @ingroup xt_xeval
     * @return xt::xarray or xt::xtensor depending on shape type
     */
    template <class T>
    inline auto eval(T&& t) -> std::enable_if_t<detail::is_container<std::decay_t<T>>::value, T&&>
    {
        return std::forward<T>(t);
    }

    /// @cond DOXYGEN_INCLUDE_SFINAE
    template <class T>
    inline auto eval(T&& t)
        -> std::enable_if_t<!detail::is_container<std::decay_t<T>>::value, temporary_type_t<T>>
    {
        return std::forward<T>(t);
    }

    /// @endcond

    namespace detail
    {
        /**********************************
         * has_same_layout implementation *
         **********************************/

        template <layout_type L = layout_type::any, class E>
        constexpr bool has_same_layout()
        {
            return (std::decay_t<E>::static_layout == L) || (L == layout_type::any);
        }

        template <layout_type L = layout_type::any, class E>
        constexpr bool has_same_layout(E&&)
        {
            return has_same_layout<L, E>();
        }

        template <class E1, class E2>
        constexpr bool has_same_layout(E1&&, E2&&)
        {
            return has_same_layout<std::decay_t<E1>::static_layout, E2>();
        }

        /*********************************
         * has_fixed_dims implementation *
         *********************************/

        template <class E>
        constexpr bool has_fixed_dims()
        {
            return detail::is_array<typename std::decay_t<E>::shape_type>::value;
        }

        template <class E>
        constexpr bool has_fixed_dims(E&&)
        {
            return has_fixed_dims<E>();
        }

        /****************************************
         * as_xarray_container_t implementation *
         ****************************************/

        template <class E, layout_type L>
        using as_xarray_container_t = xarray<typename std::decay_t<E>::value_type, layout_remove_any(L)>;

        /*****************************************
         * as_xtensor_container_t implementation *
         *****************************************/

        template <class E, layout_type L>
        using as_xtensor_container_t = xtensor<
            typename std::decay_t<E>::value_type,
            std::tuple_size<typename std::decay_t<E>::shape_type>::value,
            layout_remove_any(L)>;
    }

    /**
     * Force evaluation of xexpression not providing a data interface
     * and convert to the required layout.
     *
     * @code{.cpp}
     * xt::xarray<double, xt::layout_type::row_major> a = {1, 2, 3, 4};
     *
     * // take reference to a (no copy!)
     * auto&& b = xt::as_strided(a);
     *
     * // xarray<double> with the required layout
     * auto&& c = xt::as_strided<xt::layout_type::column_major>(a);
     *
     * // xexpression
     * auto&& a_cast = xt::cast<int>(a);
     *
     * // xarray<int>, not an xexpression
     * auto&& d = xt::as_strided(a_cast);
     *
     * // xarray<int> with the required layout
     * auto&& e = xt::as_strided<xt::layout_type::column_major>(a_cast);
     * @endcode
     *
     * @warning This function should be used in a local context only.
     *          Returning the value returned by this function could lead to a dangling reference.
     * @ingroup xt_xeval
     * @return The expression when it already provides a data interface with the correct layout,
     *         an evaluated xt::xarray or xt::xtensor depending on shape type otherwise.
     */
    template <layout_type L = layout_type::any, class E>
    inline auto as_strided(E&& e)
        -> std::enable_if_t<has_data_interface<std::decay_t<E>>::value && detail::has_same_layout<L, E>(), E&&>
    {
        return std::forward<E>(e);
    }

    /// @cond DOXYGEN_INCLUDE_SFINAE
    template <layout_type L = layout_type::any, class E>
    inline auto as_strided(E&& e) -> std::enable_if_t<
        (!(has_data_interface<std::decay_t<E>>::value && detail::has_same_layout<L, E>()))
            && detail::has_fixed_dims<E>(),
        detail::as_xtensor_container_t<E, L>>
    {
        return e;
    }

    template <layout_type L = layout_type::any, class E>
    inline auto as_strided(E&& e) -> std::enable_if_t<
        (!(has_data_interface<std::decay_t<E>>::value && detail::has_same_layout<L, E>()))
            && (!detail::has_fixed_dims<E>()),
        detail::as_xarray_container_t<E, L>>
    {
        return e;
    }

    /// @endcond
}

#endif
