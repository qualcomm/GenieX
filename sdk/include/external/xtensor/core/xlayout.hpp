/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_LAYOUT_HPP
#define XTENSOR_LAYOUT_HPP

#include <type_traits>

// Do not include anything else here.
// xlayout.hpp is included in xtensor_forward.hpp
// and we don't want to bring other headers to it.
#include "../core/xtensor_config.hpp"

namespace xt
{
    /*! layout_type enum for xcontainer based xexpressions */
    enum class layout_type
    {
        /*! dynamic layout_type: you can resize to row major, column major, or use custom strides */
        dynamic = 0x00,
        /*! layout_type compatible with all others */
        any = 0xFF,
        /*! row major layout_type */
        row_major = 0x01,
        /*! column major layout_type */
        column_major = 0x02
    };

    /**
     * Implementation of the following logical table:
     *
     *        | d | a | r | c |
     *      --+---+---+---+---+
     *      d | d | d | d | d |
     *      a | d | a | r | c |
     *      r | d | r | r | d |
     *      c | d | c | d | c |
     *      d = dynamic, a = any, r = row_major, c = column_major.
     *
     * Using bitmasks to avoid nested if-else statements.
     *
     * @param args the input layouts.
     * @return the output layout, computed with the previous logical table.
     */
    template <class... Args>
    constexpr layout_type compute_layout(Args... args) noexcept;

    constexpr layout_type default_assignable_layout(layout_type l) noexcept;

    constexpr layout_type layout_remove_any(const layout_type layout) noexcept;

    /******************
     * Implementation *
     ******************/

    namespace detail
    {
        constexpr layout_type compute_layout_impl() noexcept
        {
            return layout_type::any;
        }

        constexpr layout_type compute_layout_impl(layout_type l) noexcept
        {
            return l;
        }

        constexpr layout_type compute_layout_impl(layout_type lhs, layout_type rhs) noexcept
        {
            using type = std::underlying_type_t<layout_type>;
            return layout_type(static_cast<type>(lhs) & static_cast<type>(rhs));
        }

        template <class... Args>
        constexpr layout_type compute_layout_impl(layout_type lhs, Args... args) noexcept
        {
            return compute_layout_impl(lhs, compute_layout_impl(args...));
        }
    }

    template <class... Args>
    constexpr layout_type compute_layout(Args... args) noexcept
    {
        return detail::compute_layout_impl(args...);
    }

    constexpr layout_type default_assignable_layout(layout_type l) noexcept
    {
        return (l == layout_type::row_major || l == layout_type::column_major) ? l : XTENSOR_DEFAULT_LAYOUT;
    }

    constexpr layout_type layout_remove_any(const layout_type layout) noexcept
    {
        return layout == layout_type::any ? XTENSOR_DEFAULT_LAYOUT : layout;
    }
}

#endif
