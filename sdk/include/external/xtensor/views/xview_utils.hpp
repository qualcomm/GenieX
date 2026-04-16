/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_VIEW_UTILS_HPP
#define XTENSOR_VIEW_UTILS_HPP

#include <array>

#include "../core/xlayout.hpp"
#include "../core/xtensor_forward.hpp"
#include "../views/xslice.hpp"

namespace xt
{

    /********************************
     * helper functions declaration *
     ********************************/

    // number of integral types in the specified sequence of types
    template <class... S>
    constexpr std::size_t integral_count();

    // number of integral types in the specified sequence of types before specified index
    template <class... S>
    constexpr std::size_t integral_count_before(std::size_t i);

    // index in the specified sequence of types of the ith non-integral type
    template <class... S>
    constexpr std::size_t integral_skip(std::size_t i);

    // number of newaxis types in the specified sequence of types
    template <class... S>
    constexpr std::size_t newaxis_count();

    // number of newaxis types in the specified sequence of types before specified index
    template <class... S>
    constexpr std::size_t newaxis_count_before(std::size_t i);

    // index in the specified sequence of types of the ith non-newaxis type
    template <class... S>
    constexpr std::size_t newaxis_skip(std::size_t i);

    template <class S, class It>
    inline disable_xslice<S, std::size_t> get_slice_value(const S& s, It&) noexcept
    {
        return static_cast<std::size_t>(s);
    }

    template <class S, class It>
    inline auto get_slice_value(const xslice<S>& slice, It& it) noexcept
    {
        return slice.derived_cast()(typename S::size_type(*it));
    }

    /***********************
     * view_temporary_type *
     ***********************/

    namespace detail
    {
        template <class T, class S, layout_type L, class... SL>
        struct view_temporary_type_impl
        {
            using type = xt::xarray<T, L>;
        };

        template <class T, class I, std::size_t N, layout_type L, class... SL>
        struct view_temporary_type_impl<T, std::array<I, N>, L, SL...>
        {
            using type = xt::xtensor<T, N + newaxis_count<SL...>() - integral_count<SL...>(), L>;
        };
    }

    template <class E, class... SL>
    struct view_temporary_type
    {
        using type = typename detail::view_temporary_type_impl<
            std::decay_t<typename E::value_type>,
            typename E::shape_type,
            E::static_layout,
            SL...>::type;
    };

    template <class E, class... SL>
    using view_temporary_type_t = typename view_temporary_type<E, SL...>::type;

    /************************
     * count integral types *
     ************************/

    namespace detail
    {

        template <class T, class... S>
        struct integral_count_impl
        {
            static constexpr std::size_t count(std::size_t i) noexcept
            {
                return i
                           ? (integral_count_impl<S...>::count(i - 1)
                              + (xtl::is_integral<std::remove_reference_t<T>>::value ? 1 : 0))
                           : 0;
            }
        };

        template <>
        struct integral_count_impl<void>
        {
            static constexpr std::size_t count(std::size_t /*i*/) noexcept
            {
                return 0;
            }
        };
    }

    template <class... S>
    constexpr std::size_t integral_count()
    {
        return detail::integral_count_impl<S..., void>::count(sizeof...(S));
    }

    template <class... S>
    constexpr std::size_t integral_count_before(std::size_t i)
    {
        return detail::integral_count_impl<S..., void>::count(i);
    }

    /***********************
     * count newaxis types *
     ***********************/

    namespace detail
    {
        template <class T>
        struct is_newaxis : std::false_type
        {
        };

        template <class T>
        struct is_newaxis<xnewaxis<T>> : public std::true_type
        {
        };

        template <class T, class... S>
        struct newaxis_count_impl
        {
            static constexpr std::size_t count(std::size_t i) noexcept
            {
                return i
                           ? (newaxis_count_impl<S...>::count(i - 1)
                              + (is_newaxis<std::remove_reference_t<T>>::value ? 1 : 0))
                           : 0;
            }
        };

        template <>
        struct newaxis_count_impl<void>
        {
            static constexpr std::size_t count(std::size_t /*i*/) noexcept
            {
                return 0;
            }
        };
    }

    template <class... S>
    constexpr std::size_t newaxis_count()
    {
        return detail::newaxis_count_impl<S..., void>::count(sizeof...(S));
    }

    template <class... S>
    constexpr std::size_t newaxis_count_before(std::size_t i)
    {
        return detail::newaxis_count_impl<S..., void>::count(i);
    }

    /**********************************
     * index of ith non-integral type *
     **********************************/

    namespace detail
    {

        template <class T, class... S>
        struct integral_skip_impl
        {
            static constexpr std::size_t count(std::size_t i) noexcept
            {
                return i == 0 ? count_impl() : count_impl(i);
            }

        private:

            static constexpr std::size_t count_impl(std::size_t i) noexcept
            {
                return 1
                       + (xtl::is_integral<std::remove_reference_t<T>>::value
                              ? integral_skip_impl<S...>::count(i)
                              : integral_skip_impl<S...>::count(i - 1));
            }

            static constexpr std::size_t count_impl() noexcept
            {
                return xtl::is_integral<std::remove_reference_t<T>>::value
                           ? 1 + integral_skip_impl<S...>::count(0)
                           : 0;
            }
        };

        template <>
        struct integral_skip_impl<void>
        {
            static constexpr std::size_t count(std::size_t i) noexcept
            {
                return i;
            }
        };
    }

    template <class... S>
    constexpr std::size_t integral_skip(std::size_t i)
    {
        return detail::integral_skip_impl<S..., void>::count(i);
    }

    /*********************************
     * index of ith non-newaxis type *
     *********************************/

    namespace detail
    {

        template <class T, class... S>
        struct newaxis_skip_impl
        {
            static constexpr std::size_t count(std::size_t i) noexcept
            {
                return i == 0 ? count_impl() : count_impl(i);
            }

        private:

            static constexpr std::size_t count_impl(std::size_t i) noexcept
            {
                return 1
                       + (is_newaxis<std::remove_reference_t<T>>::value
                              ? newaxis_skip_impl<S...>::count(i)
                              : newaxis_skip_impl<S...>::count(i - 1));
            }

            static constexpr std::size_t count_impl() noexcept
            {
                return is_newaxis<std::remove_reference_t<T>>::value ? 1 + newaxis_skip_impl<S...>::count(0)
                                                                     : 0;
            }
        };

        template <>
        struct newaxis_skip_impl<void>
        {
            static constexpr std::size_t count(std::size_t i) noexcept
            {
                return i;
            }
        };
    }

    template <class... S>
    constexpr std::size_t newaxis_skip(std::size_t i)
    {
        return detail::newaxis_skip_impl<S..., void>::count(i);
    }
}

#endif
