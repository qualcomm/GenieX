/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_XSET_OPERATION_HPP
#define XTENSOR_XSET_OPERATION_HPP

#include <algorithm>
#include <functional>
#include <type_traits>

#include <xtl/xsequence.hpp>

#include "../containers/xscalar.hpp"
#include "../core/xfunction.hpp"
#include "../core/xmath.hpp"
#include "../core/xstrides.hpp"
#include "../utils/xutils.hpp"
#include "../views/xstrided_view.hpp"

namespace xt
{

    namespace detail
    {

        template <bool lvalue>
        struct lambda_isin
        {
            template <class E>
            static auto make(E&& e)
            {
                return [&e](const auto& t)
                {
                    return std::find(e.begin(), e.end(), t) != e.end();
                };
            }
        };

        template <>
        struct lambda_isin<false>
        {
            template <class E>
            static auto make(E&& e)
            {
                return [e](const auto& t)
                {
                    return std::find(e.begin(), e.end(), t) != e.end();
                };
            }
        };

    }

    /**
     * @ingroup logical_operators
     * @brief isin
     *
     * Returns a boolean array of the same shape as ``element`` that is ``true`` where an element of
     * ``element`` is in ``test_elements`` and ``False`` otherwise.
     * @param element an \ref xexpression
     * @param test_elements an array
     * @return a boolean array
     */
    template <class E, class T>
    inline auto isin(E&& element, std::initializer_list<T> test_elements) noexcept
    {
        auto lambda = [test_elements](const auto& t)
        {
            return std::find(test_elements.begin(), test_elements.end(), t) != test_elements.end();
        };
        return make_lambda_xfunction(std::move(lambda), std::forward<E>(element));
    }

    /**
     * @ingroup logical_operators
     * @brief isin
     *
     * Returns a boolean array of the same shape as ``element`` that is ``true`` where an element of
     * ``element`` is in ``test_elements`` and ``False`` otherwise.
     * @param element an \ref xexpression
     * @param test_elements an array
     * @return a boolean array
     */
    template <class E, class F>
    inline auto isin(E&& element, F&& test_elements) noexcept
        requires(has_iterator_interface_concept<F>)
    {
        auto lambda = detail::lambda_isin<std::is_lvalue_reference<F>::value>::make(std::forward<F>(test_elements
        ));
        return make_lambda_xfunction(std::move(lambda), std::forward<E>(element));
    }

    /**
     * @ingroup logical_operators
     * @brief isin
     *
     * Returns a boolean array of the same shape as ``element`` that is ``true`` where an element of
     * ``element`` is in ``test_elements`` and ``False`` otherwise.
     * @param element an \ref xexpression
     * @param test_elements_begin iterator to the beginning of an array
     * @param test_elements_end iterator to the end of an array
     * @return a boolean array
     */
    template <class E, iterator_concept I>
    inline auto isin(E&& element, I&& test_elements_begin, I&& test_elements_end) noexcept
    {
        auto lambda = [&test_elements_begin, &test_elements_end](const auto& t)
        {
            return std::find(test_elements_begin, test_elements_end, t) != test_elements_end;
        };
        return make_lambda_xfunction(std::move(lambda), std::forward<E>(element));
    }

    /**
     * @ingroup logical_operators
     * @brief in1d
     *
     * Returns a boolean array of the same shape as ``element`` that is ``true`` where an element of
     * ``element`` is in ``test_elements`` and ``False`` otherwise.
     * @param element an \ref xexpression
     * @param test_elements an array
     * @return a boolean array
     */
    template <class E, class T>
    inline auto in1d(E&& element, std::initializer_list<T> test_elements) noexcept
    {
        XTENSOR_ASSERT(element.dimension() == 1ul);
        return isin(std::forward<E>(element), std::forward<std::initializer_list<T>>(test_elements));
    }

    /**
     * @ingroup logical_operators
     * @brief in1d
     *
     * Returns a boolean array of the same shape as ``element`` that is ``true`` where an element of
     * ``element`` is in ``test_elements`` and ``False`` otherwise.
     * @param element an \ref xexpression
     * @param test_elements an array
     * @return a boolean array
     */
    template <class E, class F>
    inline auto in1d(E&& element, F&& test_elements) noexcept
        requires(has_iterator_interface_concept<F>)
    {
        XTENSOR_ASSERT(element.dimension() == 1ul);
        XTENSOR_ASSERT(test_elements.dimension() == 1ul);
        return isin(std::forward<E>(element), std::forward<F>(test_elements));
    }

    /**
     * @ingroup logical_operators
     * @brief in1d
     *
     * Returns a boolean array of the same shape as ``element`` that is ``true`` where an element of
     * ``element`` is in ``test_elements`` and ``False`` otherwise.
     * @param element an \ref xexpression
     * @param test_elements_begin iterator to the beginning of an array
     * @param test_elements_end iterator to the end of an array
     * @return a boolean array
     */
    template <class E, iterator_concept I>
    inline auto in1d(E&& element, I&& test_elements_begin, I&& test_elements_end) noexcept
    {
        XTENSOR_ASSERT(element.dimension() == 1ul);
        return isin(
            std::forward<E>(element),
            std::forward<I>(test_elements_begin),
            std::forward<I>(test_elements_end)
        );
    }

    /**
     * @ingroup searchsorted
     * @brief Find indices where elements should be inserted to maintain order.
     *
     * @param a Input array: sorted (array_like).
     * @param v Values to insert into a (array_like).
     * @param right If ``false``, the index of the first suitable location found is given.
     * @return Array of insertion points with the same shape as v.
     */
    template <class E1, class E2>
    inline auto searchsorted(E1&& a, E2&& v, bool right = true)
    {
        XTENSOR_ASSERT(std::is_sorted(a.cbegin(), a.cend()));

        auto out = xt::empty<size_t>(v.shape());

        if (right)
        {
            for (size_t i = 0; i < v.size(); ++i)
            {
                out(i) = static_cast<std::size_t>(std::lower_bound(a.cbegin(), a.cend(), v(i)) - a.cbegin());
            }
        }
        else
        {
            for (size_t i = 0; i < v.size(); ++i)
            {
                out(i) = static_cast<std::size_t>(std::upper_bound(a.cbegin(), a.cend(), v(i)) - a.cbegin());
            }
        }


        return out;
    }

}

#endif
