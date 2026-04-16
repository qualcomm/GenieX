/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_JSON_HPP
#define XTENSOR_JSON_HPP

#include <cstddef>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "../core/xtensor_config.hpp"
#include "../views/xstrided_view.hpp"

namespace xt
{
    /*************************************
     * to_json and from_json declaration *
     *************************************/

    template <template <typename U, typename V, typename... Args> class M, class E>
    enable_xexpression<E> to_json(nlohmann::basic_json<M>&, const E&);

    template <template <typename U, typename V, typename... Args> class M, class E>
    enable_xcontainer_semantics<E> from_json(const nlohmann::basic_json<M>&, E&);

    /// @cond DOXYGEN_INCLUDE_SFINAE
    template <template <typename U, typename V, typename... Args> class M, class E>
    enable_xview_semantics<E> from_json(const nlohmann::basic_json<M>&, E&);

    /// @endcond

    /****************************************
     * to_json and from_json implementation *
     ****************************************/

    namespace detail
    {
        template <template <typename U, typename V, typename... Args> class M, class D>
        void to_json_impl(nlohmann::basic_json<M>& j, const xexpression<D>& e, xstrided_slice_vector& slices)
        {
            const auto view = strided_view(e.derived_cast(), slices);
            if (view.dimension() == 0)
            {
                j = view();
            }
            else
            {
                j = nlohmann::basic_json<M>::array();
                using size_type = typename D::size_type;
                size_type nrows = view.shape()[0];
                for (size_type i = 0; i != nrows; ++i)
                {
                    slices.push_back(static_cast<std::ptrdiff_t>(i));
                    nlohmann::basic_json<M> k;
                    to_json_impl(k, e, slices);
                    j.push_back(std::move(k));
                    slices.pop_back();
                }
            }
        }

        template <template <typename U, typename V, typename... Args> class M, class D>
        inline void
        from_json_impl(const nlohmann::basic_json<M>& j, xexpression<D>& e, xstrided_slice_vector& slices)
        {
            auto view = strided_view(e.derived_cast(), slices);

            if (view.dimension() == 0)
            {
                view() = j.template get<std::remove_reference_t<decltype(view())>>();
            }
            else
            {
                using size_type = typename D::size_type;
                size_type nrows = view.shape()[0];
                for (size_type i = 0; i != nrows; ++i)
                {
                    slices.push_back(static_cast<std::ptrdiff_t>(i));
                    const nlohmann::basic_json<M>& k = j[i];
                    from_json_impl(k, e, slices);
                    slices.pop_back();
                }
            }
        }

        template <template <typename U, typename V, typename... Args> class M>
        inline unsigned int json_dimension(const nlohmann::basic_json<M>& j)
        {
            if (j.is_array() && j.size())
            {
                return 1 + json_dimension(j[0]);
            }
            else
            {
                return 0;
            }
        }

        template <template <typename U, typename V, typename... Args> class M, class S>
        inline void json_shape(const nlohmann::basic_json<M>& j, S& s, std::size_t pos = 0)
        {
            if (j.is_array())
            {
                auto size = j.size();
                s[pos] = size;
                if (size)
                {
                    json_shape(j[0], s, pos + 1);
                }
            }
        }
    }

    /**
     * @brief JSON serialization of an xtensor expression.
     *
     * The to_json method is used by the nlohmann_json package for automatic
     * serialization of user-defined types. The method is picked up by
     * argument-dependent lookup.
     *
     * @param j a JSON object
     * @param e a const \ref xexpression
     */
    template <template <typename U, typename V, typename... Args> class M, class E>
    inline enable_xexpression<E> to_json(nlohmann::basic_json<M>& j, const E& e)
    {
        auto sv = xstrided_slice_vector();
        detail::to_json_impl(j, e, sv);
    }

    /**
     * @brief JSON deserialization of a xtensor expression with a container or
     * a view semantics.
     *
     * The from_json method is used by the nlohmann_json library for automatic
     * serialization of user-defined types. The method is picked up by
     * argument-dependent lookup.
     *
     * Note: for converting a JSON object to a value, nlohmann_json requires
     * the value type to be default constructible, which is typically not the
     * case for expressions with a view semantics. In this case, from_json can
     * be called directly.
     *
     * @param j a const JSON object
     * @param e an \ref xexpression
     */
    template <template <typename U, typename V, typename... Args> class M, class E>
    inline enable_xcontainer_semantics<E> from_json(const nlohmann::basic_json<M>& j, E& e)
    {
        auto dimension = detail::json_dimension(j);
        auto s = xtl::make_sequence<typename E::shape_type>(dimension);
        detail::json_shape(j, s);

        // In the case of a container, we resize the container.
        e.resize(s);

        auto sv = xstrided_slice_vector();
        detail::from_json_impl(j, e, sv);
    }

    /// @cond DOXYGEN_INCLUDE_SFINAE
    template <template <typename U, typename V, typename... Args> class M, class E>
    inline enable_xview_semantics<E> from_json(const nlohmann::basic_json<M>& j, E& e)
    {
        typename E::shape_type s;
        detail::json_shape(j, s);

        // In the case of a view, we check the size of the container.
        if (!std::equal(s.cbegin(), s.cend(), e.shape().cbegin()))
        {
            XTENSOR_THROW(std::runtime_error, "Shape mismatch when deserializing JSON to view");
        }

        auto sv = xstrided_slice_vector();
        detail::from_json_impl(j, e, sv);
    }

    /// @endcond
}

#endif
