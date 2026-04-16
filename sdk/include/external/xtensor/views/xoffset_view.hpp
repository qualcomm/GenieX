/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_OFFSET_VIEW_HPP
#define XTENSOR_OFFSET_VIEW_HPP

#include <xtl/xcomplex.hpp>

#include "../views/xfunctor_view.hpp"

namespace xt
{
    namespace detail
    {
        template <class M, std::size_t I>
        struct offset_forwarder
        {
            using value_type = M;
            using reference = M&;
            using const_reference = const M&;
            using pointer = M*;
            using const_pointer = const M*;

            using proxy = xtl::xproxy_wrapper<M>;

            template <class value_type, class requested_type>
            using simd_return_type = xt_simd::simd_return_type<value_type, requested_type>;

            template <class T>
            decltype(auto) operator()(T&& t) const
            {
                return xtl::forward_offset<M, I>(std::forward<T>(t));
            }

            template <
                class align,
                class requested_type,
                std::size_t N,
                class E,
                class MF = M,
                class = std::enable_if_t<
                    (std::is_same<MF, double>::value || std::is_same<MF, float>::value) && I <= sizeof(MF),
                    int>>
            auto proxy_simd_load(const E& expr, std::size_t n) const
            {
                // TODO refactor using shuffle only
                auto batch = expr.template load_simd<align, requested_type, N>(n);
                if (I == 0)
                {
                    return batch.real();
                }
                else
                {
                    return batch.imag();
                }
            }

            template <
                class align,
                class simd,
                class E,
                class MF = M,
                class = std::enable_if_t<
                    (std::is_same<MF, double>::value || std::is_same<MF, float>::value) && I <= sizeof(MF),
                    int>>
            auto proxy_simd_store(E& expr, std::size_t n, const simd& batch) const
            {
                auto x = expr.template load_simd<align, double, simd::size>(n);
                if (I == 0)
                {
                    x.real() = batch;
                }
                else
                {
                    x.imag() = batch;
                }
                expr.template store_simd<align>(n, x);
            }
        };
    }

    template <class CT, class M, std::size_t I>
    using xoffset_view = xfunctor_view<detail::offset_forwarder<M, I>, CT>;

    template <class CT, class M, std::size_t I>
    using xoffset_adaptor = xfunctor_adaptor<detail::offset_forwarder<M, I>, CT>;
}

#endif
