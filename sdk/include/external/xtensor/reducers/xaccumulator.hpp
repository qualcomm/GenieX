/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_ACCUMULATOR_HPP
#define XTENSOR_ACCUMULATOR_HPP

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <type_traits>

#include "../core/xexpression.hpp"
#include "../core/xstrides.hpp"
#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"

namespace xt
{

#define DEFAULT_STRATEGY_ACCUMULATORS evaluation_strategy::immediate_type

    namespace detail
    {
        template <class V = void>
        struct accumulator_identity : xtl::identity
        {
            using value_type = V;
        };
    }

    /**************
     * accumulate *
     **************/

    template <class ACCUMULATE_FUNC, class INIT_FUNC = detail::accumulator_identity<void>>
    struct xaccumulator_functor : public std::tuple<ACCUMULATE_FUNC, INIT_FUNC>
    {
        using self_type = xaccumulator_functor<ACCUMULATE_FUNC, INIT_FUNC>;
        using base_type = std::tuple<ACCUMULATE_FUNC, INIT_FUNC>;
        using accumulate_functor_type = ACCUMULATE_FUNC;
        using init_functor_type = INIT_FUNC;
        using init_value_type = typename init_functor_type::value_type;

        xaccumulator_functor()
            : base_type()
        {
        }

        template <class RF>
        xaccumulator_functor(RF&& accumulate_func)
            : base_type(std::forward<RF>(accumulate_func), INIT_FUNC())
        {
        }

        template <class RF, class IF>
        xaccumulator_functor(RF&& accumulate_func, IF&& init_func)
            : base_type(std::forward<RF>(accumulate_func), std::forward<IF>(init_func))
        {
        }
    };

    template <class RF>
    auto make_xaccumulator_functor(RF&& accumulate_func)
    {
        using accumulator_type = xaccumulator_functor<std::remove_reference_t<RF>>;
        return accumulator_type(std::forward<RF>(accumulate_func));
    }

    template <class RF, class IF>
    auto make_xaccumulator_functor(RF&& accumulate_func, IF&& init_func)
    {
        using accumulator_type = xaccumulator_functor<std::remove_reference_t<RF>, std::remove_reference_t<IF>>;
        return accumulator_type(std::forward<RF>(accumulate_func), std::forward<IF>(init_func));
    }

    namespace detail
    {
        template <class F, class E, class EVS>
        xarray<typename std::decay_t<E>::value_type> accumulator_impl(F&&, E&&, std::size_t, EVS)
        {
            static_assert(
                !std::is_same<evaluation_strategy::lazy_type, EVS>::value,
                "Lazy accumulators not yet implemented."
            );
        }

        template <class F, class E, class EVS>
        xarray<typename std::decay_t<E>::value_type> accumulator_impl(F&&, E&&, EVS)
        {
            static_assert(
                !std::is_same<evaluation_strategy::lazy_type, EVS>::value,
                "Lazy accumulators not yet implemented."
            );
        }

        template <class T, class R>
        struct xaccumulator_return_type
        {
            using type = xarray<R>;
        };

        template <class T, layout_type L, class R>
        struct xaccumulator_return_type<xarray<T, L>, R>
        {
            using type = xarray<R, L>;
        };

        template <class T, std::size_t N, layout_type L, class R>
        struct xaccumulator_return_type<xtensor<T, N, L>, R>
        {
            using type = xtensor<R, N, L>;
        };

        template <class T, std::size_t... I, layout_type L, class R>
        struct xaccumulator_return_type<xtensor_fixed<T, xshape<I...>, L>, R>
        {
            using type = xtensor_fixed<R, xshape<I...>, L>;
        };

        template <class T, class R>
        using xaccumulator_return_type_t = typename xaccumulator_return_type<T, R>::type;

        template <class T>
        struct fixed_compute_size;

        template <class T, class R>
        struct xaccumulator_linear_return_type
        {
            using type = xtensor<R, 1>;
        };

        template <class T, layout_type L, class R>
        struct xaccumulator_linear_return_type<xarray<T, L>, R>
        {
            using type = xtensor<R, 1, L>;
        };

        template <class T, std::size_t N, layout_type L, class R>
        struct xaccumulator_linear_return_type<xtensor<T, N, L>, R>
        {
            using type = xtensor<R, 1, L>;
        };

        template <class T, std::size_t... I, layout_type L, class R>
        struct xaccumulator_linear_return_type<xtensor_fixed<T, xshape<I...>, L>, R>
        {
            using type = xtensor_fixed<R, xshape<fixed_compute_size<xshape<I...>>::value>, L>;
        };

        template <class T, class R>
        using xaccumulator_linear_return_type_t = typename xaccumulator_linear_return_type<T, R>::type;

        template <class F, class E>
        inline auto accumulator_init_with_f(F&& f, E& e, std::size_t axis)
        {
            // this function is the equivalent (but hopefully faster) to (if axis == 1)
            // e[:, 0, :, :, ...] = f(e[:, 0, :, :, ...])
            // so that all "first" values are initialized in a first pass

            std::size_t outer_loop_size, inner_loop_size, pos = 0;
            std::size_t outer_stride, inner_stride;

            auto set_loop_sizes = [&outer_loop_size, &inner_loop_size](auto first, auto last, std::ptrdiff_t ax)
            {
                outer_loop_size = std::accumulate(
                    first,
                    first + ax,
                    std::size_t(1),
                    std::multiplies<std::size_t>()
                );
                inner_loop_size = std::accumulate(
                    first + ax + 1,
                    last,
                    std::size_t(1),
                    std::multiplies<std::size_t>()
                );
            };

            // Note: add check that strides > 0
            auto set_loop_strides = [&outer_stride, &inner_stride](auto first, auto last, std::ptrdiff_t ax)
            {
                outer_stride = static_cast<std::size_t>(ax == 0 ? 1 : *std::min_element(first, first + ax));
                inner_stride = static_cast<std::size_t>(
                    (ax == std::distance(first, last) - 1) ? 1 : *std::min_element(first + ax + 1, last)
                );
            };

            set_loop_sizes(e.shape().begin(), e.shape().end(), static_cast<std::ptrdiff_t>(axis));
            set_loop_strides(e.strides().begin(), e.strides().end(), static_cast<std::ptrdiff_t>(axis));

            if (e.layout() == layout_type::column_major)
            {
                // swap for better memory locality (smaller stride in the inner loop)
                std::swap(outer_loop_size, inner_loop_size);
                std::swap(outer_stride, inner_stride);
            }

            for (std::size_t i = 0; i < outer_loop_size; ++i)
            {
                pos = i * outer_stride;
                for (std::size_t j = 0; j < inner_loop_size; ++j)
                {
                    e.storage()[pos] = f(e.storage()[pos]);
                    pos += inner_stride;
                }
            }
        }

        template <class F, class E>
        inline auto accumulator_impl(F&& f, E&& e, std::size_t axis, evaluation_strategy::immediate_type)
        {
            using init_type = typename F::init_value_type;
            using accumulate_functor_type = typename F::accumulate_functor_type;
            using expr_value_type = typename std::decay_t<E>::value_type;
            // using return_type = std::conditional_t<std::is_same<init_type, void>::value, typename
            // std::decay_t<E>::value_type, init_type>;

            using return_type = std::decay_t<decltype(std::declval<accumulate_functor_type>()(
                std::declval<init_type>(),
                std::declval<expr_value_type>()
            ))>;

            using result_type = xaccumulator_return_type_t<std::decay_t<E>, return_type>;

            if (axis >= e.dimension())
            {
                XTENSOR_THROW(std::runtime_error, "Axis larger than expression dimension in accumulator.");
            }

            result_type res = e;  // assign + make a copy, we need it anyways

            if (res.shape(axis) != std::size_t(0))
            {
                std::size_t inner_stride = static_cast<std::size_t>(res.strides()[axis]);
                std::size_t outer_stride = 1;  // either row- or column-wise (strides.back / strides.front)
                std::size_t outer_loop_size = 0;
                std::size_t inner_loop_size = 0;
                std::size_t init_size = e.shape()[axis] != std::size_t(1) ? std::size_t(1) : std::size_t(0);

                auto set_loop_sizes =
                    [&outer_loop_size, &inner_loop_size, init_size](auto first, auto last, std::ptrdiff_t ax)
                {
                    outer_loop_size = std::accumulate(first, first + ax, init_size, std::multiplies<std::size_t>());

                    inner_loop_size = std::accumulate(
                        first + ax,
                        last,
                        std::size_t(1),
                        std::multiplies<std::size_t>()
                    );
                };

                if (result_type::static_layout == layout_type::row_major)
                {
                    set_loop_sizes(res.shape().cbegin(), res.shape().cend(), static_cast<std::ptrdiff_t>(axis));
                }
                else
                {
                    set_loop_sizes(res.shape().cbegin(), res.shape().cend(), static_cast<std::ptrdiff_t>(axis + 1));
                    std::swap(inner_loop_size, outer_loop_size);
                }

                std::size_t pos = 0;

                inner_loop_size = inner_loop_size - inner_stride;

                // activate the init loop if we have an init function other than identity
                if (!std::is_same<
                        std::decay_t<typename F::init_functor_type>,
                        typename detail::accumulator_identity<init_type>>::value)
                {
                    accumulator_init_with_f(xt::get<1>(f), res, axis);
                }

                pos = 0;
                for (std::size_t i = 0; i < outer_loop_size; ++i)
                {
                    for (std::size_t j = 0; j < inner_loop_size; ++j)
                    {
                        res.storage()[pos + inner_stride] = xt::get<0>(f)(
                            res.storage()[pos],
                            res.storage()[pos + inner_stride]
                        );

                        pos += outer_stride;
                    }
                    pos += inner_stride;
                }
            }
            return res;
        }

        template <class F, class E>
        inline auto accumulator_impl(F&& f, E&& e, evaluation_strategy::immediate_type)
        {
            using init_type = typename F::init_value_type;
            using expr_value_type = typename std::decay_t<E>::value_type;
            using accumulate_functor_type = typename F::accumulate_functor_type;
            using return_type = std::decay_t<decltype(std::declval<accumulate_functor_type>()(
                std::declval<init_type>(),
                std::declval<expr_value_type>()
            ))>;
            // using return_type = std::conditional_t<std::is_same<init_type, void>::value, typename
            // std::decay_t<E>::value_type, init_type>;

            using result_type = xaccumulator_return_type_t<std::decay_t<E>, return_type>;

            std::size_t sz = e.size();
            auto result = result_type::from_shape({sz});

            if (sz != std::size_t(0))
            {
                auto it = e.template begin<XTENSOR_DEFAULT_TRAVERSAL>();
                result.storage()[0] = xt::get<1>(f)(*it);
                ++it;

                for (std::size_t idx = 0; it != e.template end<XTENSOR_DEFAULT_TRAVERSAL>(); ++it)
                {
                    result.storage()[idx + 1] = xt::get<0>(f)(result.storage()[idx], *it);
                    ++idx;
                }
            }
            return result;
        }
    }

    /**
     * Accumulate and flatten array
     * **NOTE** This function is not lazy!
     *
     * @param f functor to use for accumulation
     * @param e xexpression to be accumulated
     * @param evaluation_strategy evaluation strategy of the accumulation
     *
     * @return returns xarray<T> filled with accumulated values
     */
    template <class F, class E, class EVS = DEFAULT_STRATEGY_ACCUMULATORS, XTL_REQUIRES(is_evaluation_strategy<EVS>)>
    inline auto accumulate(F&& f, E&& e, EVS evaluation_strategy = EVS())
    {
        // Note we need to check is_integral above in order to prohibit EVS = int, and not taking the
        // std::size_t overload below!
        return detail::accumulator_impl(std::forward<F>(f), std::forward<E>(e), evaluation_strategy);
    }

    /**
     * Accumulate over axis
     * **NOTE** This function is not lazy!
     *
     * @param f Functor to use for accumulation
     * @param e xexpression to accumulate
     * @param axis Axis to perform accumulation over
     * @param evaluation_strategy evaluation strategy of the accumulation
     *
     * @return returns xarray<T> filled with accumulated values
     */
    template <class F, class E, class EVS = DEFAULT_STRATEGY_ACCUMULATORS>
    inline auto accumulate(F&& f, E&& e, std::ptrdiff_t axis, EVS evaluation_strategy = EVS())
    {
        std::size_t ax = normalize_axis(e.dimension(), axis);
        return detail::accumulator_impl(std::forward<F>(f), std::forward<E>(e), ax, evaluation_strategy);
    }
}

#endif
