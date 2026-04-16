/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_SORT_HPP
#define XTENSOR_SORT_HPP

#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>

#include <xtl/xcompare.hpp>

#include "../containers/xadapt.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xeval.hpp"
#include "../core/xmath.hpp"
#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../misc/xmanipulation.hpp"
#include "../views/xindex_view.hpp"
#include "../views/xslice.hpp"  // for xnone
#include "../views/xview.hpp"

namespace xt
{
    /**
     * @defgroup xt_xsort Sorting functions.
     *
     * Because sorting functions need to access the tensor data repeatedly, they evaluate their
     * input and may allocate temporaries.
     */

    namespace detail
    {
        template <class T>
        std::ptrdiff_t adjust_secondary_stride(std::ptrdiff_t stride, T shape)
        {
            return stride != 0 ? stride : static_cast<std::ptrdiff_t>(shape);
        }

        template <class E>
        inline std::ptrdiff_t get_secondary_stride(const E& ev)
        {
            if (ev.layout() == layout_type::row_major)
            {
                return adjust_secondary_stride(ev.strides()[ev.dimension() - 2], *(ev.shape().end() - 1));
            }

            return adjust_secondary_stride(ev.strides()[1], *(ev.shape().begin()));
        }

        template <class E>
        inline std::size_t leading_axis_n_iters(const E& ev)
        {
            if (ev.layout() == layout_type::row_major)
            {
                return std::accumulate(
                    ev.shape().begin(),
                    ev.shape().end() - 1,
                    std::size_t(1),
                    std::multiplies<>()
                );
            }
            return std::accumulate(ev.shape().begin() + 1, ev.shape().end(), std::size_t(1), std::multiplies<>());
        }

        template <class E, class F>
        inline void call_over_leading_axis(E& ev, F&& fct)
        {
            XTENSOR_ASSERT(ev.dimension() >= 2);

            const std::size_t n_iters = leading_axis_n_iters(ev);
            const std::ptrdiff_t secondary_stride = get_secondary_stride(ev);

            const auto begin = ev.data();
            const auto end = begin + n_iters * secondary_stride;
            for (auto iter = begin; iter != end; iter += secondary_stride)
            {
                fct(iter, iter + secondary_stride);
            }
        }

        template <class E1, class E2, class F>
        inline void call_over_leading_axis(E1& e1, E2& e2, F&& fct)
        {
            XTENSOR_ASSERT(e1.dimension() >= 2);
            XTENSOR_ASSERT(e1.dimension() == e2.dimension());

            const std::size_t n_iters = leading_axis_n_iters(e1);
            const std::ptrdiff_t secondary_stride1 = get_secondary_stride(e1);
            const std::ptrdiff_t secondary_stride2 = get_secondary_stride(e2);
            XTENSOR_ASSERT(secondary_stride1 == secondary_stride2);

            const auto begin1 = e1.data();
            const auto end1 = begin1 + n_iters * secondary_stride1;
            const auto begin2 = e2.data();
            const auto end2 = begin2 + n_iters * secondary_stride2;
            auto iter1 = begin1;
            auto iter2 = begin2;
            for (; (iter1 != end1) && (iter2 != end2); iter1 += secondary_stride1, iter2 += secondary_stride2)
            {
                fct(iter1, iter1 + secondary_stride1, iter2, iter2 + secondary_stride2);
            }
        }

        template <class E>
        inline std::size_t leading_axis(const E& e)
        {
            if (e.layout() == layout_type::row_major)
            {
                return e.dimension() - 1;
            }
            else if (e.layout() == layout_type::column_major)
            {
                return 0;
            }
            XTENSOR_THROW(std::runtime_error, "Layout not supported.");
        }

        // get permutations to transpose and reverse-transpose array
        inline std::pair<dynamic_shape<std::size_t>, dynamic_shape<std::size_t>>
        get_permutations(std::size_t dim, std::size_t ax, layout_type layout)
        {
            dynamic_shape<std::size_t> permutation(dim);
            std::iota(permutation.begin(), permutation.end(), std::size_t(0));
            permutation.erase(permutation.begin() + std::ptrdiff_t(ax));

            if (layout == layout_type::row_major)
            {
                permutation.push_back(ax);
            }
            else
            {
                permutation.insert(permutation.begin(), ax);
            }

            // TODO find a more clever way to get reverse permutation?
            dynamic_shape<std::size_t> reverse_permutation;
            for (std::size_t i = 0; i < dim; ++i)
            {
                auto it = std::find(permutation.begin(), permutation.end(), i);
                reverse_permutation.push_back(std::size_t(std::distance(permutation.begin(), it)));
            }

            return std::make_pair(std::move(permutation), std::move(reverse_permutation));
        }

        template <class R, class E, class F>
        inline R map_axis(const E& e, std::ptrdiff_t axis, F&& lambda)
        {
            if (e.dimension() == 1)
            {
                R res = e;
                lambda(res.begin(), res.end());
                return res;
            }

            const std::size_t ax = normalize_axis(e.dimension(), axis);
            if (ax == detail::leading_axis(e))
            {
                R res = e;
                detail::call_over_leading_axis(res, std::forward<F>(lambda));
                return res;
            }

            dynamic_shape<std::size_t> permutation, reverse_permutation;
            std::tie(permutation, reverse_permutation) = get_permutations(e.dimension(), ax, e.layout());
            R res = transpose(e, permutation);
            detail::call_over_leading_axis(res, std::forward<F>(lambda));
            res = transpose(res, reverse_permutation);
            return res;
        }

        template <class VT>
        struct flatten_sort_result_type_impl
        {
            using type = VT;
        };

        template <class VT, std::size_t N, layout_type L>
        struct flatten_sort_result_type_impl<xtensor<VT, N, L>>
        {
            using type = xtensor<VT, 1, L>;
        };

        template <class VT, class S, layout_type L>
        struct flatten_sort_result_type_impl<xtensor_fixed<VT, S, L>>
        {
            using type = xtensor_fixed<VT, xshape<fixed_compute_size<S>::value>, L>;
        };

        template <class VT>
        struct flatten_sort_result_type : flatten_sort_result_type_impl<common_tensor_type_t<VT>>
        {
        };

        template <class VT>
        using flatten_sort_result_type_t = typename flatten_sort_result_type<VT>::type;

        template <class E, class R = flatten_sort_result_type_t<E>>
        inline auto flat_sort_impl(const xexpression<E>& e)
        {
            const auto& de = e.derived_cast();
            R ev;
            ev.resize({static_cast<typename R::shape_type::value_type>(de.size())});

            std::copy(de.cbegin(), de.cend(), ev.begin());
            std::sort(ev.begin(), ev.end());

            return ev;
        }
    }

    template <class E>
    inline auto sort(const xexpression<E>& e, placeholders::xtuph /*t*/)
    {
        return detail::flat_sort_impl(e);
    }

    namespace detail
    {
        template <class T>
        struct sort_eval_type
        {
            using type = typename T::temporary_type;
        };

        template <class T, std::size_t... I, layout_type L>
        struct sort_eval_type<xtensor_fixed<T, fixed_shape<I...>, L>>
        {
            using type = xtensor<T, sizeof...(I), L>;
        };
    }

    /**
     * Sort xexpression (optionally along axis)
     * The sort is performed using the ``std::sort`` functions.
     * A copy of the xexpression is created and returned.
     *
     * @ingroup xt_xsort
     * @param e xexpression to sort
     * @param axis axis along which sort is performed
     *
     * @return sorted array (copy)
     */
    template <class E>
    inline auto sort(const xexpression<E>& e, std::ptrdiff_t axis = -1)
    {
        using eval_type = typename detail::sort_eval_type<E>::type;

        return detail::map_axis<eval_type>(
            e.derived_cast(),
            axis,
            [](auto begin, auto end)
            {
                std::sort(begin, end);
            }
        );
    }

    /*****************************
     * Implementation of argsort *
     *****************************/

    /**
     * Sorting method.
     * Predefined methods for performing indirect sorting.
     * @see argsort(const xexpression<E>&, std::ptrdiff_t, sorting_method)
     */
    enum class sorting_method
    {
        /**
         *  Faster method but with no guarantee on preservation of order of equal elements
         *  https://en.cppreference.com/w/cpp/algorithm/sort.
         */
        quick,
        /**
         *  Slower method but with guarantee on preservation of order of equal elements
         *  https://en.cppreference.com/w/cpp/algorithm/stable_sort.
         */
        stable,
    };

    namespace detail
    {
        template <class ConstRandomIt, class RandomIt, class Compare, class Method>
        inline void argsort_iter(
            ConstRandomIt data_begin,
            ConstRandomIt data_end,
            RandomIt idx_begin,
            RandomIt idx_end,
            Compare comp,
            Method method
        )
        {
            XTENSOR_ASSERT(std::distance(data_begin, data_end) >= 0);
            XTENSOR_ASSERT(std::distance(idx_begin, idx_end) == std::distance(data_begin, data_end));
            (void) idx_end;  // TODO(C++17) [[maybe_unused]] only used in assertion.

            std::iota(idx_begin, idx_end, 0);
            switch (method)
            {
                case (sorting_method::quick):
                {
                    std::sort(
                        idx_begin,
                        idx_end,
                        [&](const auto i, const auto j)
                        {
                            return comp(*(data_begin + i), *(data_begin + j));
                        }
                    );
                }
                case (sorting_method::stable):
                {
                    std::stable_sort(
                        idx_begin,
                        idx_end,
                        [&](const auto i, const auto j)
                        {
                            return comp(*(data_begin + i), *(data_begin + j));
                        }
                    );
                }
            }
        }

        template <class ConstRandomIt, class RandomIt, class Method>
        inline void
        argsort_iter(ConstRandomIt data_begin, ConstRandomIt data_end, RandomIt idx_begin, RandomIt idx_end, Method method)
        {
            return argsort_iter(
                std::move(data_begin),
                std::move(data_end),
                std::move(idx_begin),
                std::move(idx_end),
                [](const auto& x, const auto& y) -> bool
                {
                    return x < y;
                },
                method
            );
        }

        template <class VT, class T>
        struct rebind_value_type
        {
            using type = xarray<VT, xt::layout_type::dynamic>;
        };

        template <class VT, class EC, layout_type L>
        struct rebind_value_type<VT, xarray<EC, L>>
        {
            using type = xarray<VT, L>;
        };

        template <class VT, class EC, std::size_t N, layout_type L>
        struct rebind_value_type<VT, xtensor<EC, N, L>>
        {
            using type = xtensor<VT, N, L>;
        };

        template <class VT, class ET, class S, layout_type L>
        struct rebind_value_type<VT, xtensor_fixed<ET, S, L>>
        {
            using type = xtensor_fixed<VT, S, L>;
        };

        template <class VT, class T>
        struct flatten_rebind_value_type
        {
            using type = typename rebind_value_type<VT, T>::type;
        };

        template <class VT, class EC, std::size_t N, layout_type L>
        struct flatten_rebind_value_type<VT, xtensor<EC, N, L>>
        {
            using type = xtensor<VT, 1, L>;
        };

        template <class VT, class ET, class S, layout_type L>
        struct flatten_rebind_value_type<VT, xtensor_fixed<ET, S, L>>
        {
            using type = xtensor_fixed<VT, xshape<fixed_compute_size<S>::value>, L>;
        };

        template <class T>
        struct argsort_result_type
        {
            using type = typename rebind_value_type<typename T::temporary_type::size_type, typename T::temporary_type>::type;
        };

        template <class T>
        struct linear_argsort_result_type
        {
            using type = typename flatten_rebind_value_type<
                typename T::temporary_type::size_type,
                typename T::temporary_type>::type;
        };

        template <class E, class R = typename detail::linear_argsort_result_type<E>::type, class Method>
        inline auto flatten_argsort_impl(const xexpression<E>& e, Method method)
        {
            const auto& de = e.derived_cast();

            auto cit = de.template begin<layout_type::row_major>();
            using const_iterator = decltype(cit);
            auto ad = xiterator_adaptor<const_iterator, const_iterator>(cit, cit, de.size());

            using result_type = R;
            result_type result;
            result.resize({de.size()});

            detail::argsort_iter(de.cbegin(), de.cend(), result.begin(), result.end(), method);

            return result;
        }
    }

    template <class E>
    inline auto
    argsort(const xexpression<E>& e, placeholders::xtuph /*t*/, sorting_method method = sorting_method::quick)
    {
        return detail::flatten_argsort_impl(e, method);
    }

    /**
     * Argsort xexpression (optionally along axis)
     * Performs an indirect sort along the given axis. Returns an xarray
     * of indices of the same shape as e that index data along the given axis in
     * sorted order.
     *
     * @ingroup xt_xsort
     * @param e xexpression to argsort
     * @param axis axis along which argsort is performed
     * @param method sorting algorithm to use
     *
     * @return argsorted index array
     *
     * @see xt::sorting_method
     */
    template <class E>
    inline auto
    argsort(const xexpression<E>& e, std::ptrdiff_t axis = -1, sorting_method method = sorting_method::quick)
    {
        using eval_type = typename detail::sort_eval_type<E>::type;
        using result_type = typename detail::argsort_result_type<eval_type>::type;

        const auto& de = e.derived_cast();

        std::size_t ax = normalize_axis(de.dimension(), axis);

        if (de.dimension() == 1)
        {
            return detail::flatten_argsort_impl<E, result_type>(e, method);
        }

        const auto argsort = [&method](auto res_begin, auto res_end, auto ev_begin, auto ev_end)
        {
            detail::argsort_iter(ev_begin, ev_end, res_begin, res_end, method);
        };

        if (ax == detail::leading_axis(de))
        {
            result_type res = result_type::from_shape(de.shape());
            detail::call_over_leading_axis(res, de, argsort);
            return res;
        }

        dynamic_shape<std::size_t> permutation, reverse_permutation;
        std::tie(permutation, reverse_permutation) = detail::get_permutations(de.dimension(), ax, de.layout());
        eval_type ev = transpose(de, permutation);
        result_type res = result_type::from_shape(ev.shape());
        detail::call_over_leading_axis(res, ev, argsort);
        res = transpose(res, reverse_permutation);
        return res;
    }

    /************************************************
     * Implementation of partition and argpartition *
     ************************************************/

    namespace detail
    {
        /**
         * Partition a given random iterator.
         *
         * @param data_begin Start of the data to partition.
         * @param data_end Past end of the data to partition.
         * @param kth_start Start of the indices to partition.
         *        Indices must be sorted in decreasing order.
         * @param kth_end Past end of the indices to partition.
         *        Indices must be sorted in decreasing order.
         * @param comp Comparison function for `x < y`.
         */
        template <class RandomIt, class Iter, class Compare>
        inline void
        partition_iter(RandomIt data_begin, RandomIt data_end, Iter kth_begin, Iter kth_end, Compare comp)
        {
            XTENSOR_ASSERT(std::distance(data_begin, data_end) >= 0);
            XTENSOR_ASSERT(std::distance(kth_begin, kth_end) >= 0);

            using idx_type = typename std::iterator_traits<Iter>::value_type;

            idx_type k_last = static_cast<idx_type>(std::distance(data_begin, data_end));
            for (; kth_begin != kth_end; ++kth_begin)
            {
                std::nth_element(data_begin, data_begin + *kth_begin, data_begin + k_last, std::move(comp));
                k_last = *kth_begin;
            }
        }

        template <class RandomIt, class Iter>
        inline void partition_iter(RandomIt data_begin, RandomIt data_end, Iter kth_begin, Iter kth_end)
        {
            return partition_iter(
                std::move(data_begin),
                std::move(data_end),
                std::move(kth_begin),
                std::move(kth_end),
                [](const auto& x, const auto& y) -> bool
                {
                    return x < y;
                }
            );
        }
    }

    /**
     * Partially sort xexpression
     *
     * Partition shuffles the xexpression in a way so that the kth element
     * in the returned xexpression is in the place it would appear in a sorted
     * array and all elements smaller than this entry are placed (unsorted) before.
     *
     * The optional third parameter can either be an axis or ``xnone()`` in which case
     * the xexpression will be flattened.
     *
     * This function uses ``std::nth_element`` internally.
     *
     * @code{cpp}
     * xt::xarray<float> a = {1, 10, -10, 123};
     * std::cout << xt::partition(a, 0) << std::endl; // {-10, 1, 123, 10} the correct entry at index 0
     * std::cout << xt::partition(a, 3) << std::endl; // {1, 10, -10, 123} the correct entry at index 3
     * std::cout << xt::partition(a, {0, 3}) << std::endl; // {-10, 1, 10, 123} the correct entries at index 0
     * and 3 \endcode
     *
     * @ingroup xt_xsort
     * @param e input xexpression
     * @param kth_container a container of ``indices`` that should contain the correctly sorted value
     * @param axis either integer (default = -1) to sort along last axis or ``xnone()`` to flatten before
     * sorting
     *
     * @return partially sorted xcontainer
     */
    template <class E, xtl::non_integral_concept C, class R = detail::flatten_sort_result_type_t<E>>
    inline R partition(const xexpression<E>& e, C kth_container, placeholders::xtuph /*ax*/)
    {
        const auto& de = e.derived_cast();

        R ev = R::from_shape({de.size()});
        std::sort(kth_container.begin(), kth_container.end());

        std::copy(de.linear_cbegin(), de.linear_cend(), ev.linear_begin());  // flatten

        detail::partition_iter(ev.linear_begin(), ev.linear_end(), kth_container.rbegin(), kth_container.rend());

        return ev;
    }

    template <class E, class I, std::size_t N, class R = detail::flatten_sort_result_type_t<E>>
    inline R partition(const xexpression<E>& e, const I (&kth_container)[N], placeholders::xtuph tag)
    {
        return partition(
            e,
            xtl::forward_sequence<std::array<std::size_t, N>, decltype(kth_container)>(kth_container),
            tag
        );
    }

    template <class E, class R = detail::flatten_sort_result_type_t<E>>
    inline R partition(const xexpression<E>& e, std::size_t kth, placeholders::xtuph tag)
    {
        return partition(e, std::array<std::size_t, 1>({kth}), tag);
    }

    template <class E, xtl::non_integral_concept C>
    inline auto partition(const xexpression<E>& e, C kth_container, std::ptrdiff_t axis = -1)
    {
        using eval_type = typename detail::sort_eval_type<E>::type;

        std::sort(kth_container.begin(), kth_container.end());

        return detail::map_axis<eval_type>(
            e.derived_cast(),
            axis,
            [&kth_container](auto begin, auto end)
            {
                detail::partition_iter(begin, end, kth_container.rbegin(), kth_container.rend());
            }
        );
    }

    template <class E, class T, std::size_t N>
    inline auto partition(const xexpression<E>& e, const T (&kth_container)[N], std::ptrdiff_t axis = -1)
    {
        return partition(
            e,
            xtl::forward_sequence<std::array<std::size_t, N>, decltype(kth_container)>(kth_container),
            axis
        );
    }

    template <class E>
    inline auto partition(const xexpression<E>& e, std::size_t kth, std::ptrdiff_t axis = -1)
    {
        return partition(e, std::array<std::size_t, 1>({kth}), axis);
    }

    /**
     * Partially sort arguments
     *
     * Argpartition shuffles the indices to a xexpression in a way so that the index for the
     * kth element in the returned xexpression is in the place it would appear in a sorted
     * array and all elements smaller than this entry are placed (unsorted) before.
     *
     * The optional third parameter can either be an axis or ``xnone()`` in which case
     * the xexpression will be flattened.
     *
     * This function uses ``std::nth_element`` internally.
     *
     * @code{cpp}
     * xt::xarray<float> a = {1, 10, -10, 123};
     * std::cout << xt::argpartition(a, 0) << std::endl; // {2, 0, 3, 1} the correct entry at index 0
     * std::cout << xt::argpartition(a, 3) << std::endl; // {0, 1, 2, 3} the correct entry at index 3
     * std::cout << xt::argpartition(a, {0, 3}) << std::endl; // {2, 0, 1, 3} the correct entries at index 0
     * and 3 \endcode
     *
     * @ingroup xt_xsort
     * @param e input xexpression
     * @param kth_container a container of ``indices`` that should contain the correctly sorted value
     * @param axis either integer (default = -1) to sort along last axis or ``xnone()`` to flatten before
     * sorting
     *
     * @return xcontainer with indices of partial sort of input
     */
    template <
        class E,
        xtl::non_integral_concept C,
        class R = typename detail::linear_argsort_result_type<typename detail::sort_eval_type<E>::type>::type>
    inline R argpartition(const xexpression<E>& e, C kth_container, placeholders::xtuph)
    {
        using eval_type = typename detail::sort_eval_type<E>::type;
        using result_type = typename detail::linear_argsort_result_type<eval_type>::type;

        const auto& de = e.derived_cast();

        result_type res = result_type::from_shape({de.size()});

        std::sort(kth_container.begin(), kth_container.end());

        std::iota(res.linear_begin(), res.linear_end(), 0);

        detail::partition_iter(
            res.linear_begin(),
            res.linear_end(),
            kth_container.rbegin(),
            kth_container.rend(),
            [&de](std::size_t a, std::size_t b)
            {
                return de[a] < de[b];
            }
        );

        return res;
    }

    template <class E, class I, std::size_t N>
    inline auto argpartition(const xexpression<E>& e, const I (&kth_container)[N], placeholders::xtuph tag)
    {
        return argpartition(
            e,
            xtl::forward_sequence<std::array<std::size_t, N>, decltype(kth_container)>(kth_container),
            tag
        );
    }

    template <class E>
    inline auto argpartition(const xexpression<E>& e, std::size_t kth, placeholders::xtuph tag)
    {
        return argpartition(e, std::array<std::size_t, 1>({kth}), tag);
    }

    template <class E, xtl::non_integral_concept C>
    inline auto argpartition(const xexpression<E>& e, C kth_container, std::ptrdiff_t axis = -1)
    {
        using eval_type = typename detail::sort_eval_type<E>::type;
        using result_type = typename detail::argsort_result_type<eval_type>::type;

        const auto& de = e.derived_cast();

        if (de.dimension() == 1)
        {
            return argpartition<E, C, result_type>(e, std::forward<C>(kth_container), xnone());
        }

        std::sort(kth_container.begin(), kth_container.end());
        const auto argpartition_w_kth =
            [&kth_container](auto res_begin, auto res_end, auto ev_begin, auto /*ev_end*/)
        {
            std::iota(res_begin, res_end, 0);
            detail::partition_iter(
                res_begin,
                res_end,
                kth_container.rbegin(),
                kth_container.rend(),
                [&ev_begin](auto const& i, auto const& j)
                {
                    return *(ev_begin + i) < *(ev_begin + j);
                }
            );
        };

        const std::size_t ax = normalize_axis(de.dimension(), axis);
        if (ax == detail::leading_axis(de))
        {
            result_type res = result_type::from_shape(de.shape());
            detail::call_over_leading_axis(res, de, argpartition_w_kth);
            return res;
        }

        dynamic_shape<std::size_t> permutation, reverse_permutation;
        std::tie(permutation, reverse_permutation) = detail::get_permutations(de.dimension(), ax, de.layout());
        eval_type ev = transpose(de, permutation);
        result_type res = result_type::from_shape(ev.shape());
        detail::call_over_leading_axis(res, ev, argpartition_w_kth);
        res = transpose(res, reverse_permutation);
        return res;
    }

    template <class E, class I, std::size_t N>
    inline auto argpartition(const xexpression<E>& e, const I (&kth_container)[N], std::ptrdiff_t axis = -1)
    {
        return argpartition(
            e,
            xtl::forward_sequence<std::array<std::size_t, N>, decltype(kth_container)>(kth_container),
            axis
        );
    }

    template <class E>
    inline auto argpartition(const xexpression<E>& e, std::size_t kth, std::ptrdiff_t axis = -1)
    {
        return argpartition(e, std::array<std::size_t, 1>({kth}), axis);
    }

    /******************
     *  xt::quantile  *
     ******************/

    namespace detail
    {
        template <class S, class I, class K, class O>
        inline void select_indices_impl(
            const S& shape,
            const I& indices,
            std::size_t axis,
            std::size_t current_dim,
            const K& current_index,
            O& out
        )
        {
            using id_t = typename K::value_type;
            if ((current_dim < shape.size() - 1) && (current_dim == axis))
            {
                for (auto i : indices)
                {
                    auto idx = current_index;
                    idx[current_dim] = i;
                    select_indices_impl(shape, indices, axis, current_dim + 1, idx, out);
                }
            }
            else if ((current_dim < shape.size() - 1) && (current_dim != axis))
            {
                for (id_t i = 0; xtl::cmp_less(i, shape[current_dim]); ++i)
                {
                    auto idx = current_index;
                    idx[current_dim] = i;
                    select_indices_impl(shape, indices, axis, current_dim + 1, idx, out);
                }
            }
            else if ((current_dim == shape.size() - 1) && (current_dim == axis))
            {
                for (auto i : indices)
                {
                    auto idx = current_index;
                    idx[current_dim] = i;
                    out.push_back(std::move(idx));
                }
            }
            else if ((current_dim == shape.size() - 1) && (current_dim != axis))
            {
                for (id_t i = 0; xtl::cmp_less(i, shape[current_dim]); ++i)
                {
                    auto idx = current_index;
                    idx[current_dim] = i;
                    out.push_back(std::move(idx));
                }
            }
        }

        template <class S, class I>
        inline auto select_indices(const S& shape, const I& indices, std::size_t axis)
        {
            using index_type = get_strides_t<S>;
            auto out = std::vector<index_type>();
            select_indices_impl(shape, indices, axis, 0, xtl::make_sequence<index_type>(shape.size()), out);
            return out;
        }

        // TODO remove when fancy index views are implemented
        // Poor man's indexing along a single axis as in NumPy a[:, [1, 3, 4]]
        template <class E, class I>
        inline auto fancy_indexing(E&& e, const I& indices, std::ptrdiff_t axis)
        {
            const std::size_t ax = normalize_axis(e.dimension(), axis);
            using shape_t = get_strides_t<typename std::decay_t<E>::shape_type>;
            auto shape = xtl::forward_sequence<shape_t, decltype(e.shape())>(e.shape());
            shape[ax] = indices.size();
            return reshape_view(
                index_view(std::forward<E>(e), select_indices(e.shape(), indices, ax)),
                std::move(shape)
            );
        }

        template <class T, class I, class P>
        inline auto quantile_kth_gamma(std::size_t n, const P& probas, T alpha, T beta)
        {
            const auto m = alpha + probas * (T(1) - alpha - beta);
            // Evaluting since reused a lot
            const auto p_n_m = eval(probas * static_cast<T>(n) + m - 1);
            // Previous (virtual) index, may be out of bounds
            const auto j = floor(p_n_m);
            const auto j_jp1 = concatenate(xtuple(j, j + 1));
            // Both interpolation indices, k and k+1
            const auto k_kp1 = xt::cast<std::size_t>(clip(j_jp1, T(0), T(n - 1)));
            // Both interpolation coefficients, 1-gamma and gamma
            const auto omg_g = concatenate(xtuple(T(1) - (p_n_m - j), p_n_m - j));
            return std::make_pair(eval(k_kp1), eval(omg_g));
        }

        // TODO should implement unsqueeze rather
        template <class S>
        inline auto unsqueeze_shape(const S& shape, std::size_t axis)
        {
            XTENSOR_ASSERT(axis <= shape.size());
            auto new_shape = xtl::forward_sequence<xt::svector<std::size_t>, decltype(shape)>(shape);
            new_shape.insert(new_shape.begin() + axis, 1);
            return new_shape;
        }
    }

    /**
     * Compute quantiles over the given axis.
     *
     * In a sorted array represneting a distribution of numbers, the quantile of a probability ``p``
     * is the the cut value ``q`` such that a fraction ``p`` of the distribution is lesser or equal
     * to ``q``.
     * When the cutpoint falls between two elemnts of the sample distribution, a interpolation is
     * computed using the @p alpha and @p beta coefficients, as descripted in
     * (Hyndman and Fan, 1996).
     *
     * The algorithm partially sorts entries in a copy along the @p axis axis.
     *
     * @ingroup xt_xsort
     * @param e Expression containing the distribution over which the quantiles are computed.
     * @param probas An list of probability associated with each desired quantiles.
     *        All elements must be in the range ``[0, 1]``.
     * @param axis The dimension in which to compute the quantiles, *i.e* the axis representing the
     *        distribution.
     * @param alpha Interpolation parameter. Must be in the range ``[0, 1]]``.
     * @param beta Interpolation parameter. Must be in the range ``[0, 1]]``.
     * @tparam T The type in which the quantile are computed.
     * @return An expression with as many dimensions as the input @p e.
     *         The first axis correspond to the quantiles.
     *         The other axes are the axes that remain after the reduction of @p e.
     * @see (Hyndman and Fan, 1996) R. J. Hyndman and Y. Fan,
     *      "Sample quantiles in statistical packages", The American Statistician,
     *      50(4), pp. 361-365, 1996
     * @see https://en.wikipedia.org/wiki/Quantile
     */
    template <class T = double, class E, class P>
    inline auto quantile(E&& e, const P& probas, std::ptrdiff_t axis, T alpha, T beta)
    {
        XTENSOR_ASSERT(all(0. <= probas));
        XTENSOR_ASSERT(all(probas <= 1.));
        XTENSOR_ASSERT(0. <= alpha);
        XTENSOR_ASSERT(alpha <= 1.);
        XTENSOR_ASSERT(0. <= beta);
        XTENSOR_ASSERT(beta <= 1.);

        using tmp_shape_t = get_strides_t<typename std::decay_t<E>::shape_type>;
        using id_t = typename tmp_shape_t::value_type;

        const std::size_t ax = normalize_axis(e.dimension(), axis);
        const std::size_t n = e.shape()[ax];
        auto kth_gamma = detail::quantile_kth_gamma<T, id_t, P>(n, probas, alpha, beta);

        // Select relevant values for computing interpolating quantiles
        auto e_partition = xt::partition(std::forward<E>(e), kth_gamma.first, ax);
        auto e_kth = detail::fancy_indexing(std::move(e_partition), std::move(kth_gamma.first), ax);

        // Reshape interpolation coefficients
        auto gm1_g_shape = xtl::make_sequence<tmp_shape_t>(e.dimension(), 1);
        gm1_g_shape[ax] = kth_gamma.second.size();
        auto gm1_g_reshaped = reshape_view(std::move(kth_gamma.second), std::move(gm1_g_shape));

        // Compute interpolation
        // TODO(C++20) use (and create) xt::lerp in C++
        auto e_kth_g = std::move(e_kth) * std::move(gm1_g_reshaped);
        // Reshape pairwise interpolate for suming along new axis
        auto e_kth_g_shape = detail::unsqueeze_shape(e_kth_g.shape(), ax);
        e_kth_g_shape[ax] = 2;
        e_kth_g_shape[ax + 1] /= 2;
        auto quantiles = xt::sum(reshape_view(std::move(e_kth_g), std::move(e_kth_g_shape)), ax);
        // Cannot do a transpose on a non-strided expression so we have to eval
        return moveaxis(eval(std::move(quantiles)), ax, 0);
    }

    // Static proba array overload
    template <class T = double, class E, std::size_t N>
    inline auto quantile(E&& e, const T (&probas)[N], std::ptrdiff_t axis, T alpha, T beta)
    {
        return quantile(std::forward<E>(e), adapt(probas, {N}), axis, alpha, beta);
    }

    /**
     * Compute quantiles of the whole expression.
     *
     * The quantiles are computed over the whole expression, as if flatten in a one-dimensional
     * expression.
     *
     * @ingroup xt_xsort
     * @see xt::quantile(E&& e, P const& probas, std::ptrdiff_t axis, T alpha, T beta)
     */
    template <class T = double, class E, class P>
    inline auto quantile(E&& e, const P& probas, T alpha, T beta)
    {
        return quantile(xt::ravel(std::forward<E>(e)), probas, 0, alpha, beta);
    }

    // Static proba array overload
    template <class T = double, class E, std::size_t N>
    inline auto quantile(E&& e, const T (&probas)[N], T alpha, T beta)
    {
        return quantile(std::forward<E>(e), adapt(probas, {N}), alpha, beta);
    }

    /**
     * Quantile interpolation method.
     *
     * Predefined methods for interpolating quantiles, as defined in (Hyndman and Fan, 1996).
     *
     * @ingroup xt_xsort
     * @see (Hyndman and Fan, 1996) R. J. Hyndman and Y. Fan,
     *      "Sample quantiles in statistical packages", The American Statistician,
     *      50(4), pp. 361-365, 1996
     * @see xt::quantile(E&& e, P const& probas, std::ptrdiff_t axis, xt::quantile_method method)
     */
    enum class quantile_method
    {
        /** Method 4 of (Hyndman and Fan, 1996) with ``alpha=0`` and ``beta=1``. */
        interpolated_inverted_cdf = 4,
        /** Method 5 of (Hyndman and Fan, 1996) with ``alpha=1/2`` and ``beta=1/2``. */
        hazen,
        /** Method 6 of (Hyndman and Fan, 1996) with ``alpha=0`` and ``beta=0``. */
        weibull,
        /** Method 7 of (Hyndman and Fan, 1996) with ``alpha=1`` and ``beta=1``. */
        linear,
        /** Method 8 of (Hyndman and Fan, 1996) with ``alpha=1/3`` and ``beta=1/3``. */
        median_unbiased,
        /** Method 9 of (Hyndman and Fan, 1996) with ``alpha=3/8`` and ``beta=3/8``. */
        normal_unbiased,
    };

    /**
     * Compute quantiles over the given axis.
     *
     * The function takes the name of a predefined method to compute to interpolate between values.
     *
     * @ingroup xt_xsort
     * @see xt::quantile_method
     * @see xt::quantile(E&& e, P const& probas, std::ptrdiff_t axis, T alpha, T beta)
     */
    template <class T = double, class E, class P>
    inline auto
    quantile(E&& e, const P& probas, std::ptrdiff_t axis, quantile_method method = quantile_method::linear)
    {
        T alpha = 0.;
        T beta = 0.;
        switch (method)
        {
            case (quantile_method::interpolated_inverted_cdf):
            {
                alpha = 0.;
                beta = 1.;
                break;
            }
            case (quantile_method::hazen):
            {
                alpha = 0.5;
                beta = 0.5;
                break;
            }
            case (quantile_method::weibull):
            {
                alpha = 0.;
                beta = 0.;
                break;
            }
            case (quantile_method::linear):
            {
                alpha = 1.;
                beta = 1.;
                break;
            }
            case (quantile_method::median_unbiased):
            {
                alpha = 1. / 3.;
                beta = 1. / 3.;
                break;
            }
            case (quantile_method::normal_unbiased):
            {
                alpha = 3. / 8.;
                beta = 3. / 8.;
                break;
            }
        }
        return quantile(std::forward<E>(e), probas, axis, alpha, beta);
    }

    // Static proba array overload
    template <class T = double, class E, std::size_t N>
    inline auto
    quantile(E&& e, const T (&probas)[N], std::ptrdiff_t axis, quantile_method method = quantile_method::linear)
    {
        return quantile(std::forward<E>(e), adapt(probas, {N}), axis, method);
    }

    /**
     * Compute quantiles of the whole expression.
     *
     * The quantiles are computed over the whole expression, as if flatten in a one-dimensional
     * expression.
     * The function takes the name of a predefined method to compute to interpolate between values.
     *
     * @ingroup xt_xsort
     * @see xt::quantile_method
     * @see xt::quantile(E&& e, P const& probas, std::ptrdiff_t axis, xt::quantile_method method)
     */
    template <class T = double, class E, class P>
    inline auto quantile(E&& e, const P& probas, quantile_method method = quantile_method::linear)
    {
        return quantile(xt::ravel(std::forward<E>(e)), probas, 0, method);
    }

    // Static proba array overload
    template <class T = double, class E, std::size_t N>
    inline auto quantile(E&& e, const T (&probas)[N], quantile_method method = quantile_method::linear)
    {
        return quantile(std::forward<E>(e), adapt(probas, {N}), method);
    }

    /****************
     *  xt::median  *
     ****************/

    template <class E>
    inline typename std::decay_t<E>::value_type median(E&& e)
    {
        using value_type = typename std::decay_t<E>::value_type;
        auto sz = e.size();
        if (sz % 2 == 0)
        {
            std::size_t szh = sz / 2;  // integer floor div
            std::array<std::size_t, 2> kth = {szh - 1, szh};
            auto values = xt::partition(xt::flatten(e), kth);
            return (values[kth[0]] + values[kth[1]]) / value_type(2);
        }
        else
        {
            std::array<std::size_t, 1> kth = {(sz - 1) / 2};
            auto values = xt::partition(xt::flatten(e), kth);
            return values[kth[0]];
        }
    }

    /**
     * Find the median along the specified axis
     *
     * Given a vector V of length N, the median of V is the middle value of a
     * sorted copy of V, V_sorted - i e., V_sorted[(N-1)/2], when N is odd,
     * and the average of the two middle values of V_sorted when N is even.
     *
     * @ingroup xt_xsort
     * @param axis axis along which the medians are computed.
     *             If not set, computes the median along a flattened version of the input.
     * @param e input xexpression
     * @return median value
     */
    template <class E>
    inline auto median(E&& e, std::ptrdiff_t axis)
    {
        std::size_t ax = normalize_axis(e.dimension(), axis);
        std::size_t sz = e.shape()[ax];
        xstrided_slice_vector sv(e.dimension(), xt::all());

        if (sz % 2 == 0)
        {
            std::size_t szh = sz / 2;  // integer floor div
            std::array<std::size_t, 2> kth = {szh - 1, szh};
            auto values = xt::partition(std::forward<E>(e), kth, static_cast<ptrdiff_t>(ax));
            sv[ax] = xt::range(szh - 1, szh + 1);
            return xt::mean(xt::strided_view(std::move(values), std::move(sv)), {ax});
        }
        else
        {
            std::size_t szh = (sz - 1) / 2;
            std::array<std::size_t, 1> kth = {(sz - 1) / 2};
            auto values = xt::partition(std::forward<E>(e), kth, static_cast<ptrdiff_t>(ax));
            sv[ax] = xt::range(szh, szh + 1);
            return xt::mean(xt::strided_view(std::move(values), std::move(sv)), {ax});
        }
    }

    namespace detail
    {
        template <class T>
        struct argfunc_result_type
        {
            using type = xarray<std::size_t>;
        };

        template <class T, std::size_t N>
        struct argfunc_result_type<xtensor<T, N>>
        {
            using type = xtensor<std::size_t, N - 1>;
        };

        template <layout_type L, class E, class F>
        inline typename argfunc_result_type<E>::type arg_func_impl(const E& e, std::size_t axis, F&& cmp)
        {
            using eval_type = typename detail::sort_eval_type<E>::type;
            using value_type = typename E::value_type;
            using result_type = typename argfunc_result_type<E>::type;
            using result_shape_type = typename result_type::shape_type;

            if (e.dimension() == 1)
            {
                auto begin = e.template begin<L>();
                auto end = e.template end<L>();
                // todo C++17 : constexpr
                if (std::is_same<F, std::less<value_type>>::value)
                {
                    std::size_t i = static_cast<std::size_t>(std::distance(begin, std::min_element(begin, end)));
                    return xtensor<size_t, 0>{i};
                }
                else
                {
                    std::size_t i = static_cast<std::size_t>(std::distance(begin, std::max_element(begin, end)));
                    return xtensor<size_t, 0>{i};
                }
            }

            result_shape_type alt_shape;
            xt::resize_container(alt_shape, e.dimension() - 1);

            // Excluding copy, copy all of shape except for axis
            std::copy(e.shape().cbegin(), e.shape().cbegin() + std::ptrdiff_t(axis), alt_shape.begin());
            std::copy(
                e.shape().cbegin() + std::ptrdiff_t(axis) + 1,
                e.shape().cend(),
                alt_shape.begin() + std::ptrdiff_t(axis)
            );

            result_type result = result_type::from_shape(std::move(alt_shape));
            auto result_iter = result.template begin<L>();

            auto arg_func_lambda = [&result_iter, &cmp](auto begin, auto end)
            {
                std::size_t idx = 0;
                value_type val = *begin;
                ++begin;
                for (std::size_t i = 1; begin != end; ++begin, ++i)
                {
                    if (cmp(*begin, val))
                    {
                        val = *begin;
                        idx = i;
                    }
                }
                *result_iter = idx;
                ++result_iter;
            };

            if (axis != detail::leading_axis(e))
            {
                dynamic_shape<std::size_t> permutation, reverse_permutation;
                std::tie(
                    permutation,
                    reverse_permutation
                ) = detail::get_permutations(e.dimension(), axis, e.layout());

                // note: creating copy
                eval_type input = transpose(e, permutation);
                detail::call_over_leading_axis(input, arg_func_lambda);
                return result;
            }
            else
            {
                auto&& input = eval(e);
                detail::call_over_leading_axis(input, arg_func_lambda);
                return result;
            }
        }
    }

    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class E>
    inline auto argmin(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(e.derived_cast());
        auto begin = ed.template begin<L>();
        auto end = ed.template end<L>();
        std::size_t i = static_cast<std::size_t>(std::distance(begin, std::min_element(begin, end)));
        return xtensor<size_t, 0>{i};
    }

    /**
     * Find position of minimal value in xexpression.
     * By default, the returned index is into the flattened array.
     * If `axis` is specified, the indices are along the specified axis.
     *
     * @param e input xexpression
     * @param axis select axis (optional)
     *
     * @return returns xarray with positions of minimal value
     */
    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class E>
    inline auto argmin(const xexpression<E>& e, std::ptrdiff_t axis)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(e.derived_cast());
        std::size_t ax = normalize_axis(ed.dimension(), axis);
        return detail::arg_func_impl<L>(ed, ax, std::less<value_type>());
    }

    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class E>
    inline auto argmax(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(e.derived_cast());
        auto begin = ed.template begin<L>();
        auto end = ed.template end<L>();
        std::size_t i = static_cast<std::size_t>(std::distance(begin, std::max_element(begin, end)));
        return xtensor<size_t, 0>{i};
    }

    /**
     * Find position of maximal value in xexpression
     * By default, the returned index is into the flattened array.
     * If `axis` is specified, the indices are along the specified axis.
     *
     * @ingroup xt_xsort
     * @param e input xexpression
     * @param axis select axis (optional)
     *
     * @return returns xarray with positions of maximal value
     */
    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class E>
    inline auto argmax(const xexpression<E>& e, std::ptrdiff_t axis)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(e.derived_cast());
        std::size_t ax = normalize_axis(ed.dimension(), axis);
        return detail::arg_func_impl<L>(ed, ax, std::greater<value_type>());
    }

    /**
     * Find unique elements of a xexpression. This returns a flattened xtensor with
     * sorted, unique elements from the original expression.
     *
     * @ingroup xt_xsort
     * @param e input xexpression (will be flattened)
     */
    template <class E>
    inline auto unique(const xexpression<E>& e)
    {
        auto sorted = sort(e, xnone());
        auto end = std::unique(sorted.begin(), sorted.end());
        std::size_t sz = static_cast<std::size_t>(std::distance(sorted.begin(), end));
        // TODO check if we can shrink the vector without reallocation
        using value_type = typename E::value_type;
        auto result = xtensor<value_type, 1>::from_shape({sz});
        std::copy(sorted.begin(), end, result.begin());
        return result;
    }

    /**
     * Find the set difference of two xexpressions. This returns a flattened xtensor with
     * the sorted, unique values in ar1 that are not in ar2.
     *
     * @ingroup xt_xsort
     * @param ar1 input xexpression (will be flattened)
     * @param ar2 input xexpression
     */
    template <class E1, class E2>
    inline auto setdiff1d(const xexpression<E1>& ar1, const xexpression<E2>& ar2)
    {
        using value_type = typename E1::value_type;

        auto unique1 = unique(ar1);
        auto unique2 = unique(ar2);

        auto tmp = xtensor<value_type, 1>::from_shape({unique1.size()});

        auto end = std::set_difference(unique1.begin(), unique1.end(), unique2.begin(), unique2.end(), tmp.begin());

        std::size_t sz = static_cast<std::size_t>(std::distance(tmp.begin(), end));

        auto result = xtensor<value_type, 1>::from_shape({sz});

        std::copy(tmp.begin(), end, result.begin());

        return result;
    }
}

#endif
