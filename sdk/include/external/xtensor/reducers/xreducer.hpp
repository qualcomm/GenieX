/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_REDUCER_HPP
#define XTENSOR_REDUCER_HPP

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include <xtl/xfunctional.hpp>
#include <xtl/xsequence.hpp>

#include "../core/xaccessible.hpp"
#include "../core/xeval.hpp"
#include "../core/xexpression.hpp"
#include "../core/xiterable.hpp"
#include "../core/xtensor_config.hpp"
#include "../generators/xbuilder.hpp"
#include "../generators/xgenerator.hpp"
#include "../utils/xutils.hpp"

namespace xt
{
    template <template <class...> class A, class... AX, class X, XTL_REQUIRES(is_evaluation_strategy<AX>..., is_evaluation_strategy<X>)>
    auto operator|(const A<AX...>& args, const A<X>& rhs)
    {
        return std::tuple_cat(args, rhs);
    }

    struct keep_dims_type : xt::detail::option_base
    {
    };

    constexpr auto keep_dims = std::tuple<keep_dims_type>{};

    template <class T = double>
    struct xinitial : xt::detail::option_base
    {
        constexpr xinitial(T val)
            : m_val(val)
        {
        }

        constexpr T value() const
        {
            return m_val;
        }

        T m_val;
    };

    template <class T>
    constexpr auto initial(T val)
    {
        return std::make_tuple(xinitial<T>(val));
    }

    template <std::ptrdiff_t I, class T, class Tuple>
    struct tuple_idx_of_impl;

    template <std::ptrdiff_t I, class T>
    struct tuple_idx_of_impl<I, T, std::tuple<>>
    {
        static constexpr std::ptrdiff_t value = -1;
    };

    template <std::ptrdiff_t I, class T, class... Types>
    struct tuple_idx_of_impl<I, T, std::tuple<T, Types...>>
    {
        static constexpr std::ptrdiff_t value = I;
    };

    template <std::ptrdiff_t I, class T, class U, class... Types>
    struct tuple_idx_of_impl<I, T, std::tuple<U, Types...>>
    {
        static constexpr std::ptrdiff_t value = tuple_idx_of_impl<I + 1, T, std::tuple<Types...>>::value;
    };

    template <class S, class... X>
    struct decay_all;

    template <template <class...> class S, class... X>
    struct decay_all<S<X...>>
    {
        using type = S<std::decay_t<X>...>;
    };

    template <class T, class Tuple>
    struct tuple_idx_of
    {
        static constexpr std::ptrdiff_t
            value = tuple_idx_of_impl<0, std::decay_t<T>, typename decay_all<Tuple>::type>::value;
    };

    template <class R, class T>
    struct reducer_options
    {
        template <class X>
        struct initial_tester : std::false_type
        {
        };

        template <class X>
        struct initial_tester<xinitial<X>> : std::true_type
        {
        };

        // Workaround for Apple because tuple_cat is buggy!
        template <class X>
        struct initial_tester<const xinitial<X>> : std::true_type
        {
        };

        using d_t = std::decay_t<T>;

        static constexpr std::size_t initial_val_idx = xtl::mpl::find_if<initial_tester, d_t>::value;
        reducer_options() = default;

        reducer_options(const T& tpl)
        {
            if constexpr (initial_val_idx != std::tuple_size<T>::value)
            {
                initial_value = std::get < initial_val_idx != std::tuple_size<T>::value ? initial_val_idx
                                                                                        : 0 > (tpl).value();
            }
        }

        using evaluation_strategy = std::conditional_t<
            tuple_idx_of<xt::evaluation_strategy::immediate_type, d_t>::value != -1,
            xt::evaluation_strategy::immediate_type,
            xt::evaluation_strategy::lazy_type>;

        using keep_dims = std::
            conditional_t<tuple_idx_of<xt::keep_dims_type, d_t>::value != -1, std::true_type, std::false_type>;

        static constexpr bool has_initial_value = initial_val_idx != std::tuple_size<d_t>::value;

        R initial_value;

        template <class NR>
        using rebind_t = reducer_options<NR, T>;

        template <class NR>
        auto rebind(NR initial, const reducer_options<R, T>&) const
        {
            reducer_options<NR, T> res;
            res.initial_value = initial;
            return res;
        }
    };

    template <class T>
    struct is_reducer_options_impl : std::false_type
    {
    };

    template <class... X>
    struct is_reducer_options_impl<std::tuple<X...>> : std::true_type
    {
    };

    template <class T>
    struct is_reducer_options : is_reducer_options_impl<std::decay_t<T>>
    {
    };

    /**********
     * reduce *
     **********/

#define DEFAULT_STRATEGY_REDUCERS std::tuple<evaluation_strategy::lazy_type>

    template <class ST, class X, class KD = std::false_type>
    struct xreducer_shape_type;

    template <class S1, class S2>
    struct fixed_xreducer_shape_type;

    namespace detail
    {
        template <class O, class RS, class R, class E, class AX>
        inline void shape_computation(
            RS& result_shape,
            R& result,
            E& expr,
            const AX& axes,
            std::enable_if_t<!detail::is_fixed<RS>::value, int> = 0
        )
        {
            if (typename O::keep_dims())
            {
                resize_container(result_shape, expr.dimension());
                for (std::size_t i = 0; i < expr.dimension(); ++i)
                {
                    if (std::find(axes.begin(), axes.end(), i) == axes.end())
                    {
                        // i not in axes!
                        result_shape[i] = expr.shape()[i];
                    }
                    else
                    {
                        result_shape[i] = 1;
                    }
                }
            }
            else
            {
                resize_container(result_shape, expr.dimension() - axes.size());
                for (std::size_t i = 0, idx = 0; i < expr.dimension(); ++i)
                {
                    if (std::find(axes.begin(), axes.end(), i) == axes.end())
                    {
                        // i not in axes!
                        result_shape[idx] = expr.shape()[i];
                        ++idx;
                    }
                }
            }
            result.resize(result_shape, expr.layout());
        }

        // skip shape computation if already done at compile time
        template <class O, class RS, class R, class S, class AX>
        inline void
        shape_computation(RS&, R&, const S&, const AX&, std::enable_if_t<detail::is_fixed<RS>::value, int> = 0)
        {
        }
    }

    template <class F, class E, class R, XTL_REQUIRES(std::is_convertible<typename E::value_type, typename R::value_type>)>
    inline void copy_to_reduced(F&, const E& e, R& result)
    {
        if (e.layout() == layout_type::row_major)
        {
            std::copy(
                e.template cbegin<layout_type::row_major>(),
                e.template cend<layout_type::row_major>(),
                result.data()
            );
        }
        else
        {
            std::copy(
                e.template cbegin<layout_type::column_major>(),
                e.template cend<layout_type::column_major>(),
                result.data()
            );
        }
    }

    template <
        class F,
        class E,
        class R,
        XTL_REQUIRES(std::negation<std::is_convertible<typename E::value_type, typename R::value_type>>)>
    inline void copy_to_reduced(F& f, const E& e, R& result)
    {
        if (e.layout() == layout_type::row_major)
        {
            std::transform(
                e.template cbegin<layout_type::row_major>(),
                e.template cend<layout_type::row_major>(),
                result.data(),
                f
            );
        }
        else
        {
            std::transform(
                e.template cbegin<layout_type::column_major>(),
                e.template cend<layout_type::column_major>(),
                result.data(),
                f
            );
        }
    }

    template <class F, class E, class X, class O>
    inline auto reduce_immediate(F&& f, E&& e, X&& axes, O&& raw_options)
    {
        using reduce_functor_type = typename std::decay_t<F>::reduce_functor_type;
        using init_functor_type = typename std::decay_t<F>::init_functor_type;
        using expr_value_type = typename std::decay_t<E>::value_type;
        using result_type = std::decay_t<decltype(std::declval<reduce_functor_type>()(
            std::declval<init_functor_type>()(),
            std::declval<expr_value_type>()
        ))>;

        using options_t = reducer_options<result_type, std::decay_t<O>>;
        options_t options(raw_options);

        using shape_type = typename xreducer_shape_type<
            typename std::decay_t<E>::shape_type,
            std::decay_t<X>,
            typename options_t::keep_dims>::type;
        using result_container_type = typename detail::xtype_for_shape<
            shape_type>::template type<result_type, std::decay_t<E>::static_layout>;
        result_container_type result;

        // retrieve functors from triple struct
        auto reduce_fct = xt::get<0>(f);
        auto init_fct = xt::get<1>(f);
        auto merge_fct = xt::get<2>(f);

        if (axes.size() == 0)
        {
            result.resize(e.shape(), e.layout());
            auto cpf = [&reduce_fct, &init_fct](const auto& v)
            {
                return reduce_fct(static_cast<result_type>(init_fct()), v);
            };
            copy_to_reduced(cpf, e, result);
            return result;
        }

        shape_type result_shape{};
        dynamic_shape<std::size_t>
            iter_shape = xtl::forward_sequence<dynamic_shape<std::size_t>, decltype(e.shape())>(e.shape());
        dynamic_shape<std::size_t> iter_strides(e.dimension());

        // std::less is used, because as the standard says (24.4.5):
        // A sequence is sorted with respect to a comparator comp if for any iterator i pointing to the
        // sequence and any non-negative integer n such that i + n is a valid iterator pointing to an element
        // of the sequence, comp(*(i + n), *i) == false. Therefore less is required to detect duplicates.
        if (!std::is_sorted(axes.cbegin(), axes.cend(), std::less<>()))
        {
            XTENSOR_THROW(std::runtime_error, "Reducing axes should be sorted.");
        }
        if (std::adjacent_find(axes.cbegin(), axes.cend()) != axes.cend())
        {
            XTENSOR_THROW(std::runtime_error, "Reducing axes should not contain duplicates.");
        }
        if (axes.size() != 0 && axes[axes.size() - 1] > e.dimension() - 1)
        {
            XTENSOR_THROW(
                std::runtime_error,
                "Axis " + std::to_string(axes[axes.size() - 1]) + " out of bounds for reduction."
            );
        }

        detail::shape_computation<options_t>(result_shape, result, e, axes);

        // Fast track for complete reduction
        if (e.dimension() == axes.size())
        {
            result_type tmp = options_t::has_initial_value ? options.initial_value : init_fct();
            result.data()[0] = std::accumulate(e.storage().begin(), e.storage().end(), tmp, reduce_fct);
            return result;
        }

        std::size_t leading_ax = axes[(e.layout() == layout_type::row_major) ? axes.size() - 1 : 0];
        auto strides_finder = e.strides().begin() + static_cast<std::ptrdiff_t>(leading_ax);
        // The computed strides contain "0" where the shape is 1 -- therefore find the next none-zero number
        std::size_t inner_stride = static_cast<std::size_t>(*strides_finder);
        auto iter_bound = e.layout() == layout_type::row_major ? e.strides().begin() : (e.strides().end() - 1);
        while (inner_stride == 0 && strides_finder != iter_bound)
        {
            (e.layout() == layout_type::row_major) ? --strides_finder : ++strides_finder;
            inner_stride = static_cast<std::size_t>(*strides_finder);
        }

        if (inner_stride == 0)
        {
            auto cpf = [&reduce_fct, &init_fct](const auto& v)
            {
                return reduce_fct(static_cast<result_type>(init_fct()), v);
            };
            copy_to_reduced(cpf, e, result);
            return result;
        }

        std::size_t inner_loop_size = static_cast<std::size_t>(inner_stride);
        std::size_t outer_loop_size = e.shape()[leading_ax];

        // The following code merges reduction axes "at the end" (or the beginning for col_major)
        // together by increasing the size of the outer loop where appropriate
        auto merge_loops = [&outer_loop_size, &e](auto it, auto end)
        {
            auto last_ax = *it;
            ++it;
            for (; it != end; ++it)
            {
                // note that we check is_sorted, so this condition is valid
                if (std::abs(std::ptrdiff_t(*it) - std::ptrdiff_t(last_ax)) == 1)
                {
                    last_ax = *it;
                    outer_loop_size *= e.shape()[last_ax];
                }
            }
            return last_ax;
        };

        for (std::size_t i = 0, idx = 0; i < e.dimension(); ++i)
        {
            if (std::find(axes.begin(), axes.end(), i) == axes.end())
            {
                // i not in axes!
                iter_strides[i] = static_cast<std::size_t>(result.strides(
                )[typename options_t::keep_dims() ? i : idx]);
                ++idx;
            }
        }

        if (e.layout() == layout_type::row_major)
        {
            std::size_t last_ax = merge_loops(axes.rbegin(), axes.rend());

            iter_shape.erase(iter_shape.begin() + std::ptrdiff_t(last_ax), iter_shape.end());
            iter_strides.erase(iter_strides.begin() + std::ptrdiff_t(last_ax), iter_strides.end());
        }
        else if (e.layout() == layout_type::column_major)
        {
            // we got column_major here
            std::size_t last_ax = merge_loops(axes.begin(), axes.end());

            // erasing the front vs the back
            iter_shape.erase(iter_shape.begin(), iter_shape.begin() + std::ptrdiff_t(last_ax + 1));
            iter_strides.erase(iter_strides.begin(), iter_strides.begin() + std::ptrdiff_t(last_ax + 1));

            // and reversing, to make it work with the same next_idx function
            std::reverse(iter_shape.begin(), iter_shape.end());
            std::reverse(iter_strides.begin(), iter_strides.end());
        }
        else
        {
            XTENSOR_THROW(std::runtime_error, "Layout not supported in immediate reduction.");
        }

        xindex temp_idx(iter_shape.size());
        auto next_idx = [&iter_shape, &iter_strides, &temp_idx]()
        {
            std::size_t i = iter_shape.size();
            for (; i > 0; --i)
            {
                if (std::ptrdiff_t(temp_idx[i - 1]) >= std::ptrdiff_t(iter_shape[i - 1]) - 1)
                {
                    temp_idx[i - 1] = 0;
                }
                else
                {
                    temp_idx[i - 1]++;
                    break;
                }
            }

            return std::make_pair(
                i == 0,
                std::inner_product(temp_idx.begin(), temp_idx.end(), iter_strides.begin(), std::ptrdiff_t(0))
            );
        };

        auto begin = e.data();
        auto out = result.data();
        auto out_begin = result.data();

        std::ptrdiff_t next_stride = 0;

        std::pair<bool, std::ptrdiff_t> idx_res(false, 0);

        // Remark: eventually some modifications here to make conditions faster where merge + accumulate is
        // the same function (e.g. check std::is_same<decltype(merge_fct), decltype(reduce_fct)>::value) ...

        auto merge_border = out;
        bool merge = false;

        // TODO there could be some performance gain by removing merge checking
        //      when axes.size() == 1 and even next_idx could be removed for something simpler (next_stride
        //      always the same) best way to do this would be to create a function that takes (begin, out,
        //      outer_loop_size, inner_loop_size, next_idx_lambda)
        // Decide if going about it row-wise or col-wise
        if (inner_stride == 1)
        {
            while (idx_res.first != true)
            {
                // for unknown reasons it's much faster to use a temporary variable and
                // std::accumulate here -- probably some cache behavior
                result_type tmp = init_fct();
                tmp = std::accumulate(begin, begin + outer_loop_size, tmp, reduce_fct);

                // use merge function if necessary
                *out = merge ? merge_fct(*out, tmp) : tmp;

                begin += outer_loop_size;

                idx_res = next_idx();
                next_stride = idx_res.second;
                out = out_begin + next_stride;

                if (out > merge_border)
                {
                    // looped over once
                    merge = false;
                    merge_border = out;
                }
                else
                {
                    merge = true;
                }
            };
        }
        else
        {
            while (idx_res.first != true)
            {
                std::transform(
                    out,
                    out + inner_loop_size,
                    begin,
                    out,
                    [merge, &init_fct, &reduce_fct](auto&& v1, auto&& v2)
                    {
                        return merge ? reduce_fct(v1, v2) :
                                     // cast because return type of identity function is not upcasted
                                   reduce_fct(static_cast<result_type>(init_fct()), v2);
                    }
                );

                begin += inner_stride;
                for (std::size_t i = 1; i < outer_loop_size; ++i)
                {
                    std::transform(out, out + inner_loop_size, begin, out, reduce_fct);
                    begin += inner_stride;
                }

                idx_res = next_idx();
                next_stride = idx_res.second;
                out = out_begin + next_stride;

                if (out > merge_border)
                {
                    // looped over once
                    merge = false;
                    merge_border = out;
                }
                else
                {
                    merge = true;
                }
            };
        }
        if (options_t::has_initial_value)
        {
            std::transform(
                result.data(),
                result.data() + result.size(),
                result.data(),
                [&merge_fct, &options](auto&& v)
                {
                    return merge_fct(v, options.initial_value);
                }
            );
        }
        return result;
    }

    /*********************
     * xreducer functors *
     *********************/

    template <class T>
    struct const_value
    {
        using value_type = T;

        constexpr const_value() = default;

        constexpr const_value(T t)
            : m_value(t)
        {
        }

        constexpr T operator()() const
        {
            return m_value;
        }

        template <class NT>
        using rebind_t = const_value<NT>;

        template <class NT>
        const_value<NT> rebind() const;

        T m_value;
    };

    namespace detail
    {
        template <class T, bool B>
        struct evaluated_value_type
        {
            using type = T;
        };

        template <class T>
        struct evaluated_value_type<T, true>
        {
            using type = typename std::decay_t<decltype(xt::eval(std::declval<T>()))>;
        };

        template <class T, bool B>
        using evaluated_value_type_t = typename evaluated_value_type<T, B>::type;
    }

    template <class REDUCE_FUNC, class INIT_FUNC = const_value<long int>, class MERGE_FUNC = REDUCE_FUNC>
    struct xreducer_functors : public std::tuple<REDUCE_FUNC, INIT_FUNC, MERGE_FUNC>
    {
        using self_type = xreducer_functors<REDUCE_FUNC, INIT_FUNC, MERGE_FUNC>;
        using base_type = std::tuple<REDUCE_FUNC, INIT_FUNC, MERGE_FUNC>;
        using reduce_functor_type = REDUCE_FUNC;
        using init_functor_type = INIT_FUNC;
        using merge_functor_type = MERGE_FUNC;
        using init_value_type = typename init_functor_type::value_type;

        xreducer_functors()
            : base_type()
        {
        }

        template <class RF>
        xreducer_functors(RF&& reduce_func)
            : base_type(std::forward<RF>(reduce_func), INIT_FUNC(), reduce_func)
        {
        }

        template <class RF, class IF>
        xreducer_functors(RF&& reduce_func, IF&& init_func)
            : base_type(std::forward<RF>(reduce_func), std::forward<IF>(init_func), reduce_func)
        {
        }

        template <class RF, class IF, class MF>
        xreducer_functors(RF&& reduce_func, IF&& init_func, MF&& merge_func)
            : base_type(std::forward<RF>(reduce_func), std::forward<IF>(init_func), std::forward<MF>(merge_func))
        {
        }

        reduce_functor_type get_reduce() const
        {
            return std::get<0>(upcast());
        }

        init_functor_type get_init() const
        {
            return std::get<1>(upcast());
        }

        merge_functor_type get_merge() const
        {
            return std::get<2>(upcast());
        }

        template <class NT>
        using rebind_t = xreducer_functors<REDUCE_FUNC, const_value<NT>, MERGE_FUNC>;

        template <class NT>
        rebind_t<NT> rebind()
        {
            return make_xreducer_functor(get_reduce(), get_init().template rebind<NT>(), get_merge());
        }

    private:

        // Workaround for clang-cl
        const base_type& upcast() const
        {
            return static_cast<const base_type&>(*this);
        }
    };

    template <class RF>
    auto make_xreducer_functor(RF&& reduce_func)
    {
        using reducer_type = xreducer_functors<std::remove_reference_t<RF>>;
        return reducer_type(std::forward<RF>(reduce_func));
    }

    template <class RF, class IF>
    auto make_xreducer_functor(RF&& reduce_func, IF&& init_func)
    {
        using reducer_type = xreducer_functors<std::remove_reference_t<RF>, std::remove_reference_t<IF>>;
        return reducer_type(std::forward<RF>(reduce_func), std::forward<IF>(init_func));
    }

    template <class RF, class IF, class MF>
    auto make_xreducer_functor(RF&& reduce_func, IF&& init_func, MF&& merge_func)
    {
        using reducer_type = xreducer_functors<
            std::remove_reference_t<RF>,
            std::remove_reference_t<IF>,
            std::remove_reference_t<MF>>;
        return reducer_type(
            std::forward<RF>(reduce_func),
            std::forward<IF>(init_func),
            std::forward<MF>(merge_func)
        );
    }

    /**********************
     * xreducer extension *
     **********************/

    namespace extension
    {
        template <class Tag, class F, class CT, class X, class O>
        struct xreducer_base_impl;

        template <class F, class CT, class X, class O>
        struct xreducer_base_impl<xtensor_expression_tag, F, CT, X, O>
        {
            using type = xtensor_empty_base;
        };

        template <class F, class CT, class X, class O>
        struct xreducer_base : xreducer_base_impl<xexpression_tag_t<CT>, F, CT, X, O>
        {
        };

        template <class F, class CT, class X, class O>
        using xreducer_base_t = typename xreducer_base<F, CT, X, O>::type;
    }

    /************
     * xreducer *
     ************/

    template <class F, class CT, class X, class O>
    class xreducer;

    template <class F, class CT, class X, class O>
    class xreducer_stepper;

    template <class F, class CT, class X, class O>
    struct xiterable_inner_types<xreducer<F, CT, X, O>>
    {
        using xexpression_type = std::decay_t<CT>;
        using inner_shape_type = typename xreducer_shape_type<
            typename xexpression_type::shape_type,
            std::decay_t<X>,
            typename O::keep_dims>::type;
        using const_stepper = xreducer_stepper<F, CT, X, O>;
        using stepper = const_stepper;
    };

    template <class F, class CT, class X, class O>
    struct xcontainer_inner_types<xreducer<F, CT, X, O>>
    {
        using xexpression_type = std::decay_t<CT>;
        using reduce_functor_type = typename std::decay_t<F>::reduce_functor_type;
        using init_functor_type = typename std::decay_t<F>::init_functor_type;
        using merge_functor_type = typename std::decay_t<F>::merge_functor_type;
        using substepper_type = typename xexpression_type::const_stepper;
        using raw_value_type = std::decay_t<decltype(std::declval<reduce_functor_type>()(
            std::declval<init_functor_type>()(),
            *std::declval<substepper_type>()
        ))>;
        using value_type = typename detail::evaluated_value_type_t<raw_value_type, is_xexpression<raw_value_type>::value>;

        using reference = value_type;
        using const_reference = value_type;
        using size_type = typename xexpression_type::size_type;
    };

    template <class T>
    struct select_dim_mapping_type
    {
        using type = T;
    };

    template <std::size_t... I>
    struct select_dim_mapping_type<fixed_shape<I...>>
    {
        using type = std::array<std::size_t, sizeof...(I)>;
    };

    /**
     * @class xreducer
     * @brief Reducing function operating over specified axes.
     *
     * The xreducer class implements an \ref xexpression applying
     * a reducing function to an \ref xexpression over the specified
     * axes.
     *
     * @tparam F a tuple of functors (class \ref xreducer_functors or compatible)
     * @tparam CT the closure type of the \ref xexpression to reduce
     * @tparam X the list of axes
     *
     * The reducer's result_type is deduced from the result type of function
     * <tt>F::reduce_functor_type</tt> when called with elements of the expression @tparam CT.
     *
     * @sa reduce
     */
    template <class F, class CT, class X, class O>
    class xreducer : public xsharable_expression<xreducer<F, CT, X, O>>,
                     public xconst_iterable<xreducer<F, CT, X, O>>,
                     public xaccessible<xreducer<F, CT, X, O>>,
                     public extension::xreducer_base_t<F, CT, X, O>
    {
    public:

        using self_type = xreducer<F, CT, X, O>;
        using inner_types = xcontainer_inner_types<self_type>;

        using reduce_functor_type = typename inner_types::reduce_functor_type;
        using init_functor_type = typename inner_types::init_functor_type;
        using merge_functor_type = typename inner_types::merge_functor_type;
        using xreducer_functors_type = xreducer_functors<reduce_functor_type, init_functor_type, merge_functor_type>;

        using xexpression_type = typename inner_types::xexpression_type;
        using axes_type = X;

        using extension_base = extension::xreducer_base_t<F, CT, X, O>;
        using expression_tag = typename extension_base::expression_tag;

        using substepper_type = typename inner_types::substepper_type;
        using value_type = typename inner_types::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = value_type*;
        using const_pointer = const value_type*;

        using size_type = typename inner_types::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using iterable_base = xconst_iterable<self_type>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using shape_type = inner_shape_type;

        using dim_mapping_type = typename select_dim_mapping_type<inner_shape_type>::type;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;
        using bool_load_type = typename xexpression_type::bool_load_type;

        static constexpr layout_type static_layout = layout_type::dynamic;
        static constexpr bool contiguous_layout = false;

        template <class Func, class CTA, class AX, class OX>
        xreducer(Func&& func, CTA&& e, AX&& axes, OX&& options);

        const inner_shape_type& shape() const noexcept;
        layout_type layout() const noexcept;
        bool is_contiguous() const noexcept;

        template <class... Args>
        const_reference operator()(Args... args) const;
        template <class... Args>
        const_reference unchecked(Args... args) const;

        template <class It>
        const_reference element(It first, It last) const;

        const xexpression_type& expression() const noexcept;

        template <class S>
        bool broadcast_shape(S& shape, bool reuse_cache = false) const;

        template <class S>
        bool has_linear_assign(const S& strides) const noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape, layout_type) const noexcept;

        template <class E, class Func = F, class Opts = O>
        using rebind_t = xreducer<Func, E, X, Opts>;

        template <class E>
        rebind_t<E> build_reducer(E&& e) const;

        template <class E, class Func, class Opts>
        rebind_t<E, Func, Opts> build_reducer(E&& e, Func&& func, Opts&& opts) const;

        xreducer_functors_type functors() const
        {
            return xreducer_functors_type(m_reduce, m_init, m_merge);  // TODO: understand why
                                                                       // make_xreducer_functor is throwing an
                                                                       // error
        }

        const O& options() const
        {
            return m_options;
        }

    private:

        CT m_e;
        reduce_functor_type m_reduce;
        init_functor_type m_init;
        merge_functor_type m_merge;
        axes_type m_axes;
        inner_shape_type m_shape;
        dim_mapping_type m_dim_mapping;
        O m_options;

        friend class xreducer_stepper<F, CT, X, O>;
    };

    /*************************
     * reduce implementation *
     *************************/

    namespace detail
    {
        template <class F, class E, class X, class O>
        inline auto reduce_impl(F&& f, E&& e, X&& axes, evaluation_strategy::lazy_type, O&& options)
        {
            decltype(auto) normalized_axes = normalize_axis(e, std::forward<X>(axes));

            using reduce_functor_type = typename std::decay_t<F>::reduce_functor_type;
            using init_functor_type = typename std::decay_t<F>::init_functor_type;
            using value_type = std::decay_t<decltype(std::declval<reduce_functor_type>()(
                std::declval<init_functor_type>()(),
                *std::declval<typename std::decay_t<E>::const_stepper>()
            ))>;
            using evaluated_value_type = evaluated_value_type_t<value_type, is_xexpression<value_type>::value>;

            using reducer_type = xreducer<
                F,
                const_xclosure_t<E>,
                xtl::const_closure_type_t<decltype(normalized_axes)>,
                reducer_options<evaluated_value_type, std::decay_t<O>>>;
            return reducer_type(
                std::forward<F>(f),
                std::forward<E>(e),
                std::forward<decltype(normalized_axes)>(normalized_axes),
                std::forward<O>(options)
            );
        }

        template <class F, class E, class X, class O>
        inline auto reduce_impl(F&& f, E&& e, X&& axes, evaluation_strategy::immediate_type, O&& options)
        {
            decltype(auto) normalized_axes = normalize_axis(e, std::forward<X>(axes));
            return reduce_immediate(
                std::forward<F>(f),
                eval(std::forward<E>(e)),
                std::forward<decltype(normalized_axes)>(normalized_axes),
                std::forward<O>(options)
            );
        }
    }

#define DEFAULT_STRATEGY_REDUCERS std::tuple<evaluation_strategy::lazy_type>

    namespace detail
    {
        template <class T>
        struct is_xreducer_functors_impl : std::false_type
        {
        };

        template <class RF, class IF, class MF>
        struct is_xreducer_functors_impl<xreducer_functors<RF, IF, MF>> : std::true_type
        {
        };

        template <class T>
        using is_xreducer_functors = is_xreducer_functors_impl<std::decay_t<T>>;
    }

    /**
     * @brief Returns an \ref xexpression applying the specified reducing
     * function to an expression over the given axes.
     *
     * @param f the reducing function to apply.
     * @param e the \ref xexpression to reduce.
     * @param axes the list of axes.
     * @param options evaluation strategy to use (lazy (default), or immediate)
     *
     * The returned expression either hold a const reference to \p e or a copy
     * depending on whether \p e is an lvalue or an rvalue.
     */

    template <
        class F,
        class E,
        class X,
        class EVS = DEFAULT_STRATEGY_REDUCERS,
        XTL_REQUIRES(std::negation<is_reducer_options<X>>, detail::is_xreducer_functors<F>)>
    inline auto reduce(F&& f, E&& e, X&& axes, EVS&& options = EVS())
    {
        return detail::reduce_impl(
            std::forward<F>(f),
            std::forward<E>(e),
            std::forward<X>(axes),
            typename reducer_options<int, EVS>::evaluation_strategy{},
            std::forward<EVS>(options)
        );
    }

    template <
        class F,
        class E,
        class X,
        class EVS = DEFAULT_STRATEGY_REDUCERS,
        XTL_REQUIRES(std::negation<is_reducer_options<X>>, std::negation<detail::is_xreducer_functors<F>>)>
    inline auto reduce(F&& f, E&& e, X&& axes, EVS&& options = EVS())
    {
        return reduce(
            make_xreducer_functor(std::forward<F>(f)),
            std::forward<E>(e),
            std::forward<X>(axes),
            std::forward<EVS>(options)
        );
    }

    template <
        class F,
        class E,
        class EVS = DEFAULT_STRATEGY_REDUCERS,
        XTL_REQUIRES(is_reducer_options<EVS>, detail::is_xreducer_functors<F>)>
    inline auto reduce(F&& f, E&& e, EVS&& options = EVS())
    {
        xindex_type_t<typename std::decay_t<E>::shape_type> ar;
        resize_container(ar, e.dimension());
        std::iota(ar.begin(), ar.end(), 0);
        return detail::reduce_impl(
            std::forward<F>(f),
            std::forward<E>(e),
            std::move(ar),
            typename reducer_options<int, std::decay_t<EVS>>::evaluation_strategy{},
            std::forward<EVS>(options)
        );
    }

    template <
        class F,
        class E,
        class EVS = DEFAULT_STRATEGY_REDUCERS,
        XTL_REQUIRES(is_reducer_options<EVS>, std::negation<detail::is_xreducer_functors<F>>)>
    inline auto reduce(F&& f, E&& e, EVS&& options = EVS())
    {
        return reduce(make_xreducer_functor(std::forward<F>(f)), std::forward<E>(e), std::forward<EVS>(options));
    }

    template <
        class F,
        class E,
        class I,
        std::size_t N,
        class EVS = DEFAULT_STRATEGY_REDUCERS,
        XTL_REQUIRES(detail::is_xreducer_functors<F>)>
    inline auto reduce(F&& f, E&& e, const I (&axes)[N], EVS options = EVS())
    {
        using axes_type = std::array<std::size_t, N>;
        auto ax = xt::forward_normalize<axes_type>(e, axes);
        return detail::reduce_impl(
            std::forward<F>(f),
            std::forward<E>(e),
            std::move(ax),
            typename reducer_options<int, EVS>::evaluation_strategy{},
            options
        );
    }

    template <
        class F,
        class E,
        class I,
        std::size_t N,
        class EVS = DEFAULT_STRATEGY_REDUCERS,
        XTL_REQUIRES(std::negation<detail::is_xreducer_functors<F>>)>
    inline auto reduce(F&& f, E&& e, const I (&axes)[N], EVS options = EVS())
    {
        return reduce(make_xreducer_functor(std::forward<F>(f)), std::forward<E>(e), axes, options);
    }

    /********************
     * xreducer_stepper *
     ********************/

    template <class F, class CT, class X, class O>
    class xreducer_stepper
    {
    public:

        using self_type = xreducer_stepper<F, CT, X, O>;
        using xreducer_type = xreducer<F, CT, X, O>;

        using value_type = typename xreducer_type::value_type;
        using reference = typename xreducer_type::value_type;
        using pointer = typename xreducer_type::const_pointer;
        using size_type = typename xreducer_type::size_type;
        using difference_type = typename xreducer_type::difference_type;

        using xexpression_type = typename xreducer_type::xexpression_type;
        using substepper_type = typename xexpression_type::const_stepper;
        using shape_type = typename xreducer_type::shape_type;

        xreducer_stepper(
            const xreducer_type& red,
            size_type offset,
            bool end = false,
            layout_type l = default_assignable_layout(xexpression_type::static_layout)
        );

        reference operator*() const;

        void step(size_type dim);
        void step_back(size_type dim);
        void step(size_type dim, size_type n);
        void step_back(size_type dim, size_type n);
        void reset(size_type dim);
        void reset_back(size_type dim);

        void to_begin();
        void to_end(layout_type l);

    private:

        reference initial_value() const;
        reference aggregate(size_type dim) const;
        reference aggregate_impl(size_type dim, /*keep_dims=*/std::false_type) const;
        reference aggregate_impl(size_type dim, /*keep_dims=*/std::true_type) const;

        substepper_type get_substepper_begin() const;
        size_type get_dim(size_type dim) const noexcept;
        size_type shape(size_type i) const noexcept;
        size_type axis(size_type i) const noexcept;

        const xreducer_type* m_reducer;
        size_type m_offset;
        mutable substepper_type m_stepper;
    };

    /******************
     * xreducer utils *
     ******************/

    namespace detail
    {
        template <std::size_t X, std::size_t... I>
        struct in
        {
            static constexpr bool value = std::disjunction<std::integral_constant<bool, X == I>...>::value;
        };

        template <std::size_t Z, class S1, class S2, class R>
        struct fixed_xreducer_shape_type_impl;

        template <std::size_t Z, std::size_t... I, std::size_t... J, std::size_t... R>
        struct fixed_xreducer_shape_type_impl<Z, fixed_shape<I...>, fixed_shape<J...>, fixed_shape<R...>>
        {
            using type = std::conditional_t<
                in<Z, J...>::value,
                typename fixed_xreducer_shape_type_impl<Z - 1, fixed_shape<I...>, fixed_shape<J...>, fixed_shape<R...>>::type,
                typename fixed_xreducer_shape_type_impl<
                    Z - 1,
                    fixed_shape<I...>,
                    fixed_shape<J...>,
                    fixed_shape<detail::at<Z, I...>::value, R...>>::type>;
        };

        template <std::size_t... I, std::size_t... J, std::size_t... R>
        struct fixed_xreducer_shape_type_impl<0, fixed_shape<I...>, fixed_shape<J...>, fixed_shape<R...>>
        {
            using type = std::
                conditional_t<in<0, J...>::value, fixed_shape<R...>, fixed_shape<detail::at<0, I...>::value, R...>>;
        };

        /***************************
         * helper for return types *
         ***************************/

        template <class T>
        struct xreducer_size_type
        {
            using type = std::size_t;
        };

        template <class T>
        using xreducer_size_type_t = typename xreducer_size_type<T>::type;

        template <class T>
        struct xreducer_temporary_type
        {
            using type = T;
        };

        template <class T>
        using xreducer_temporary_type_t = typename xreducer_temporary_type<T>::type;

        /********************************
         * Default const_value rebinder *
         ********************************/

        template <class T, class U>
        struct const_value_rebinder
        {
            static const_value<U> run(const const_value<T>& t)
            {
                return const_value<U>(t.m_value);
            }
        };
    }

    /*******************************************
     * Init functor const_value implementation *
     *******************************************/

    template <class T>
    template <class NT>
    const_value<NT> const_value<T>::rebind() const
    {
        return detail::const_value_rebinder<T, NT>::run(*this);
    }

    /*****************************
     * fixed_xreducer_shape_type *
     *****************************/

    template <class S1, class S2>
    struct fixed_xreducer_shape_type;

    template <std::size_t... I, std::size_t... J>
    struct fixed_xreducer_shape_type<fixed_shape<I...>, fixed_shape<J...>>
    {
        using type = typename detail::
            fixed_xreducer_shape_type_impl<sizeof...(I) - 1, fixed_shape<I...>, fixed_shape<J...>, fixed_shape<>>::type;
    };

    // meta-function returning the shape type for an xreducer
    template <class ST, class X, class O>
    struct xreducer_shape_type
    {
        using type = promote_shape_t<ST, std::decay_t<X>>;
    };

    template <class I1, std::size_t N1, class I2, std::size_t N2>
    struct xreducer_shape_type<std::array<I1, N1>, std::array<I2, N2>, std::true_type>
    {
        using type = std::array<I2, N1>;
    };

    template <class I1, std::size_t N1, class I2, std::size_t N2>
    struct xreducer_shape_type<std::array<I1, N1>, std::array<I2, N2>, std::false_type>
    {
        using type = std::array<I2, N1 - N2>;
    };

    template <std::size_t... I, class I2, std::size_t N2>
    struct xreducer_shape_type<fixed_shape<I...>, std::array<I2, N2>, std::false_type>
    {
        using type = std::conditional_t<sizeof...(I) == N2, fixed_shape<>, std::array<I2, sizeof...(I) - N2>>;
    };

    namespace detail
    {
        template <class S1, class S2>
        struct ixconcat;

        template <class T, T... I1, T... I2>
        struct ixconcat<std::integer_sequence<T, I1...>, std::integer_sequence<T, I2...>>
        {
            using type = std::integer_sequence<T, I1..., I2...>;
        };

        template <class T, T X, std::size_t N>
        struct repeat_integer_sequence
        {
            using type = typename ixconcat<
                std::integer_sequence<T, X>,
                typename repeat_integer_sequence<T, X, N - 1>::type>::type;
        };

        template <class T, T X>
        struct repeat_integer_sequence<T, X, 0>
        {
            using type = std::integer_sequence<T>;
        };

        template <class T, T X>
        struct repeat_integer_sequence<T, X, 2>
        {
            using type = std::integer_sequence<T, X, X>;
        };

        template <class T, T X>
        struct repeat_integer_sequence<T, X, 1>
        {
            using type = std::integer_sequence<T, X>;
        };
    }

    template <std::size_t... I, class I2, std::size_t N2>
    struct xreducer_shape_type<fixed_shape<I...>, std::array<I2, N2>, std::true_type>
    {
        template <std::size_t... X>
        static constexpr auto get_type(std::index_sequence<X...>)
        {
            return fixed_shape<X...>{};
        }

        // if all axes reduced
        using type = std::conditional_t<
            sizeof...(I) == N2,
            decltype(get_type(typename detail::repeat_integer_sequence<std::size_t, std::size_t(1), N2>::type{})),
            std::array<I2, sizeof...(I)>>;
    };

    // Note adding "A" to prevent compilation in case nothing else matches
    template <std::size_t... I, std::size_t... J, class O>
    struct xreducer_shape_type<fixed_shape<I...>, fixed_shape<J...>, O>
    {
        using type = typename fixed_xreducer_shape_type<fixed_shape<I...>, fixed_shape<J...>>::type;
    };

    namespace detail
    {
        template <class S, class E, class X, class M>
        inline void shape_and_mapping_computation(S& shape, E& e, const X& axes, M& mapping, std::false_type)
        {
            auto first = e.shape().begin();
            auto last = e.shape().end();
            auto exclude_it = axes.begin();

            using value_type = typename S::value_type;
            using difference_type = typename S::difference_type;
            auto d_first = shape.begin();
            auto map_first = mapping.begin();

            auto iter = first;
            while (iter != last && exclude_it != axes.end())
            {
                auto diff = std::distance(first, iter);
                if (diff != difference_type(*exclude_it))
                {
                    *d_first++ = *iter++;
                    *map_first++ = value_type(diff);
                }
                else
                {
                    ++iter;
                    ++exclude_it;
                }
            }

            auto diff = std::distance(first, iter);
            auto end = std::distance(iter, last);
            std::iota(map_first, map_first + end, diff);
            std::copy(iter, last, d_first);
        }

        template <class S, class E, class X, class M>
        inline void
        shape_and_mapping_computation_keep_dim(S& shape, E& e, const X& axes, M& mapping, std::false_type)
        {
            for (std::size_t i = 0; i < e.dimension(); ++i)
            {
                if (std::find(axes.cbegin(), axes.cend(), i) == axes.cend())
                {
                    // i not in axes!
                    shape[i] = e.shape()[i];
                }
                else
                {
                    shape[i] = 1;
                }
            }
            std::iota(mapping.begin(), mapping.end(), 0);
        }

        template <class S, class E, class X, class M>
        inline void shape_and_mapping_computation(S&, E&, const X&, M&, std::true_type)
        {
        }

        template <class S, class E, class X, class M>
        inline void shape_and_mapping_computation_keep_dim(S&, E&, const X&, M&, std::true_type)
        {
        }
    }

    /***************************
     * xreducer implementation *
     ***************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xreducer expression applying the specified
     * function to the given expression over the given axes.
     *
     * @param func the function to apply
     * @param e the expression to reduce
     * @param axes the axes along which the reduction is performed
     */
    template <class F, class CT, class X, class O>
    template <class Func, class CTA, class AX, class OX>
    inline xreducer<F, CT, X, O>::xreducer(Func&& func, CTA&& e, AX&& axes, OX&& options)
        : m_e(std::forward<CTA>(e))
        , m_reduce(xt::get<0>(func))
        , m_init(xt::get<1>(func))
        , m_merge(xt::get<2>(func))
        , m_axes(std::forward<AX>(axes))
        , m_shape(xtl::make_sequence<inner_shape_type>(
              typename O::keep_dims() ? m_e.dimension() : m_e.dimension() - m_axes.size(),
              0
          ))
        , m_dim_mapping(xtl::make_sequence<dim_mapping_type>(
              typename O::keep_dims() ? m_e.dimension() : m_e.dimension() - m_axes.size(),
              0
          ))
        , m_options(std::forward<OX>(options))
    {
        // std::less is used, because as the standard says (24.4.5):
        // A sequence is sorted with respect to a comparator comp if for any iterator i pointing to the
        // sequence and any non-negative integer n such that i + n is a valid iterator pointing to an element
        // of the sequence, comp(*(i + n), *i) == false. Therefore less is required to detect duplicates.
        if (!std::is_sorted(m_axes.cbegin(), m_axes.cend(), std::less<>()))
        {
            XTENSOR_THROW(std::runtime_error, "Reducing axes should be sorted.");
        }
        if (std::adjacent_find(m_axes.cbegin(), m_axes.cend()) != m_axes.cend())
        {
            XTENSOR_THROW(std::runtime_error, "Reducing axes should not contain duplicates.");
        }
        if (m_axes.size() != 0 && m_axes[m_axes.size() - 1] > m_e.dimension() - 1)
        {
            XTENSOR_THROW(
                std::runtime_error,
                "Axis " + std::to_string(m_axes[m_axes.size() - 1]) + " out of bounds for reduction."
            );
        }

        if (!typename O::keep_dims())
        {
            detail::shape_and_mapping_computation(
                m_shape,
                m_e,
                m_axes,
                m_dim_mapping,
                detail::is_fixed<shape_type>{}
            );
        }
        else
        {
            detail::shape_and_mapping_computation_keep_dim(
                m_shape,
                m_e,
                m_axes,
                m_dim_mapping,
                detail::is_fixed<shape_type>{}
            );
        }
    }

    //@}

    /**
     * @name Size and shape
     */

    /**
     * Returns the shape of the expression.
     */
    template <class F, class CT, class X, class O>
    inline auto xreducer<F, CT, X, O>::shape() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    /**
     * Returns the shape of the expression.
     */
    template <class F, class CT, class X, class O>
    inline layout_type xreducer<F, CT, X, O>::layout() const noexcept
    {
        return static_layout;
    }

    template <class F, class CT, class X, class O>
    inline bool xreducer<F, CT, X, O>::is_contiguous() const noexcept
    {
        return false;
    }

    //@}

    /**
     * @name Data
     */

    /**
     * Returns a constant reference to the element at the specified position in the reducer.
     * @param args a list of indices specifying the position in the reducer. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the reducer.
     */
    template <class F, class CT, class X, class O>
    template <class... Args>
    inline auto xreducer<F, CT, X, O>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        std::array<std::size_t, sizeof...(Args)> arg_array = {{static_cast<std::size_t>(args)...}};
        return element(arg_array.cbegin(), arg_array.cend());
    }

    /**
     * Returns a constant reference to the element at the specified position in the reducer.
     * @param args a list of indices specifying the position in the reducer. Indices
     * must be unsigned integers, the number of indices must be equal to the number of
     * dimensions of the reducer, else the behavior is undefined.
     *
     * @warning This method is meant for performance, for expressions with a dynamic
     * number of dimensions (i.e. not known at compile time). Since it may have
     * undefined behavior (see parameters), operator() should be preferred whenever
     * it is possible.
     * @warning This method is NOT compatible with broadcasting, meaning the following
     * code has undefined behavior:
     * @code{.cpp}
     * xt::xarray<double> a = {{0, 1}, {2, 3}};
     * xt::xarray<double> b = {0, 1};
     * auto fd = a + b;
     * double res = fd.uncheked(0, 1);
     * @endcode
     */
    template <class F, class CT, class X, class O>
    template <class... Args>
    inline auto xreducer<F, CT, X, O>::unchecked(Args... args) const -> const_reference
    {
        std::array<std::size_t, sizeof...(Args)> arg_array = {{static_cast<std::size_t>(args)...}};
        return element(arg_array.cbegin(), arg_array.cend());
    }

    /**
     * Returns a constant reference to the element at the specified position in the reducer.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the reducer.
     */
    template <class F, class CT, class X, class O>
    template <class It>
    inline auto xreducer<F, CT, X, O>::element(It first, It last) const -> const_reference
    {
        XTENSOR_TRY(check_element_index(shape(), first, last));
        auto stepper = const_stepper(*this, 0);
        if (first != last)
        {
            size_type dim = 0;
            // drop left most elements
            auto size = std::ptrdiff_t(this->dimension()) - std::distance(first, last);
            auto begin = first - size;
            while (begin != last)
            {
                if (begin < first)
                {
                    stepper.step(dim++, std::size_t(0));
                    begin++;
                }
                else
                {
                    stepper.step(dim++, std::size_t(*begin++));
                }
            }
        }
        return *stepper;
    }

    /**
     * Returns a constant reference to the underlying expression of the reducer.
     */
    template <class F, class CT, class X, class O>
    inline auto xreducer<F, CT, X, O>::expression() const noexcept -> const xexpression_type&
    {
        return m_e;
    }

    //@}

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the reducer to the specified parameter.
     * @param shape the result shape
     * @param reuse_cache parameter for internal optimization
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class F, class CT, class X, class O>
    template <class S>
    inline bool xreducer<F, CT, X, O>::broadcast_shape(S& shape, bool) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Checks whether the xreducer can be linearly assigned to an expression
     * with the specified strides.
     * @return a boolean indicating whether a linear assign is possible
     */
    template <class F, class CT, class X, class O>
    template <class S>
    inline bool xreducer<F, CT, X, O>::has_linear_assign(const S& /*strides*/) const noexcept
    {
        return false;
    }

    //@}

    template <class F, class CT, class X, class O>
    template <class S>
    inline auto xreducer<F, CT, X, O>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(*this, offset);
    }

    template <class F, class CT, class X, class O>
    template <class S>
    inline auto xreducer<F, CT, X, O>::stepper_end(const S& shape, layout_type l) const noexcept
        -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(*this, offset, true, l);
    }

    template <class F, class CT, class X, class O>
    template <class E>
    inline auto xreducer<F, CT, X, O>::build_reducer(E&& e) const -> rebind_t<E>
    {
        return rebind_t<E>(
            std::make_tuple(m_reduce, m_init, m_merge),
            std::forward<E>(e),
            axes_type(m_axes),
            m_options
        );
    }

    template <class F, class CT, class X, class O>
    template <class E, class Func, class Opts>
    inline auto xreducer<F, CT, X, O>::build_reducer(E&& e, Func&& func, Opts&& opts) const
        -> rebind_t<E, Func, Opts>
    {
        return rebind_t<E, Func, Opts>(
            std::forward<Func>(func),
            std::forward<E>(e),
            axes_type(m_axes),
            std::forward<Opts>(opts)
        );
    }

    /***********************************
     * xreducer_stepper implementation *
     ***********************************/

    template <class F, class CT, class X, class O>
    inline xreducer_stepper<F, CT, X, O>::xreducer_stepper(
        const xreducer_type& red,
        size_type offset,
        bool end,
        layout_type l
    )
        : m_reducer(&red)
        , m_offset(offset)
        , m_stepper(get_substepper_begin())
    {
        if (end)
        {
            to_end(l);
        }
    }

    template <class F, class CT, class X, class O>
    inline auto xreducer_stepper<F, CT, X, O>::operator*() const -> reference
    {
        reference r = aggregate(0);
        return r;
    }

    template <class F, class CT, class X, class O>
    inline void xreducer_stepper<F, CT, X, O>::step(size_type dim)
    {
        if (dim >= m_offset)
        {
            m_stepper.step(get_dim(dim - m_offset));
        }
    }

    template <class F, class CT, class X, class O>
    inline void xreducer_stepper<F, CT, X, O>::step_back(size_type dim)
    {
        if (dim >= m_offset)
        {
            m_stepper.step_back(get_dim(dim - m_offset));
        }
    }

    template <class F, class CT, class X, class O>
    inline void xreducer_stepper<F, CT, X, O>::step(size_type dim, size_type n)
    {
        if (dim >= m_offset)
        {
            m_stepper.step(get_dim(dim - m_offset), n);
        }
    }

    template <class F, class CT, class X, class O>
    inline void xreducer_stepper<F, CT, X, O>::step_back(size_type dim, size_type n)
    {
        if (dim >= m_offset)
        {
            m_stepper.step_back(get_dim(dim - m_offset), n);
        }
    }

    template <class F, class CT, class X, class O>
    inline void xreducer_stepper<F, CT, X, O>::reset(size_type dim)
    {
        if (dim >= m_offset)
        {
            // Because the reducer uses `reset` to reset the non-reducing axes,
            // we need to prevent that here for the KD case where.
            if (typename O::keep_dims()
                && std::binary_search(m_reducer->m_axes.begin(), m_reducer->m_axes.end(), dim - m_offset))
            {
                // If keep dim activated, and dim is in the axes, do nothing!
                return;
            }
            m_stepper.reset(get_dim(dim - m_offset));
        }
    }

    template <class F, class CT, class X, class O>
    inline void xreducer_stepper<F, CT, X, O>::reset_back(size_type dim)
    {
        if (dim >= m_offset)
        {
            // Note that for *not* KD this is not going to do anything
            if (typename O::keep_dims()
                && std::binary_search(m_reducer->m_axes.begin(), m_reducer->m_axes.end(), dim - m_offset))
            {
                // If keep dim activated, and dim is in the axes, do nothing!
                return;
            }
            m_stepper.reset_back(get_dim(dim - m_offset));
        }
    }

    template <class F, class CT, class X, class O>
    inline void xreducer_stepper<F, CT, X, O>::to_begin()
    {
        m_stepper.to_begin();
    }

    template <class F, class CT, class X, class O>
    inline void xreducer_stepper<F, CT, X, O>::to_end(layout_type l)
    {
        m_stepper.to_end(l);
    }

    template <class F, class CT, class X, class O>
    inline auto xreducer_stepper<F, CT, X, O>::initial_value() const -> reference
    {
        return O::has_initial_value ? m_reducer->m_options.initial_value
                                    : static_cast<reference>(m_reducer->m_init());
    }

    template <class F, class CT, class X, class O>
    inline auto xreducer_stepper<F, CT, X, O>::aggregate(size_type dim) const -> reference
    {
        reference res;
        if (m_reducer->m_e.size() == size_type(0))
        {
            res = initial_value();
        }
        else if (m_reducer->m_e.shape().empty() || m_reducer->m_axes.size() == 0)
        {
            res = m_reducer->m_reduce(initial_value(), *m_stepper);
        }
        else
        {
            res = aggregate_impl(dim, typename O::keep_dims());
            if (O::has_initial_value && dim == 0)
            {
                res = m_reducer->m_merge(m_reducer->m_options.initial_value, res);
            }
        }
        return res;
    }

    template <class F, class CT, class X, class O>
    inline auto xreducer_stepper<F, CT, X, O>::aggregate_impl(size_type dim, std::false_type) const -> reference
    {
        // reference can be std::array, hence the {} initializer
        reference res = {};
        size_type index = axis(dim);
        size_type size = shape(index);
        if (dim != m_reducer->m_axes.size() - 1)
        {
            res = aggregate_impl(dim + 1, typename O::keep_dims());
            for (size_type i = 1; i != size; ++i)
            {
                m_stepper.step(index);
                res = m_reducer->m_merge(res, aggregate_impl(dim + 1, typename O::keep_dims()));
            }
        }
        else
        {
            res = m_reducer->m_reduce(static_cast<reference>(m_reducer->m_init()), *m_stepper);
            for (size_type i = 1; i != size; ++i)
            {
                m_stepper.step(index);
                res = m_reducer->m_reduce(res, *m_stepper);
            }
        }
        m_stepper.reset(index);
        return res;
    }

    template <class F, class CT, class X, class O>
    inline auto xreducer_stepper<F, CT, X, O>::aggregate_impl(size_type dim, std::true_type) const -> reference
    {
        // reference can be std::array, hence the {} initializer
        reference res = {};
        auto ax_it = std::find(m_reducer->m_axes.begin(), m_reducer->m_axes.end(), dim);
        if (ax_it != m_reducer->m_axes.end())
        {
            size_type index = dim;
            size_type size = m_reducer->m_e.shape()[index];
            if (ax_it != m_reducer->m_axes.end() - 1 && size != 0)
            {
                res = aggregate_impl(dim + 1, typename O::keep_dims());
                for (size_type i = 1; i != size; ++i)
                {
                    m_stepper.step(index);
                    res = m_reducer->m_merge(res, aggregate_impl(dim + 1, typename O::keep_dims()));
                }
            }
            else
            {
                res = m_reducer->m_reduce(static_cast<reference>(m_reducer->m_init()), *m_stepper);
                for (size_type i = 1; i != size; ++i)
                {
                    m_stepper.step(index);
                    res = m_reducer->m_reduce(res, *m_stepper);
                }
            }
            m_stepper.reset(index);
        }
        else
        {
            if (dim < m_reducer->m_e.dimension())
            {
                res = aggregate_impl(dim + 1, typename O::keep_dims());
            }
        }
        return res;
    }

    template <class F, class CT, class X, class O>
    inline auto xreducer_stepper<F, CT, X, O>::get_substepper_begin() const -> substepper_type
    {
        return m_reducer->m_e.stepper_begin(m_reducer->m_e.shape());
    }

    template <class F, class CT, class X, class O>
    inline auto xreducer_stepper<F, CT, X, O>::get_dim(size_type dim) const noexcept -> size_type
    {
        return m_reducer->m_dim_mapping[dim];
    }

    template <class F, class CT, class X, class O>
    inline auto xreducer_stepper<F, CT, X, O>::shape(size_type i) const noexcept -> size_type
    {
        return m_reducer->m_e.shape()[i];
    }

    template <class F, class CT, class X, class O>
    inline auto xreducer_stepper<F, CT, X, O>::axis(size_type i) const noexcept -> size_type
    {
        return m_reducer->m_axes[i];
    }
}

#endif
