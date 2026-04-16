#ifndef XTENSOR_XBLOCKWISE_REDUCER_FUNCTORS_HPP
#define XTENSOR_XBLOCKWISE_REDUCER_FUNCTORS_HPP


#include <sstream>
#include <string>
#include <tuple>
#include <typeinfo>

#include "../chunk/xchunked_array.hpp"
#include "../chunk/xchunked_assign.hpp"
#include "../chunk/xchunked_view.hpp"
#include "../containers/xarray.hpp"
#include "../core/xexpression.hpp"
#include "../core/xmath.hpp"
#include "../generators/xbuilder.hpp"
#include "../reducers/xnorm.hpp"
#include "../reducers/xreducer.hpp"
#include "../utils/xutils.hpp"
#include "xtl/xclosure.hpp"
#include "xtl/xsequence.hpp"

namespace xt
{
    namespace detail
    {
        namespace blockwise
        {

            struct empty_reduction_variable
            {
            };

            struct simple_functor_base
            {
                template <class E>
                auto reduction_variable(const E&) const
                {
                    return empty_reduction_variable();
                }

                template <class MR, class E, class R>
                void finalize(const MR&, E&, const R&) const
                {
                }
            };

            template <class T_E, class T_I = void>
            struct sum_functor : public simple_functor_base
            {
                using value_type = typename std::decay_t<decltype(xt::sum<T_I>(std::declval<xarray<T_E>>()))>::value_type;

                template <class E, class A, class O>
                auto compute(const E& input, const A& axes, const O& options) const
                {
                    return xt::sum<value_type>(input, axes, options);
                }

                template <class BR, class E, class MR>
                auto merge(const BR& block_result, bool first, E& result, MR&) const
                {
                    if (first)
                    {
                        xt::noalias(result) = block_result;
                    }
                    else
                    {
                        xt::noalias(result) += block_result;
                    }
                }
            };

            template <class T_E, class T_I = void>
            struct prod_functor : public simple_functor_base
            {
                using value_type = typename std::decay_t<decltype(xt::sum<T_I>(std::declval<xarray<T_E>>()))>::value_type;

                template <class E, class A, class O>
                auto compute(const E& input, const A& axes, const O& options) const
                {
                    return xt::prod<value_type>(input, axes, options);
                }

                template <class BR, class E, class MR>
                auto merge(const BR& block_result, bool first, E& result, MR&) const
                {
                    if (first)
                    {
                        xt::noalias(result) = block_result;
                    }
                    else
                    {
                        xt::noalias(result) *= block_result;
                    }
                }
            };

            template <class T_E, class T_I = void>
            struct amin_functor : public simple_functor_base
            {
                using value_type = typename std::decay_t<decltype(xt::amin<T_I>(std::declval<xarray<T_E>>()))>::value_type;

                template <class E, class A, class O>
                auto compute(const E& input, const A& axes, const O& options) const
                {
                    return xt::amin(input, axes, options);
                }

                template <class BR, class E, class MR>
                auto merge(const BR& block_result, bool first, E& result, MR&) const
                {
                    if (first)
                    {
                        xt::noalias(result) = block_result;
                    }
                    else
                    {
                        xt::noalias(result) = xt::minimum(block_result, result);
                    }
                }
            };

            template <class T_E, class T_I = void>
            struct amax_functor : public simple_functor_base
            {
                using value_type = typename std::decay_t<decltype(xt::amax<T_I>(std::declval<xarray<T_E>>()))>::value_type;

                template <class E, class A, class O>
                auto compute(const E& input, const A& axes, const O& options) const
                {
                    return xt::amax(input, axes, options);
                }

                template <class BR, class E, class MR>
                auto merge(const BR& block_result, bool first, E& result, MR&) const
                {
                    if (first)
                    {
                        xt::noalias(result) = block_result;
                    }
                    else
                    {
                        xt::noalias(result) = xt::maximum(block_result, result);
                    }
                }
            };

            template <class T_E, class T_I = void>
            struct mean_functor
            {
                using value_type = typename std::decay_t<decltype(xt::mean<T_I>(std::declval<xarray<T_E>>()))>::value_type;

                template <class E, class A, class O>
                auto compute(const E& input, const A& axes, const O& options) const
                {
                    return xt::sum<value_type>(input, axes, options);
                }

                template <class E>
                auto reduction_variable(const E&) const
                {
                    return empty_reduction_variable();
                }

                template <class BR, class E>
                auto merge(const BR& block_result, bool first, E& result, empty_reduction_variable&) const
                {
                    if (first)
                    {
                        xt::noalias(result) = block_result;
                    }
                    else
                    {
                        xt::noalias(result) += block_result;
                    }
                }

                template <class E, class R>
                void finalize(const empty_reduction_variable&, E& results, const R& reducer) const
                {
                    const auto& axes = reducer.axes();
                    std::decay_t<decltype(reducer.input_shape()[0])> factor = 1;
                    for (auto a : axes)
                    {
                        factor *= reducer.input_shape()[a];
                    }
                    xt::noalias(results) /= static_cast<typename E::value_type>(factor);
                }
            };

            template <class T_E, class T_I = void>
            struct variance_functor
            {
                using value_type = typename std::decay_t<decltype(xt::variance<T_I>(std::declval<xarray<T_E>>())
                )>::value_type;

                template <class E, class A, class O>
                auto compute(const E& input, const A& axes, const O& options) const
                {
                    double weight = 1.0;
                    for (auto a : axes)
                    {
                        weight *= static_cast<double>(input.shape()[a]);
                    }


                    return std::make_tuple(
                        xt::variance<value_type>(input, axes, options),
                        xt::mean<value_type>(input, axes, options),
                        weight
                    );
                }

                template <class E>
                auto reduction_variable(const E&) const
                {
                    return std::make_tuple(xarray<value_type>(), 0.0);
                }

                template <class BR, class E, class MR>
                auto merge(const BR& block_result, bool first, E& variance_a, MR& mr) const
                {
                    auto& mean_a = std::get<0>(mr);
                    auto& n_a = std::get<1>(mr);

                    const auto& variance_b = std::get<0>(block_result);
                    const auto& mean_b = std::get<1>(block_result);
                    const auto& n_b = std::get<2>(block_result);
                    if (first)
                    {
                        xt::noalias(variance_a) = variance_b;
                        xt::noalias(mean_a) = mean_b;
                        n_a += n_b;
                    }
                    else
                    {
                        auto new_mean = (n_a * mean_a + n_b * mean_b) / (n_a + n_b);
                        auto new_variance = (n_a * variance_a + n_b * variance_b
                                             + n_a * xt::pow(mean_a - new_mean, 2)
                                             + n_b * xt::pow(mean_b - new_mean, 2))
                                            / (n_a + n_b);
                        xt::noalias(variance_a) = new_variance;
                        xt::noalias(mean_a) = new_mean;
                        n_a += n_b;
                    }
                }

                template <class MR, class E, class R>
                void finalize(const MR&, E&, const R&) const
                {
                }
            };

            template <class T_E, class T_I = void>
            struct stddev_functor : public variance_functor<T_E, T_I>
            {
                template <class MR, class E, class R>
                void finalize(const MR&, E& results, const R&) const
                {
                    xt::noalias(results) = xt::sqrt(results);
                }
            };

            template <class T_E>
            struct norm_l0_functor : public simple_functor_base
            {
                using value_type = typename std::decay_t<decltype(xt::norm_l0(std::declval<xarray<T_E>>()))>::value_type;

                template <class E, class A, class O>
                auto compute(const E& input, const A& axes, const O& options) const
                {
                    return xt::sum<value_type>(xt::not_equal(input, xt::zeros<T_E>(input.shape())), axes, options);
                }

                template <class BR, class E, class MR>
                auto merge(const BR& block_result, bool first, E& result, MR&) const
                {
                    if (first)
                    {
                        xt::noalias(result) = block_result;
                    }
                    else
                    {
                        xt::noalias(result) += block_result;
                    }
                }
            };

            template <class T_E>
            struct norm_l1_functor : public simple_functor_base
            {
                using value_type = typename std::decay_t<decltype(xt::norm_l1(std::declval<xarray<T_E>>()))>::value_type;

                template <class E, class A, class O>
                auto compute(const E& input, const A& axes, const O& options) const
                {
                    return xt::sum<value_type>(xt::abs(input), axes, options);
                }

                template <class BR, class E, class MR>
                auto merge(const BR& block_result, bool first, E& result, MR&) const
                {
                    if (first)
                    {
                        xt::noalias(result) = block_result;
                    }
                    else
                    {
                        xt::noalias(result) += block_result;
                    }
                }
            };

            template <class T_E>
            struct norm_l2_functor
            {
                using value_type = typename std::decay_t<decltype(xt::norm_l2(std::declval<xarray<T_E>>()))>::value_type;

                template <class E, class A, class O>
                auto compute(const E& input, const A& axes, const O& options) const
                {
                    return xt::sum<value_type>(xt::square(input), axes, options);
                }

                template <class E>
                auto reduction_variable(const E&) const
                {
                    return empty_reduction_variable();
                }

                template <class BR, class E>
                auto merge(const BR& block_result, bool first, E& result, empty_reduction_variable&) const
                {
                    if (first)
                    {
                        xt::noalias(result) = block_result;
                    }
                    else
                    {
                        xt::noalias(result) += block_result;
                    }
                }

                template <class E, class R>
                void finalize(const empty_reduction_variable&, E& results, const R&) const
                {
                    xt::noalias(results) = xt::sqrt(results);
                }
            };

            template <class T_E>
            struct norm_sq_functor : public simple_functor_base
            {
                using value_type = typename std::decay_t<decltype(xt::norm_sq(std::declval<xarray<T_E>>()))>::value_type;

                template <class E, class A, class O>
                auto compute(const E& input, const A& axes, const O& options) const
                {
                    return xt::sum<value_type>(xt::square(input), axes, options);
                }

                template <class BR, class E, class MR>
                auto merge(const BR& block_result, bool first, E& result, MR&) const
                {
                    if (first)
                    {
                        xt::noalias(result) = block_result;
                    }
                    else
                    {
                        xt::noalias(result) += block_result;
                    }
                }
            };

            template <class T_E>
            struct norm_linf_functor : public simple_functor_base
            {
                using value_type = typename std::decay_t<decltype(xt::norm_linf(std::declval<xarray<T_E>>()))>::value_type;

                template <class E, class A, class O>
                auto compute(const E& input, const A& axes, const O& options) const
                {
                    return xt::amax<value_type>(xt::abs(input), axes, options);
                }

                template <class BR, class E, class MR>
                auto merge(const BR& block_result, bool first, E& result, MR&) const
                {
                    if (first)
                    {
                        xt::noalias(result) = block_result;
                    }
                    else
                    {
                        xt::noalias(result) = xt::maximum(block_result, result);
                    }
                }
            };

            template <class T_E>
            class norm_lp_to_p_functor
            {
            public:

                using value_type = typename std::decay_t<
                    decltype(xt::norm_lp_to_p(std::declval<xarray<T_E>>(), 1.0))>::value_type;

                norm_lp_to_p_functor(double p)
                    : m_p(p)
                {
                }

                template <class E, class A, class O>
                auto compute(const E& input, const A& axes, const O& options) const
                {
                    return xt::sum<value_type>(xt::pow(input, m_p), axes, options);
                }

                template <class E>
                auto reduction_variable(const E&) const
                {
                    return empty_reduction_variable();
                }

                template <class BR, class E>
                auto merge(const BR& block_result, bool first, E& result, empty_reduction_variable&) const
                {
                    if (first)
                    {
                        xt::noalias(result) = block_result;
                    }
                    else
                    {
                        xt::noalias(result) += block_result;
                    }
                }

                template <class E, class R>
                void finalize(const empty_reduction_variable&, E&, const R&) const
                {
                }

            private:

                double m_p;
            };

            template <class T_E>
            class norm_lp_functor
            {
            public:

                norm_lp_functor(double p)
                    : m_p(p)
                {
                }

                using value_type = typename std::decay_t<decltype(xt::norm_lp(std::declval<xarray<T_E>>(), 1.0)
                )>::value_type;

                template <class E, class A, class O>
                auto compute(const E& input, const A& axes, const O& options) const
                {
                    return xt::sum<value_type>(xt::pow(input, m_p), axes, options);
                }

                template <class E>
                auto reduction_variable(const E&) const
                {
                    return empty_reduction_variable();
                }

                template <class BR, class E>
                auto merge(const BR& block_result, bool first, E& result, empty_reduction_variable&) const
                {
                    if (first)
                    {
                        xt::noalias(result) = block_result;
                    }
                    else
                    {
                        xt::noalias(result) += block_result;
                    }
                }

                template <class E, class R>
                void finalize(const empty_reduction_variable&, E& results, const R&) const
                {
                    results = xt::pow(results, 1.0 / m_p);
                }

            private:

                double m_p;
            };


        }
    }
}

#endif
