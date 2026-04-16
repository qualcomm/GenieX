/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_OPERATION_HPP
#define XTENSOR_OPERATION_HPP

#include <algorithm>
#include <functional>
#include <type_traits>

#include <xtl/xsequence.hpp>

#include "../containers/xscalar.hpp"
#include "../core/xfunction.hpp"
#include "../core/xstrides.hpp"
#include "../views/xstrided_view.hpp"

namespace xt
{

    /***********
     * helpers *
     ***********/

#define UNARY_OPERATOR_FUNCTOR(NAME, OP)               \
    struct NAME                                        \
    {                                                  \
        template <class A1>                            \
        constexpr auto operator()(const A1& arg) const \
        {                                              \
            return OP arg;                             \
        }                                              \
        template <class B>                             \
        constexpr auto simd_apply(const B& arg) const  \
        {                                              \
            return OP arg;                             \
        }                                              \
    }

#define DEFINE_COMPLEX_OVERLOAD(OP)                                                           \
    template <class T1, class T2, XTL_REQUIRES(std::negation<std::is_same<T1, T2>>)>          \
    constexpr auto operator OP(const std::complex<T1>& arg1, const std::complex<T2>& arg2)    \
    {                                                                                         \
        using result_type = typename xtl::promote_type_t<std::complex<T1>, std::complex<T2>>; \
        return (result_type(arg1) OP result_type(arg2));                                      \
    }                                                                                         \
                                                                                              \
    template <class T1, class T2, XTL_REQUIRES(std::negation<std::is_same<T1, T2>>)>          \
    constexpr auto operator OP(const T1& arg1, const std::complex<T2>& arg2)                  \
    {                                                                                         \
        using result_type = typename xtl::promote_type_t<T1, std::complex<T2>>;               \
        return (result_type(arg1) OP result_type(arg2));                                      \
    }                                                                                         \
                                                                                              \
    template <class T1, class T2, XTL_REQUIRES(std::negation<std::is_same<T1, T2>>)>          \
    constexpr auto operator OP(const std::complex<T1>& arg1, const T2& arg2)                  \
    {                                                                                         \
        using result_type = typename xtl::promote_type_t<std::complex<T1>, T2>;               \
        return (result_type(arg1) OP result_type(arg2));                                      \
    }

#define BINARY_OPERATOR_FUNCTOR(NAME, OP)                              \
    struct NAME                                                        \
    {                                                                  \
        template <class T1, class T2>                                  \
        constexpr auto operator()(T1&& arg1, T2&& arg2) const          \
        {                                                              \
            using xt::detail::operator OP;                             \
            return (std::forward<T1>(arg1) OP std::forward<T2>(arg2)); \
        }                                                              \
        template <class B>                                             \
        constexpr auto simd_apply(const B& arg1, const B& arg2) const  \
        {                                                              \
            return (arg1 OP arg2);                                     \
        }                                                              \
    }

    namespace detail
    {
        DEFINE_COMPLEX_OVERLOAD(+);
        DEFINE_COMPLEX_OVERLOAD(-);
        DEFINE_COMPLEX_OVERLOAD(*);
        DEFINE_COMPLEX_OVERLOAD(/);
        DEFINE_COMPLEX_OVERLOAD(%);
        DEFINE_COMPLEX_OVERLOAD(||);
        DEFINE_COMPLEX_OVERLOAD(&&);
        DEFINE_COMPLEX_OVERLOAD(|);
        DEFINE_COMPLEX_OVERLOAD(&);
        DEFINE_COMPLEX_OVERLOAD(^);
        DEFINE_COMPLEX_OVERLOAD(<<);
        DEFINE_COMPLEX_OVERLOAD(>>);
        DEFINE_COMPLEX_OVERLOAD(<);
        DEFINE_COMPLEX_OVERLOAD(<=);
        DEFINE_COMPLEX_OVERLOAD(>);
        DEFINE_COMPLEX_OVERLOAD(>=);
        DEFINE_COMPLEX_OVERLOAD(==);
        DEFINE_COMPLEX_OVERLOAD(!=);

        UNARY_OPERATOR_FUNCTOR(identity, +);
        UNARY_OPERATOR_FUNCTOR(negate, -);
        BINARY_OPERATOR_FUNCTOR(plus, +);
        BINARY_OPERATOR_FUNCTOR(minus, -);
        BINARY_OPERATOR_FUNCTOR(multiplies, *);
        BINARY_OPERATOR_FUNCTOR(divides, /);
        BINARY_OPERATOR_FUNCTOR(modulus, %);
        BINARY_OPERATOR_FUNCTOR(logical_or, ||);
        BINARY_OPERATOR_FUNCTOR(logical_and, &&);
        UNARY_OPERATOR_FUNCTOR(logical_not, !);
        BINARY_OPERATOR_FUNCTOR(bitwise_or, |);
        BINARY_OPERATOR_FUNCTOR(bitwise_and, &);
        BINARY_OPERATOR_FUNCTOR(bitwise_xor, ^);
        UNARY_OPERATOR_FUNCTOR(bitwise_not, ~);
        BINARY_OPERATOR_FUNCTOR(left_shift, <<);
        BINARY_OPERATOR_FUNCTOR(right_shift, >>);
        BINARY_OPERATOR_FUNCTOR(less, <);
        BINARY_OPERATOR_FUNCTOR(less_equal, <=);
        BINARY_OPERATOR_FUNCTOR(greater, >);
        BINARY_OPERATOR_FUNCTOR(greater_equal, >=);
        BINARY_OPERATOR_FUNCTOR(equal_to, ==);
        BINARY_OPERATOR_FUNCTOR(not_equal_to, !=);

        struct conditional_ternary
        {
            template <class B>
            using get_batch_bool = typename xt_simd::simd_traits<typename xt_simd::revert_simd_traits<B>::type>::bool_type;

            template <class B, class A1, class A2>
            constexpr auto operator()(const B& cond, const A1& v1, const A2& v2) const noexcept
            {
                return xtl::select(cond, v1, v2);
            }

            template <class B>
            constexpr B simd_apply(const get_batch_bool<B>& t1, const B& t2, const B& t3) const noexcept
            {
                return xt_simd::select(t1, t2, t3);
            }
        };

        template <class R>
        struct cast
        {
            struct functor
            {
                using result_type = R;

                template <class A1>
                constexpr result_type operator()(const A1& arg) const
                {
                    return static_cast<R>(arg);
                }

                // SIMD conversion disabled for now since it does not make sense
                // in most of the cases
                /*constexpr simd_result_type simd_apply(const simd_value_type& arg) const
                {
                    return static_cast<R>(arg);
                }*/
            };
        };

        template <class Tag, class F, class... E>
        struct select_xfunction_expression;

        template <class F, class... E>
        struct select_xfunction_expression<xtensor_expression_tag, F, E...>
        {
            using type = xfunction<F, E...>;
        };

        template <class F, class... E>
        struct select_xfunction_expression<xoptional_expression_tag, F, E...>
        {
            using type = xfunction<F, E...>;
        };

        template <class Tag, class F, class... E>
        using select_xfunction_expression_t = typename select_xfunction_expression<Tag, F, E...>::type;

        template <class F, class... E>
        struct xfunction_type
        {
            using expression_tag = xexpression_tag_t<E...>;
            using functor_type = F;
            using type = select_xfunction_expression_t<expression_tag, functor_type, const_xclosure_t<E>...>;
        };

        template <class F, class... E>
        inline auto make_xfunction(E&&... e) noexcept
        {
            using function_type = xfunction_type<F, E...>;
            using functor_type = typename function_type::functor_type;
            using type = typename function_type::type;
            return type(functor_type(), std::forward<E>(e)...);
        }

        // On MSVC, the second argument of enable_if_t is always evaluated, even if the condition is false.
        // Wrapping the xfunction type in the xfunction_type metafunction avoids this evaluation when
        // the condition is false, since it leads to a tricky bug preventing from using operator+ and
        // operator- on vector and arrays iterators.
        template <class F, class... E>
        using xfunction_type_t = typename std::
            enable_if_t<has_xexpression<std::decay_t<E>...>::value, xfunction_type<F, E...>>::type;
    }

#undef UNARY_OPERATOR_FUNCTOR
#undef BINARY_OPERATOR_FUNCTOR

    /*************
     * operators *
     *************/

    /**
     * @defgroup arithmetic_operators Arithmetic operators
     */

    /**
     * @ingroup arithmetic_operators
     * @brief Identity
     *
     * Returns an \ref xfunction for the element-wise identity
     * of \a e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto operator+(E&& e) noexcept -> detail::xfunction_type_t<detail::identity, E>
    {
        return detail::make_xfunction<detail::identity>(std::forward<E>(e));
    }

    /**
     * @ingroup arithmetic_operators
     * @brief Opposite
     *
     * Returns an \ref xfunction for the element-wise opposite
     * of \a e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto operator-(E&& e) noexcept -> detail::xfunction_type_t<detail::negate, E>
    {
        return detail::make_xfunction<detail::negate>(std::forward<E>(e));
    }

    /**
     * @ingroup arithmetic_operators
     * @brief Addition
     *
     * Returns an \ref xfunction for the element-wise addition
     * of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto operator+(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<detail::plus, E1, E2>
    {
        return detail::make_xfunction<detail::plus>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup arithmetic_operators
     * @brief Substraction
     *
     * Returns an \ref xfunction for the element-wise substraction
     * of \a e2 to \a e1.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto operator-(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<detail::minus, E1, E2>
    {
        return detail::make_xfunction<detail::minus>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup arithmetic_operators
     * @brief Multiplication
     *
     * Returns an \ref xfunction for the element-wise multiplication
     * of \a e1 by \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto operator*(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<detail::multiplies, E1, E2>
    {
        return detail::make_xfunction<detail::multiplies>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup arithmetic_operators
     * @brief Division
     *
     * Returns an \ref xfunction for the element-wise division
     * of \a e1 by \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto operator/(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<detail::divides, E1, E2>
    {
        return detail::make_xfunction<detail::divides>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup arithmetic_operators
     * @brief Modulus
     *
     * Returns an \ref xfunction for the element-wise modulus
     * of \a e1 by \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto operator%(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<detail::modulus, E1, E2>
    {
        return detail::make_xfunction<detail::modulus>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @defgroup logical_operators Logical operators
     */

    /**
     * @ingroup logical_operators
     * @brief Or
     *
     * Returns an \ref xfunction for the element-wise or
     * of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto operator||(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<detail::logical_or, E1, E2>
    {
        return detail::make_xfunction<detail::logical_or>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup logical_operators
     * @brief And
     *
     * Returns an \ref xfunction for the element-wise and
     * of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto operator&&(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<detail::logical_and, E1, E2>
    {
        return detail::make_xfunction<detail::logical_and>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup logical_operators
     * @brief Not
     *
     * Returns an \ref xfunction for the element-wise not
     * of \a e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto operator!(E&& e) noexcept -> detail::xfunction_type_t<detail::logical_not, E>
    {
        return detail::make_xfunction<detail::logical_not>(std::forward<E>(e));
    }

    /**
     * @defgroup bitwise_operators Bitwise operators
     */

    /**
     * @ingroup bitwise_operators
     * @brief Bitwise and
     *
     * Returns an \ref xfunction for the element-wise bitwise and
     * of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto operator&(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<detail::bitwise_and, E1, E2>
    {
        return detail::make_xfunction<detail::bitwise_and>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup bitwise_operators
     * @brief Bitwise or
     *
     * Returns an \ref xfunction for the element-wise bitwise or
     * of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto operator|(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<detail::bitwise_or, E1, E2>
    {
        return detail::make_xfunction<detail::bitwise_or>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup bitwise_operators
     * @brief Bitwise xor
     *
     * Returns an \ref xfunction for the element-wise bitwise xor
     * of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto operator^(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<detail::bitwise_xor, E1, E2>
    {
        return detail::make_xfunction<detail::bitwise_xor>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup bitwise_operators
     * @brief Bitwise not
     *
     * Returns an \ref xfunction for the element-wise bitwise not
     * of \a e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto operator~(E&& e) noexcept -> detail::xfunction_type_t<detail::bitwise_not, E>
    {
        return detail::make_xfunction<detail::bitwise_not>(std::forward<E>(e));
    }

    /**
     * @ingroup bitwise_operators
     * @brief Bitwise left shift
     *
     * Returns an \ref xfunction for the element-wise bitwise left shift of e1
     * by e2.
     * @param e1 an \ref xexpression
     * @param e2 an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto left_shift(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<detail::left_shift, E1, E2>
    {
        return detail::make_xfunction<detail::left_shift>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup bitwise_operators
     * @brief Bitwise left shift
     *
     * Returns an \ref xfunction for the element-wise bitwise left shift of e1
     * by e2.
     * @param e1 an \ref xexpression
     * @param e2 an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto right_shift(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<detail::right_shift, E1, E2>
    {
        return detail::make_xfunction<detail::right_shift>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    namespace detail
    {
        // Shift operator is not available for all the types, so the xfunction type instantiation
        // has to be delayed, enable_if_t is not sufficient
        template <class F, class E1, class E2>
        struct shift_function_getter
        {
            using type = xfunction_type_t<F, E1, E2>;
        };

        template <bool B, class T>
        struct eval_enable_if
        {
            using type = typename T::type;
        };

        template <class T>
        struct eval_enable_if<false, T>
        {
        };

        template <bool B, class T>
        using eval_enable_if_t = typename eval_enable_if<B, T>::type;

        template <class F, class E1, class E2>
        using shift_return_type_t = eval_enable_if_t<
            is_xexpression<std::decay_t<E1>>::value,
            shift_function_getter<F, E1, E2>>;
    }

    /**
     * @ingroup bitwise_operators
     * @brief Bitwise left shift
     *
     * Returns an \ref xfunction for the element-wise bitwise left shift of e1
     * by e2.
     * @param e1 an \ref xexpression
     * @param e2 an \ref xexpression
     * @return an \ref xfunction
     * @sa left_shift
     */
    template <class E1, class E2>
    inline auto operator<<(E1&& e1, E2&& e2) noexcept
        -> detail::shift_return_type_t<detail::left_shift, E1, E2>
    {
        return left_shift(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup bitwise_operators
     * @brief Bitwise right shift
     *
     * Returns an \ref xfunction for the element-wise bitwise right shift of e1
     * by e2.
     * @param e1 an \ref xexpression
     * @param e2 an \ref xexpression
     * @return an \ref xfunction
     * @sa right_shift
     */
    template <class E1, class E2>
    inline auto operator>>(E1&& e1, E2&& e2) -> detail::shift_return_type_t<detail::right_shift, E1, E2>
    {
        return right_shift(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @defgroup comparison_operators Comparison operators
     */

    /**
     * @ingroup comparison_operators
     * @brief Lesser than
     *
     * Returns an \ref xfunction for the element-wise
     * lesser than comparison of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto operator<(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<detail::less, E1, E2>
    {
        return detail::make_xfunction<detail::less>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup comparison_operators
     * @brief Lesser or equal
     *
     * Returns an \ref xfunction for the element-wise
     * lesser or equal comparison of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto operator<=(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<detail::less_equal, E1, E2>
    {
        return detail::make_xfunction<detail::less_equal>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup comparison_operators
     * @brief Greater than
     *
     * Returns an \ref xfunction for the element-wise
     * greater than comparison of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto operator>(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<detail::greater, E1, E2>
    {
        return detail::make_xfunction<detail::greater>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup comparison_operators
     * @brief Greater or equal
     *
     * Returns an \ref xfunction for the element-wise
     * greater or equal comparison of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto operator>=(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<detail::greater_equal, E1, E2>
    {
        return detail::make_xfunction<detail::greater_equal>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup comparison_operators
     * @brief Equality
     *
     * Returns true if \a e1 and \a e2 have the same shape
     * and hold the same values. Unlike other comparison
     * operators, this does not return an \ref xfunction.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return a boolean
     */
    template <class E1, class E2>
    inline std::enable_if_t<xoptional_comparable<E1, E2>::value, bool>
    operator==(const xexpression<E1>& e1, const xexpression<E2>& e2)
    {
        const E1& de1 = e1.derived_cast();
        const E2& de2 = e2.derived_cast();
        bool res = de1.dimension() == de2.dimension()
                   && std::equal(de1.shape().begin(), de1.shape().end(), de2.shape().begin());
        auto iter1 = de1.begin();
        auto iter2 = de2.begin();
        auto iter_end = de1.end();
        while (res && iter1 != iter_end)
        {
            res = (*iter1++ == *iter2++);
        }
        return res;
    }

    /**
     * @ingroup comparison_operators
     * @brief Inequality
     *
     * Returns true if \a e1 and \a e2 have different shapes
     * or hold the different values. Unlike other comparison
     * operators, this does not return an \ref xfunction.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return a boolean
     */
    template <class E1, class E2>
    inline bool operator!=(const xexpression<E1>& e1, const xexpression<E2>& e2)
    {
        return !(e1 == e2);
    }

    /**
     * @ingroup comparison_operators
     * @brief Element-wise equality
     *
     * Returns an \ref xfunction for the element-wise
     * equality of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto equal(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<detail::equal_to, E1, E2>
    {
        return detail::make_xfunction<detail::equal_to>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup comparison_operators
     * @brief Element-wise inequality
     *
     * Returns an \ref xfunction for the element-wise
     * inequality of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto not_equal(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<detail::not_equal_to, E1, E2>
    {
        return detail::make_xfunction<detail::not_equal_to>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup comparison_operators
     * @brief Lesser than
     *
     * Returns an \ref xfunction for the element-wise
     * lesser than comparison of \a e1 and \a e2. This
     * function is equivalent to operator<(E1&&, E2&&).
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto less(E1&& e1, E2&& e2) noexcept -> decltype(std::forward<E1>(e1) < std::forward<E2>(e2))
    {
        return std::forward<E1>(e1) < std::forward<E2>(e2);
    }

    /**
     * @ingroup comparison_operators
     * @brief Lesser or equal
     *
     * Returns an \ref xfunction for the element-wise
     * lesser or equal comparison of \a e1 and \a e2. This
     * function is equivalent to operator<=(E1&&, E2&&).
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto less_equal(E1&& e1, E2&& e2) noexcept -> decltype(std::forward<E1>(e1) <= std::forward<E2>(e2))
    {
        return std::forward<E1>(e1) <= std::forward<E2>(e2);
    }

    /**
     * @ingroup comparison_operators
     * @brief Greater than
     *
     * Returns an \ref xfunction for the element-wise
     * greater than comparison of \a e1 and \a e2. This
     * function is equivalent to operator>(E1&&, E2&&).
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto greater(E1&& e1, E2&& e2) noexcept -> decltype(std::forward<E1>(e1) > std::forward<E2>(e2))
    {
        return std::forward<E1>(e1) > std::forward<E2>(e2);
    }

    /**
     * @ingroup comparison_operators
     * @brief Greater or equal
     *
     * Returns an \ref xfunction for the element-wise
     * greater or equal comparison of \a e1 and \a e2.
     * This function is equivalent to operator>=(E1&&, E2&&).
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto greater_equal(E1&& e1, E2&& e2) noexcept
        -> decltype(std::forward<E1>(e1) >= std::forward<E2>(e2))
    {
        return std::forward<E1>(e1) >= std::forward<E2>(e2);
    }

    /**
     * @ingroup logical_operators
     * @brief Ternary selection
     *
     * Returns an \ref xfunction for the element-wise
     * ternary selection (i.e. operator ? :) of \a e1,
     * \a e2 and \a e3.
     * @param e1 a boolean \ref xexpression
     * @param e2 an \ref xexpression or a scalar
     * @param e3 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2, class E3>
    inline auto where(E1&& e1, E2&& e2, E3&& e3) noexcept
        -> detail::xfunction_type_t<detail::conditional_ternary, E1, E2, E3>
    {
        return detail::make_xfunction<detail::conditional_ternary>(
            std::forward<E1>(e1),
            std::forward<E2>(e2),
            std::forward<E3>(e3)
        );
    }

    namespace detail
    {
        template <layout_type L>
        struct next_idx_impl;

        template <>
        struct next_idx_impl<layout_type::row_major>
        {
            template <class S, class I>
            inline auto operator()(const S& shape, I& idx)
            {
                for (std::size_t j = shape.size(); j > 0; --j)
                {
                    std::size_t i = j - 1;
                    if (idx[i] >= shape[i] - 1)
                    {
                        idx[i] = 0;
                    }
                    else
                    {
                        idx[i]++;
                        return idx;
                    }
                }
                // return empty index, happens at last iteration step, but remains unused
                return I();
            }
        };

        template <>
        struct next_idx_impl<layout_type::column_major>
        {
            template <class S, class I>
            inline auto operator()(const S& shape, I& idx)
            {
                for (std::size_t i = 0; i < shape.size(); ++i)
                {
                    if (idx[i] >= shape[i] - 1)
                    {
                        idx[i] = 0;
                    }
                    else
                    {
                        idx[i]++;
                        return idx;
                    }
                }
                // return empty index, happens at last iteration step, but remains unused
                return I();
            }
        };

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S, class I>
        inline auto next_idx(const S& shape, I& idx)
        {
            next_idx_impl<L> nii;
            return nii(shape, idx);
        }
    }

    /**
     * @ingroup logical_operators
     * @brief return vector of indices where T is not zero
     *
     * @param arr input array
     * @return vector of vectors, one for each dimension of arr, containing
     * the indices of the non-zero elements in that dimension
     */
    template <class T>
    inline auto nonzero(const T& arr)
    {
        auto shape = arr.shape();
        using index_type = xindex_type_t<typename T::shape_type>;
        using size_type = typename T::size_type;

        auto idx = xtl::make_sequence<index_type>(arr.dimension(), 0);
        std::vector<std::vector<size_type>> indices(arr.dimension());

        size_type total_size = compute_size(shape);
        for (size_type i = 0; i < total_size; i++, detail::next_idx(shape, idx))
        {
            if (arr.element(std::begin(idx), std::end(idx)))
            {
                for (std::size_t n = 0; n < indices.size(); ++n)
                {
                    indices.at(n).push_back(idx[n]);
                }
            }
        }

        return indices;
    }

    /**
     * @ingroup logical_operators
     * @brief return vector of indices where condition is true
     *        (equivalent to \a nonzero(condition))
     *
     * @param condition input array
     * @return vector of \a index_types where condition is not equal to zero
     */
    template <class T>
    inline auto where(const T& condition)
    {
        return nonzero(condition);
    }

    /**
     * @ingroup logical_operators
     * @brief return vector of indices where arr is not zero
     *
     * @tparam L the traversal order
     * @param arr input array
     * @return vector of index_types where arr is not equal to zero (use `xt::from_indices` to convert)
     *
     * @sa xt::from_indices
     */
    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class T>
    inline auto argwhere(const T& arr)
    {
        auto shape = arr.shape();
        using index_type = xindex_type_t<typename T::shape_type>;
        using size_type = typename T::size_type;

        auto idx = xtl::make_sequence<index_type>(arr.dimension(), 0);
        std::vector<index_type> indices;

        size_type total_size = compute_size(shape);
        for (size_type i = 0; i < total_size; i++, detail::next_idx<L>(shape, idx))
        {
            if (arr.element(std::begin(idx), std::end(idx)))
            {
                indices.push_back(idx);
            }
        }

        return indices;
    }

    /**
     * @ingroup logical_operators
     * @brief Any
     *
     * Returns true if any of the values of \a e is truthy,
     * false otherwise.
     * @param e an \ref xexpression
     * @return a boolean
     */
    template <class E>
    inline bool any(E&& e)
    {
        using xtype = std::decay_t<E>;
        using value_type = typename xtype::value_type;
        return std::any_of(
            e.cbegin(),
            e.cend(),
            [](const value_type& el)
            {
                return el;
            }
        );
    }

    /**
     * @ingroup logical_operators
     * @brief Any
     *
     * Returns true if all of the values of \a e are truthy,
     * false otherwise.
     * @param e an \ref xexpression
     * @return a boolean
     */
    template <class E>
    inline bool all(E&& e)
    {
        using xtype = std::decay_t<E>;
        using value_type = typename xtype::value_type;
        return std::all_of(
            e.cbegin(),
            e.cend(),
            [](const value_type& el)
            {
                return el;
            }
        );
    }

    /**
     * @defgroup casting_operators Casting operators
     */

    /**
     * @ingroup casting_operators
     * @brief Element-wise ``static_cast``.
     *
     * Returns an \ref xfunction for the element-wise
     * static_cast of \a e to type R.
     *
     * @param e an \ref xexpression or a scalar
     * @return an \ref xfunction
     */

    template <class R, class E>
    inline auto cast(E&& e) noexcept -> detail::xfunction_type_t<typename detail::cast<R>::functor, E>
    {
        return detail::make_xfunction<typename detail::cast<R>::functor>(std::forward<E>(e));
    }

}

#endif
