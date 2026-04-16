/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) Ullrich Koethe
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_NORM_HPP
#define XTENSOR_NORM_HPP

#include <cmath>
// std::abs(int) prior to C++ 17
#include <complex>
#include <cstdlib>

#include <xtl/xtype_traits.hpp>

#include "../core/xmath.hpp"
#include "../core/xoperation.hpp"
#include "../utils/xutils.hpp"

namespace xt
{
    /********************************************
     * type inference for norm and squared norm *
     ********************************************/

    template <class T>
    struct norm_type;

    template <class T>
    struct squared_norm_type;

    namespace traits_detail
    {

        template <class T, bool scalar = xtl::is_arithmetic<T>::value>
        struct norm_of_scalar_impl;

        template <class T>
        struct norm_of_scalar_impl<T, false>
        {
            static const bool value = false;
            using norm_type = void*;
            using squared_norm_type = void*;
        };

        template <class T>
        struct norm_of_scalar_impl<T, true>
        {
            static const bool value = true;
            using norm_type = xtl::promote_type_t<T>;
            using squared_norm_type = xtl::promote_type_t<T>;
        };

        template <class T, bool integral = xtl::is_integral<T>::value, bool floating = std::is_floating_point<T>::value>
        struct norm_of_array_elements_impl;

        template <>
        struct norm_of_array_elements_impl<void*, false, false>
        {
            using norm_type = void*;
            using squared_norm_type = void*;
        };

        template <class T>
        struct norm_of_array_elements_impl<T, false, false>
        {
            using norm_type = typename norm_type<T>::type;
            using squared_norm_type = typename squared_norm_type<T>::type;
        };

        template <class T>
        struct norm_of_array_elements_impl<T, true, false>
        {
            static_assert(
                !std::is_same<T, char>::value,
                "'char' is not a numeric type, use 'signed char' or 'unsigned char'."
            );

            using norm_type = double;
            using squared_norm_type = uint64_t;
        };

        template <class T>
        struct norm_of_array_elements_impl<T, false, true>
        {
            using norm_type = double;
            using squared_norm_type = double;
        };

        template <>
        struct norm_of_array_elements_impl<long double, false, true>
        {
            using norm_type = long double;
            using squared_norm_type = long double;
        };

        template <class ARRAY>
        struct norm_of_vector_impl
        {
            static void* test(...);

            template <class U>
            static typename U::value_type test(U*, typename U::value_type* = 0);

            using T = decltype(test(std::declval<ARRAY*>()));

            static const bool value = !std::is_same<T, void*>::value;

            using norm_type = typename norm_of_array_elements_impl<T>::norm_type;
            using squared_norm_type = typename norm_of_array_elements_impl<T>::squared_norm_type;
        };

        template <class U>
        struct norm_type_base
        {
            using T = std::decay_t<U>;

            static_assert(
                !std::is_same<T, char>::value,
                "'char' is not a numeric type, use 'signed char' or 'unsigned char'."
            );

            using norm_of_scalar = norm_of_scalar_impl<T>;
            using norm_of_vector = norm_of_vector_impl<T>;

            static const bool value = norm_of_scalar::value || norm_of_vector::value;

            static_assert(value, "norm_type<T> are undefined for type U.");
        };
    }  // namespace traits_detail

    /**
     * @brief Traits class for the result type of the <tt>norm_l2()</tt> function.
     *
     * Member 'type' defines the result of <tt>norm_l2(t)</tt>, where <tt>t</tt>
     * is of type @tparam T. It implements the following rules designed to
     * minimize the potential for overflow:
     *   - @tparam T is an arithmetic type: 'type' is the result type of <tt>abs(t)</tt>.
     *   - @tparam T is a container of 'long double' elements: 'type' is <tt>long double</tt>.
     *   - @tparam T is a container of another arithmetic type: 'type' is <tt>double</tt>.
     *   - @tparam T is a container of some other type: 'type' is the element's norm type,
     *
     * Containers are recognized by having an embedded typedef 'value_type'.
     * To change the behavior for a case not covered here, specialize the
     * <tt>traits_detail::norm_type_base</tt> template.
     */
    template <class T>
    struct norm_type : traits_detail::norm_type_base<T>
    {
        using base_type = traits_detail::norm_type_base<T>;

        using type = typename std::conditional<
            base_type::norm_of_vector::value,
            typename base_type::norm_of_vector::norm_type,
            typename base_type::norm_of_scalar::norm_type>::type;
    };

    /**
     * Abbreviation of 'typename norm_type<T>::type'.
     */
    template <class T>
    using norm_type_t = typename norm_type<T>::type;

    /**
     * @brief Traits class for the result type of the <tt>norm_sq()</tt> function.
     *
     * Member 'type' defines the result of <tt>norm_sq(t)</tt>, where <tt>t</tt>
     * is of type @tparam T. It implements the following rules designed to
     * minimize the potential for overflow:
     *   - @tparam T is an arithmetic type: 'type' is the result type of <tt>t*t</tt>.
     *   - @tparam T is a container of 'long double' elements: 'type' is <tt>long double</tt>.
     *   - @tparam T is a container of another floating-point type: 'type' is <tt>double</tt>.
     *   - @tparam T is a container of integer elements: 'type' is <tt>uint64_t</tt>.
     *   - @tparam T is a container of some other type: 'type' is the element's squared norm type,
     *
     *  Containers are recognized by having an embedded typedef 'value_type'.
     *  To change the behavior for a case not covered here, specialize the
     *  <tt>traits_detail::norm_type_base</tt> template.
     */
    template <class T>
    struct squared_norm_type : traits_detail::norm_type_base<T>
    {
        using base_type = traits_detail::norm_type_base<T>;

        using type = typename std::conditional<
            base_type::norm_of_vector::value,
            typename base_type::norm_of_vector::squared_norm_type,
            typename base_type::norm_of_scalar::squared_norm_type>::type;
    };

    /**
     * Abbreviation of 'typename squared_norm_type<T>::type'.
     */
    template <class T>
    using squared_norm_type_t = typename squared_norm_type<T>::type;

    /*************************************
     * norm functions for built-in types *
     *************************************/

///@cond DOXYGEN_INCLUDE_SFINAE
#define XTENSOR_DEFINE_SIGNED_NORMS(T)                                                \
    inline auto norm_lp(T t, double p) noexcept                                       \
    {                                                                                 \
        using rt = decltype(std::abs(t));                                             \
        return p == 0.0 ? static_cast<rt>(t != 0) : std::abs(t);                      \
    }                                                                                 \
    inline auto norm_lp_to_p(T t, double p) noexcept                                  \
    {                                                                                 \
        using rt = xtl::real_promote_type_t<T>;                                       \
        return p == 0.0 ? static_cast<rt>(t != 0)                                     \
                        : std::pow(static_cast<rt>(std::abs(t)), static_cast<rt>(p)); \
    }                                                                                 \
    inline std::size_t norm_l0(T t) noexcept                                          \
    {                                                                                 \
        return (t != 0);                                                              \
    }                                                                                 \
    inline auto norm_l1(T t) noexcept                                                 \
    {                                                                                 \
        return std::abs(t);                                                           \
    }                                                                                 \
    inline auto norm_l2(T t) noexcept                                                 \
    {                                                                                 \
        return std::abs(t);                                                           \
    }                                                                                 \
    inline auto norm_linf(T t) noexcept                                               \
    {                                                                                 \
        return std::abs(t);                                                           \
    }                                                                                 \
    inline auto norm_sq(T t) noexcept                                                 \
    {                                                                                 \
        return t * t;                                                                 \
    }

    XTENSOR_DEFINE_SIGNED_NORMS(signed char)
    XTENSOR_DEFINE_SIGNED_NORMS(short)
    XTENSOR_DEFINE_SIGNED_NORMS(int)
    XTENSOR_DEFINE_SIGNED_NORMS(long)
    XTENSOR_DEFINE_SIGNED_NORMS(long long)
    XTENSOR_DEFINE_SIGNED_NORMS(float)
    XTENSOR_DEFINE_SIGNED_NORMS(double)
    XTENSOR_DEFINE_SIGNED_NORMS(long double)

#undef XTENSOR_DEFINE_SIGNED_NORMS

#define XTENSOR_DEFINE_UNSIGNED_NORMS(T)                                                              \
    inline T norm_lp(T t, double p) noexcept                                                          \
    {                                                                                                 \
        return p == 0.0 ? (t != 0) : t;                                                               \
    }                                                                                                 \
    inline auto norm_lp_to_p(T t, double p) noexcept                                                  \
    {                                                                                                 \
        using rt = xtl::real_promote_type_t<T>;                                                       \
        return p == 0.0 ? static_cast<rt>(t != 0) : std::pow(static_cast<rt>(t), static_cast<rt>(p)); \
    }                                                                                                 \
    inline T norm_l0(T t) noexcept                                                                    \
    {                                                                                                 \
        return t != 0 ? 1 : 0;                                                                        \
    }                                                                                                 \
    inline T norm_l1(T t) noexcept                                                                    \
    {                                                                                                 \
        return t;                                                                                     \
    }                                                                                                 \
    inline T norm_l2(T t) noexcept                                                                    \
    {                                                                                                 \
        return t;                                                                                     \
    }                                                                                                 \
    inline T norm_linf(T t) noexcept                                                                  \
    {                                                                                                 \
        return t;                                                                                     \
    }                                                                                                 \
    inline auto norm_sq(T t) noexcept                                                                 \
    {                                                                                                 \
        return t * t;                                                                                 \
    }

    XTENSOR_DEFINE_UNSIGNED_NORMS(unsigned char)
    XTENSOR_DEFINE_UNSIGNED_NORMS(unsigned short)
    XTENSOR_DEFINE_UNSIGNED_NORMS(unsigned int)
    XTENSOR_DEFINE_UNSIGNED_NORMS(unsigned long)
    XTENSOR_DEFINE_UNSIGNED_NORMS(unsigned long long)

#undef XTENSOR_DEFINE_UNSIGNED_NORMS

    /***********************************
     * norm functions for std::complex *
     ***********************************/

    /**
     * @brief L0 pseudo-norm of a complex number.
     * Equivalent to <tt>t != 0</tt>.
     */
    template <class T>
    inline uint64_t norm_l0(const std::complex<T>& t) noexcept
    {
        return t.real() != 0 || t.imag() != 0;
    }

    /**
     * @brief L1 norm of a complex number.
     */
    template <class T>
    inline auto norm_l1(const std::complex<T>& t) noexcept
    {
        return std::abs(t.real()) + std::abs(t.imag());
    }

    /**
     * @brief L2 norm of a complex number.
     * Equivalent to <tt>std::abs(t)</tt>.
     */
    template <class T>
    inline auto norm_l2(const std::complex<T>& t) noexcept
    {
        return std::abs(t);
    }

    /**
     * @brief Squared norm of a complex number.
     * Equivalent to <tt>std::norm(t)</tt> (yes, the C++ standard really defines
     * <tt>norm()</tt> to compute the squared norm).
     */
    template <class T>
    inline auto norm_sq(const std::complex<T>& t) noexcept
    {
        // Does not use std::norm since it returns a std::complex on OSX
        return t.real() * t.real() + t.imag() * t.imag();
    }

    /**
     * @brief L-infinity norm of a complex number.
     */
    template <class T>
    inline auto norm_linf(const std::complex<T>& t) noexcept
    {
        return (std::max)(std::abs(t.real()), std::abs(t.imag()));
    }

    /**
     * @brief p-th power of the Lp norm of a complex number.
     */
    template <class T>
    inline auto norm_lp_to_p(const std::complex<T>& t, double p) noexcept
    {
        using rt = decltype(std::pow(std::abs(t.real()), static_cast<T>(p)));
        return p == 0 ? static_cast<rt>(t.real() != 0 || t.imag() != 0)
                      : std::pow(std::abs(t.real()), static_cast<T>(p))
                            + std::pow(std::abs(t.imag()), static_cast<T>(p));
    }

    /**
     * @brief Lp norm of a complex number.
     */
    template <class T>
    inline auto norm_lp(const std::complex<T>& t, double p) noexcept
    {
        return p == 0 ? norm_lp_to_p(t, p) : std::pow(norm_lp_to_p(t, p), 1.0 / p);
    }

    /***********************************
     * norm functions for xexpressions *
     ***********************************/

#define XTENSOR_NORM_FUNCTION_AXES(NAME)                                                             \
    template <class E, class I, std::size_t N, class EVS = DEFAULT_STRATEGY_REDUCERS>                \
    inline auto NAME(E&& e, const I(&axes)[N], EVS es = EVS()) noexcept                              \
    {                                                                                                \
        using axes_type = std::array<typename std::decay_t<E>::size_type, N>;                        \
        return NAME(std::forward<E>(e), xtl::forward_sequence<axes_type, decltype(axes)>(axes), es); \
    }

    namespace detail
    {
        template <class T>
        struct norm_value_type
        {
            using type = T;
        };

        template <class T>
        struct norm_value_type<std::complex<T>>
        {
            using type = T;
        };

        template <class T>
        using norm_value_type_t = typename norm_value_type<T>::type;
    }

#define XTENSOR_EMPTY
#define XTENSOR_COMMA ,
#define XTENSOR_NORM_FUNCTION(NAME, RESULT_TYPE, REDUCE_EXPR, REDUCE_OP, MERGE_FUNC)                                       \
    template <class E, class X, class EVS = DEFAULT_STRATEGY_REDUCERS, XTL_REQUIRES(std::negation<is_reducer_options<X>>)> \
    inline auto NAME(E&& e, X&& axes, EVS es = EVS()) noexcept                                                             \
    {                                                                                                                      \
        using value_type = typename std::decay_t<E>::value_type;                                                           \
        using result_type = detail::norm_value_type_t<RESULT_TYPE>;                                                        \
                                                                                                                           \
        auto reduce_func = [](result_type const& r, value_type const& v)                                                   \
        {                                                                                                                  \
            return REDUCE_EXPR(r REDUCE_OP NAME(v));                                                                       \
        };                                                                                                                 \
                                                                                                                           \
        return xt::reduce(                                                                                                 \
            make_xreducer_functor(std::move(reduce_func), const_value<result_type>(0), MERGE_FUNC<result_type>()),         \
            std::forward<E>(e),                                                                                            \
            std::forward<X>(axes),                                                                                         \
            es                                                                                                             \
        );                                                                                                                 \
    }                                                                                                                      \
                                                                                                                           \
    template <class E, class EVS = DEFAULT_STRATEGY_REDUCERS, XTL_REQUIRES(is_xexpression<E>)>                             \
    inline auto NAME(E&& e, EVS es = EVS()) noexcept                                                                       \
    {                                                                                                                      \
        return NAME(std::forward<E>(e), arange(e.dimension()), es);                                                        \
    }                                                                                                                      \
    XTENSOR_NORM_FUNCTION_AXES(NAME)

    XTENSOR_NORM_FUNCTION(norm_l0, unsigned long long, XTENSOR_EMPTY, +, std::plus)
    XTENSOR_NORM_FUNCTION(norm_l1, xtl::big_promote_type_t<value_type>, XTENSOR_EMPTY, +, std::plus)
    XTENSOR_NORM_FUNCTION(norm_sq, xtl::big_promote_type_t<value_type>, XTENSOR_EMPTY, +, std::plus)
    XTENSOR_NORM_FUNCTION(
        norm_linf,
        decltype(norm_linf(std::declval<value_type>())),
        (std::max<result_type>),
        XTENSOR_COMMA,
        math::maximum
    )

#undef XTENSOR_EMPTY
#undef XTENSOR_COMMA
#undef XTENSOR_NORM_FUNCTION
#undef XTENSOR_NORM_FUNCTION_AXES
    /// @endcond
    /**
     * @ingroup red_functions
     * @brief L0 (count) pseudo-norm of an array-like argument over given axes.
     *
     * Returns an \ref xreducer for the L0 pseudo-norm of the elements across given \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the norm is computed (optional)
     * @param es evaluation strategy to use (lazy (default), or immediate)
     * @return an \ref xreducer (or xcontainer, depending on evaluation strategy)
     * When no axes are provided, the norm is calculated over the entire array. In this case,
     * the reducer represents a scalar result, otherwise an array of appropriate dimension.
     */
    template <class E, class X, class EVS, class>
    auto norm_l0(E&& e, X&& axes, EVS es) noexcept;

    /**
     * @ingroup red_functions
     * @brief L1 norm of an array-like argument over given axes.
     *
     * Returns an \ref xreducer for the L1 norm of the elements across given \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the norm is computed (optional)
     * @param es evaluation strategy to use (lazy (default), or immediate)
     * @return an \ref xreducer (or xcontainer, depending on evaluation strategy)
     * When no axes are provided, the norm is calculated over the entire array. In this case,
     * the reducer represents a scalar result, otherwise an array of appropriate dimension.
     */
    template <class E, class X, class EVS, class>
    auto norm_l1(E&& e, X&& axes, EVS es) noexcept;

    /**
     * @ingroup red_functions
     * @brief Squared L2 norm of an array-like argument over given axes.
     *
     * Returns an \ref xreducer for the squared L2 norm of the elements across given \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the norm is computed (optional)
     * @param es evaluation strategy to use (lazy (default), or immediate)
     * @return an \ref xreducer (or xcontainer, depending on evaluation strategy)
     * When no axes are provided, the norm is calculated over the entire array. In this case,
     * the reducer represents a scalar result, otherwise an array of appropriate dimension.
     */
    template <class E, class X, class EVS, class>
    auto norm_sq(E&& e, X&& axes, EVS es) noexcept;

    /**
     * @ingroup red_functions
     * @brief L2 norm of a scalar or array-like argument.
     * @param e an xexpression
     * @param es evaluation strategy to use (lazy (default), or immediate)
     *  For scalar types: implemented as <tt>abs(t)</tt><br>
     *  otherwise: implemented as <tt>sqrt(norm_sq(t))</tt>.
     */
    template <class E, class EVS = DEFAULT_STRATEGY_REDUCERS, XTL_REQUIRES(is_xexpression<E>)>
    inline auto norm_l2(E&& e, EVS es = EVS()) noexcept
    {
        using std::sqrt;
        return sqrt(norm_sq(std::forward<E>(e), es));
    }

    /**
     * @ingroup red_functions
     * @brief L2 norm of an array-like argument over given axes.
     *
     * Returns an \ref xreducer for the L2 norm of the elements across given \em axes.
     * @param e an \ref xexpression
     * @param es evaluation strategy to use (lazy (default), or immediate)
     * @param axes the axes along which the norm is computed
     * @return an \ref xreducer (specifically: <tt>sqrt(norm_sq(e, axes))</tt>) (or xcontainer, depending on
     * evaluation strategy)
     */
    template <
        class E,
        class X,
        class EVS = DEFAULT_STRATEGY_REDUCERS,
        XTL_REQUIRES(is_xexpression<E>, std::negation<is_reducer_options<X>>)>
    inline auto norm_l2(E&& e, X&& axes, EVS es = EVS()) noexcept
    {
        return sqrt(norm_sq(std::forward<E>(e), std::forward<X>(axes), es));
    }

    template <class E, class I, std::size_t N, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto norm_l2(E&& e, const I (&axes)[N], EVS es = EVS()) noexcept
    {
        using axes_type = std::array<typename std::decay_t<E>::size_type, N>;
        return sqrt(norm_sq(std::forward<E>(e), xtl::forward_sequence<axes_type, decltype(axes)>(axes), es));
    }

    /**
     * @ingroup red_functions
     * @brief Infinity (maximum) norm of an array-like argument over given axes.
     *
     * Returns an \ref xreducer for the infinity norm of the elements across given \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the norm is computed (optional)
     * @param es evaluation strategy to use (lazy (default), or immediate)
     * @return an \ref xreducer (or xcontainer, depending on evaluation strategy)
     * When no axes are provided, the norm is calculated over the entire array. In this case,
     * the reducer represents a scalar result, otherwise an array of appropriate dimension.
     */
    template <class E, class X, class EVS, class>
    auto norm_linf(E&& e, X&& axes, EVS es) noexcept;

    /**
     * @ingroup red_functions
     * @brief p-th power of the Lp norm of an array-like argument over given axes.
     *
     * Returns an \ref xreducer for the p-th power of the Lp norm of the elements across given \em axes.
     * @param e an \ref xexpression
     * @param p
     * @param axes the axes along which the norm is computed (optional)
     * @param es evaluation strategy to use (lazy (default), or immediate)
     * @return an \ref xreducer (or xcontainer, depending on evaluation strategy)
     * When no axes are provided, the norm is calculated over the entire array. In this case,
     * the reducer represents a scalar result, otherwise an array of appropriate dimension.
     */
    template <class E, class X, class EVS = DEFAULT_STRATEGY_REDUCERS, XTL_REQUIRES(std::negation<is_reducer_options<X>>)>
    inline auto norm_lp_to_p(E&& e, double p, X&& axes, EVS es = EVS()) noexcept
    {
        using value_type = typename std::decay_t<E>::value_type;
        using result_type = norm_type_t<std::decay_t<E>>;

        auto reduce_func = [p](const result_type& r, const value_type& v)
        {
            return r + norm_lp_to_p(v, p);
        };
        return xt::reduce(
            make_xreducer_functor(std::move(reduce_func), xt::const_value<result_type>(0), std::plus<result_type>()),
            std::forward<E>(e),
            std::forward<X>(axes),
            es
        );
    }

    template <class E, class EVS = DEFAULT_STRATEGY_REDUCERS, XTL_REQUIRES(is_xexpression<E>)>
    inline auto norm_lp_to_p(E&& e, double p, EVS es = EVS()) noexcept
    {
        return norm_lp_to_p(std::forward<E>(e), p, arange(e.dimension()), es);
    }

    template <class E, class I, std::size_t N, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto norm_lp_to_p(E&& e, double p, const I (&axes)[N], EVS es = EVS()) noexcept
    {
        using axes_type = std::array<typename std::decay_t<E>::size_type, N>;
        return norm_lp_to_p(std::forward<E>(e), p, xtl::forward_sequence<axes_type, decltype(axes)>(axes), es);
    }

    /**
     * @ingroup red_functions
     * @brief Lp norm of an array-like argument over given axes.
     *
     * Returns an \ref xreducer for the Lp norm (p != 0) of the elements across given \em axes.
     * @param e an \ref xexpression
     * @param p
     * @param axes the axes along which the norm is computed (optional)
     * @param es evaluation strategy to use (lazy (default), or immediate)
     * @return an \ref xreducer (or xcontainer, depending on evaluation strategy)
     * When no axes are provided, the norm is calculated over the entire array. In this case,
     * the reducer represents a scalar result, otherwise an array of appropriate dimension.
     */
    template <class E, class X, class EVS = DEFAULT_STRATEGY_REDUCERS, XTL_REQUIRES(std::negation<is_reducer_options<X>>)>
    inline auto norm_lp(E&& e, double p, X&& axes, EVS es = EVS())
    {
        XTENSOR_PRECONDITION(p != 0, "norm_lp(): p must be nonzero, use norm_l0() instead.");
        return pow(norm_lp_to_p(std::forward<E>(e), p, std::forward<X>(axes), es), 1.0 / p);
    }

    template <class E, class EVS = DEFAULT_STRATEGY_REDUCERS, XTL_REQUIRES(is_xexpression<E>)>
    inline auto norm_lp(E&& e, double p, EVS es = EVS())
    {
        return norm_lp(std::forward<E>(e), p, arange(e.dimension()), es);
    }

    template <class E, class I, std::size_t N, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto norm_lp(E&& e, double p, const I (&axes)[N], EVS es = EVS())
    {
        using axes_type = std::array<typename std::decay_t<E>::size_type, N>;
        return norm_lp(std::forward<E>(e), p, xtl::forward_sequence<axes_type, decltype(axes)>(axes), es);
    }

    /**
     * @ingroup red_functions
     * @brief Induced L1 norm of a matrix.
     *
     * Returns an \ref xreducer for the induced L1 norm (i.e. the maximum of the L1 norms of e's columns).
     * @param e a 2D \ref xexpression
     * @param es evaluation strategy to use (lazy (default), or immediate)
     * @return an \ref xreducer (or xcontainer, depending on evaluation strategy)
     */
    template <class E, class EVS = DEFAULT_STRATEGY_REDUCERS, XTL_REQUIRES(is_xexpression<E>)>
    inline auto norm_induced_l1(E&& e, EVS es = EVS())
    {
        XTENSOR_PRECONDITION(
            e.dimension() == 2,
            "norm_induced_l1(): only applicable to matrices (e.dimension() must be 2)."
        );
        return norm_linf(norm_l1(std::forward<E>(e), {0}, es), es);
    }

    /**
     * @ingroup red_functions
     * @brief Induced L-infinity norm of a matrix.
     *
     * Returns an \ref xreducer for the induced L-infinity norm (i.e. the maximum of the L1 norms of e's
     * rows).
     * @param e a 2D \ref xexpression
     * @param es evaluation strategy to use (lazy (default), or immediate)
     * @return an \ref xreducer (or xcontainer, depending on evaluation strategy)
     */
    template <class E, class EVS = DEFAULT_STRATEGY_REDUCERS, XTL_REQUIRES(is_xexpression<E>)>
    inline auto norm_induced_linf(E&& e, EVS es = EVS())
    {
        XTENSOR_PRECONDITION(
            e.dimension() == 2,
            "norm_induced_linf(): only applicable to matrices (e.dimension() must be 2)."
        );
        return norm_linf(norm_l1(std::forward<E>(e), {1}, es), es);
    }

}  // namespace xt

#endif
