/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_COMPLEX_HPP
#define XTENSOR_COMPLEX_HPP

#include <type_traits>
#include <utility>

#include <xtl/xcomplex.hpp>

#include "../core/xexpression.hpp"
#include "../generators/xbuilder.hpp"
#include "../views/xoffset_view.hpp"

namespace xt
{

    /**
     * @defgroup xt_xcomplex
     *
     * Defined in ``xtensor/xcomplex.hpp``
     */

    /******************************
     * real and imag declarations *
     ******************************/

    template <class E>
    decltype(auto) real(E&& e) noexcept;

    template <class E>
    decltype(auto) imag(E&& e) noexcept;

    /********************************
     * real and imag implementation *
     ********************************/

    namespace detail
    {
        template <bool iscomplex = true>
        struct complex_helper
        {
            template <class E>
            inline static auto real(E&& e) noexcept
            {
                using real_type = typename std::decay_t<E>::value_type::value_type;
                return xoffset_view<xclosure_t<E>, real_type, 0>(std::forward<E>(e));
            }

            template <class E>
            inline static auto imag(E&& e) noexcept
            {
                using real_type = typename std::decay_t<E>::value_type::value_type;
                return xoffset_view<xclosure_t<E>, real_type, sizeof(real_type)>(std::forward<E>(e));
            }
        };

        template <>
        struct complex_helper<false>
        {
            template <class E>
            inline static decltype(auto) real(E&& e) noexcept
            {
                return std::forward<E>(e);
            }

            template <class E>
            inline static auto imag(E&& e) noexcept
            {
                return zeros<typename std::decay_t<E>::value_type>(e.shape());
            }
        };

        template <bool isexpression = true>
        struct complex_expression_helper
        {
            template <class E>
            inline static decltype(auto) real(E&& e) noexcept
            {
                return detail::complex_helper<xtl::is_complex<typename std::decay_t<E>::value_type>::value>::real(
                    std::forward<E>(e)
                );
            }

            template <class E>
            inline static decltype(auto) imag(E&& e) noexcept
            {
                return detail::complex_helper<xtl::is_complex<typename std::decay_t<E>::value_type>::value>::imag(
                    std::forward<E>(e)
                );
            }
        };

        template <>
        struct complex_expression_helper<false>
        {
            template <class E>
            inline static decltype(auto) real(E&& e) noexcept
            {
                return xtl::forward_real(std::forward<E>(e));
            }

            template <class E>
            inline static decltype(auto) imag(E&& e) noexcept
            {
                return xtl::forward_imag(std::forward<E>(e));
            }
        };
    }

    /**
     * Return an xt::xexpression representing the real part of the given expression.
     *
     * The returned expression either hold a const reference to @p e or a copy
     * depending on whether @p e is an lvalue or an rvalue.
     *
     * @ingroup xt_xcomplex
     * @tparam e The xt::xexpression
     */
    template <class E>
    inline decltype(auto) real(E&& e) noexcept
    {
        return detail::complex_expression_helper<is_xexpression<std::decay_t<E>>::value>::real(std::forward<E>(e
        ));
    }

    /**
     * Return an xt::xexpression representing the imaginary part of the given expression.
     *
     * The returned expression either hold a const reference to @p e or a copy
     * depending on whether @p e is an lvalue or an rvalue.
     *
     * @ingroup xt_xcomplex
     * @tparam e The xt::xexpression
     */
    template <class E>
    inline decltype(auto) imag(E&& e) noexcept
    {
        return detail::complex_expression_helper<is_xexpression<std::decay_t<E>>::value>::imag(std::forward<E>(e
        ));
    }

#define UNARY_COMPLEX_FUNCTOR(NS, NAME)             \
    struct NAME##_fun                               \
    {                                               \
        template <class T>                          \
        constexpr auto operator()(const T& t) const \
        {                                           \
            using NS::NAME;                         \
            return NAME(t);                         \
        }                                           \
                                                    \
        template <class B>                          \
        constexpr auto simd_apply(const B& t) const \
        {                                           \
            using NS::NAME;                         \
            return NAME(t);                         \
        }                                           \
    }

    namespace math
    {
        namespace detail
        {
            template <class T>
            constexpr std::complex<T> conj_impl(const std::complex<T>& c)
            {
                return std::complex<T>(c.real(), -c.imag());
            }

            template <class T>
            constexpr std::complex<T> conj_impl(const T& real)
            {
                return std::complex<T>(real, 0);
            }

#ifdef XTENSOR_USE_XSIMD
            template <class T, class A>
            xsimd::complex_batch_type_t<xsimd::batch<T, A>> conj_impl(const xsimd::batch<T, A>& z)
            {
                return xsimd::conj(z);
            }
#endif
        }

        UNARY_COMPLEX_FUNCTOR(std, norm);
        UNARY_COMPLEX_FUNCTOR(std, arg);
        UNARY_COMPLEX_FUNCTOR(detail, conj_impl);
    }

#undef UNARY_COMPLEX_FUNCTOR

    /**
     * Return an xt::xfunction evaluating to the complex conjugate of the given expression.
     *
     * @ingroup xt_xcomplex
     * @param e the xt::xexpression
     */
    template <class E>
    inline auto conj(E&& e) noexcept
    {
        using functor = math::conj_impl_fun;
        using type = xfunction<functor, const_xclosure_t<E>>;
        return type(functor(), std::forward<E>(e));
    }

    /**
     * Calculates the phase angle (in radians) elementwise for the complex numbers in @p e.
     *
     * @ingroup xt_xcomplex
     * @param e the xt::xexpression
     */
    template <class E>
    inline auto arg(E&& e) noexcept
    {
        using functor = math::arg_fun;
        using type = xfunction<functor, const_xclosure_t<E>>;
        return type(functor(), std::forward<E>(e));
    }

    /**
     * Calculates the phase angle elementwise for the complex numbers in @p e.
     *
     * Note that this function might be slightly less performant than xt::arg.
     *
     * @ingroup xt_xcomplex
     * @param e the xt::xexpression
     * @param deg calculate angle in degrees instead of radians
     */
    template <class E>
    inline auto angle(E&& e, bool deg = false) noexcept
    {
        using value_type = xtl::complex_value_type_t<typename std::decay_t<E>::value_type>;
        value_type multiplier = 1.0;
        if (deg)
        {
            multiplier = value_type(180) / numeric_constants<value_type>::PI;
        }
        return arg(std::forward<E>(e)) * std::move(multiplier);
    }

    /**
     * Calculates the squared magnitude elementwise for the complex numbers in @p e.
     *
     * Equivalent to ``xt::pow(xt::real(e), 2) + xt::pow(xt::imag(e), 2)``.
     * @ingroup xt_xcomplex
     * @param e the xt::xexpression
     */
    template <class E>
    inline auto norm(E&& e) noexcept
    {
        using functor = math::norm_fun;
        using type = xfunction<functor, const_xclosure_t<E>>;
        return type(functor(), std::forward<E>(e));
    }
}
#endif
