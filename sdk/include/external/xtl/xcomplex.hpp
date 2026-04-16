/***************************************************************************
* Copyright (c) Sylvain Corlay and Johan Mabille and Wolf Vollprecht       *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_COMPLEX_HPP
#define XTL_COMPLEX_HPP

#if !defined(_MSC_VER)
#include <cmath>
using std::copysign;
#endif

#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <sstream>
#include <string>

#ifdef __CLING__
#include <nlohmann/json.hpp>
#endif

#include "xclosure.hpp"
#include "xtl_config.hpp"
#include "xtype_traits.hpp"

namespace xtl
{
    template <class CTR, class CTI = CTR, bool ieee_compliant = false>
    class xcomplex;

    /**************
     * is_complex *
     **************/

    namespace detail
    {
        template <class T>
        struct is_complex : std::false_type
        {
        };

        template <class T>
        struct is_complex<std::complex<T>> : std::true_type
        {
        };
    }

    template <class T>
    struct is_complex
    {
        static constexpr bool value = detail::is_complex<std::decay_t<T>>::value;
    };

    /***************
     * is_xcomplex *
     ***************/

    namespace detail
    {
        template <class T>
        struct is_xcomplex : std::false_type
        {
        };

        template <class CTR, class CTI, bool B>
        struct is_xcomplex<xcomplex<CTR, CTI, B>> : std::true_type
        {
        };
    }

    template <class T>
    struct is_xcomplex
    {
        static constexpr bool value = detail::is_xcomplex<std::decay_t<T>>::value;
    };

    /******************
     * is_gen_complex *
     ******************/

    template <class T>
    using is_gen_complex = std::disjunction<is_complex<std::decay_t<T>>, is_xcomplex<std::decay_t<T>>>;

    /****************************
     * enable / disable complex *
     ****************************/

    template <class E, class R = void>
    using disable_xcomplex = std::enable_if_t<!is_gen_complex<E>::value, R>;

    template <class E, class R = void>
    using enable_xcomplex = std::enable_if_t<is_gen_complex<E>::value, R>;

    /*****************
     * enable_scalar *
     *****************/

    template <class E, class R = void>
    using enable_scalar = std::enable_if_t<xtl::is_arithmetic<E>::value, R>;

    /*******************
     * common_xcomplex *
     *******************/

    template <class CTR1, class CTI1, bool ieee1, class CTR2, class CTI2, bool ieee2>
    struct common_xcomplex
    {
        using type = xcomplex<std::common_type_t<CTR1, CTI1>, std::common_type_t<CTR2, CTI2>, ieee1 || ieee2>;
    };

    template <class CTR1, class CTI1, bool ieee1, class CTR2, class CTI2, bool ieee2>
    using common_xcomplex_t = typename common_xcomplex<CTR1, CTI1, ieee1, CTR2, CTI2, ieee2>::type;

    /**********************
     * temporary_xcomplex *
     **********************/

    template <class CTR, class CTI, bool ieee>
    struct temporary_xcomplex
    {
        using type = xcomplex<std::decay_t<CTR>, std::decay_t<CTI>, ieee>;
    };

    template <class CTR, class CTI, bool ieee>
    using temporary_xcomplex_t = typename temporary_xcomplex<CTR, CTI, ieee>::type;

    /************
     * xcomplex *
     ************/

    template <class CTR, class CTI, bool ieee_compliant>
    class xcomplex
    {
    public:

        static_assert(std::is_same<std::decay_t<CTR>, std::decay_t<CTI>>::value,
                      "closure types must have the same value type");

        using value_type = std::common_type_t<CTR, CTI>;
        using self_type = xcomplex<CTR, CTI, ieee_compliant>;
        using temporary_type = temporary_xcomplex_t<CTR, CTI, ieee_compliant>;

        using real_reference = std::add_lvalue_reference_t<CTR>;
        using real_const_reference = std::add_lvalue_reference_t<std::add_const_t<CTR>>;
        using real_rvalue_reference = std::conditional_t<std::is_reference<CTR>::value, apply_cv_t<CTR, value_type>&, value_type>;
        using real_rvalue_const_reference = std::conditional_t<std::is_reference<CTR>::value, const value_type&, value_type>;

        using imag_reference = std::add_lvalue_reference_t<CTI>;
        using imag_const_reference = std::add_lvalue_reference_t<std::add_const_t<CTI>>;
        using imag_rvalue_reference = std::conditional_t<std::is_reference<CTI>::value, apply_cv_t<CTI, value_type>&, value_type>;
        using imag_rvalue_const_reference = std::conditional_t<std::is_reference<CTI>::value, const value_type&, value_type>;

        constexpr xcomplex() noexcept
            : m_real(), m_imag()
        {
        }

        template <class OCTR,
            std::enable_if_t<
                std::conjunction<
                    std::negation<is_gen_complex<OCTR>>,
                    std::is_constructible<CTR, OCTR&&>,
                    std::is_convertible<OCTR&&, CTR>
                >::value,
                bool
            > = true>
        constexpr xcomplex(OCTR&& re) noexcept
            : m_real(std::forward<OCTR>(re)), m_imag()
        {
        }

        template <class OCTR,
            std::enable_if_t<
                std::conjunction<
                    std::negation<is_gen_complex<OCTR>>,
                    std::is_constructible<CTR, OCTR&&>,
                    std::negation<std::is_convertible<OCTR&&, CTR>>
            >::value,
            bool
        > = true>
        explicit constexpr xcomplex(OCTR&& re) noexcept
            : m_real(std::forward<OCTR>(re)), m_imag()
        {
        }

        template <class OCTR, class OCTI>
        explicit constexpr xcomplex(OCTR&& re, OCTI&& im) noexcept
            : m_real(std::forward<OCTR>(re)), m_imag(std::forward<OCTI>(im))
        {
        }

        template <class OCTR, class OCTI, bool OB>
        explicit constexpr xcomplex(const xcomplex<OCTR, OCTI, OB>& rhs) noexcept
            : m_real(rhs.real()), m_imag(rhs.imag())
        {
        }

        template <class OCTR, class OCTI, bool OB>
        explicit constexpr xcomplex(xcomplex<OCTR, OCTI, OB>&& rhs) noexcept
            : m_real(std::move(rhs).real()), m_imag(std::move(rhs).imag())
        {
        }

        template <class T>
        constexpr xcomplex(const std::complex<T>& rhs) noexcept
            : m_real(rhs.real()), m_imag(rhs.imag())
        {
        }

        template <class T>
        constexpr xcomplex(std::complex<T>&& rhs) noexcept
            : m_real(std::move(rhs).real()), m_imag(std::move(rhs).imag())
        {
        }

        template <class OCTR>
        disable_xcomplex<OCTR, self_type&> operator=(OCTR&& rhs) noexcept;
        template <class OCTR, class OCTI, bool OB>
        self_type& operator=(const xcomplex<OCTR, OCTI, OB>& rhs) noexcept;
        template <class OCTR, class OCTI, bool OB>
        self_type& operator=(xcomplex<OCTR, OCTI, OB>&& rhs) noexcept;

        operator std::complex<std::decay_t<CTR>>() const noexcept;

        template <class OCTR, class OCTI, bool OB>
        self_type& operator+=(const xcomplex<OCTR, OCTI, OB>& rhs) noexcept;
        template <class OCTR, class OCTI, bool OB>
        self_type& operator-=(const xcomplex<OCTR, OCTI, OB>& rhs) noexcept;
        template <class OCTR, class OCTI, bool OB>
        self_type& operator*=(const xcomplex<OCTR, OCTI, OB>& rhs) noexcept;
        template <class OCTR, class OCTI, bool OB>
        self_type& operator/=(const xcomplex<OCTR, OCTI, OB>& rhs) noexcept;

        template <class T>
        disable_xcomplex<T, self_type&> operator+=(const T& rhs) noexcept;
        template <class T>
        disable_xcomplex<T, self_type&> operator-=(const T& rhs) noexcept;
        template <class T>
        disable_xcomplex<T, self_type&> operator*=(const T& rhs) noexcept;
        template <class T>
        disable_xcomplex<T, self_type&> operator/=(const T& rhs) noexcept;

        real_reference real() & noexcept;
        real_rvalue_reference real() && noexcept;
        constexpr real_const_reference real() const & noexcept;
        constexpr real_rvalue_const_reference real() const && noexcept;

        imag_reference imag() & noexcept;
        imag_rvalue_reference imag() && noexcept;
        constexpr imag_const_reference imag() const & noexcept;
        constexpr imag_rvalue_const_reference imag() const && noexcept;

        xclosure_pointer<self_type&> operator&() & noexcept;
        xclosure_pointer<const self_type&> operator&() const & noexcept;
        xclosure_pointer<self_type> operator&() && noexcept;

    private:

        CTR m_real;
        CTI m_imag;
    };

    /**********************
     * xcomplex operators *
     **********************/

    template <class CTR1, class CTI1, bool B1, class CTR2, class CTI2, bool B2>
    bool operator==(const xcomplex<CTR1, CTI1, B1>& lhs, const xcomplex<CTR2, CTI2, B2>& rhs) noexcept;

    template <class CTR1, class CTI1, bool B1, class CTR2, class CTI2, bool B2>
    bool operator!=(const xcomplex<CTR1, CTI1, B1>& lhs, const xcomplex<CTR2, CTI2, B2>& rhs) noexcept;

    template <class OC, class OT, class CTR, class CTI, bool B>
    std::basic_ostream<OC, OT>& operator<<(std::basic_ostream<OC, OT>& out, const xcomplex<CTR, CTI, B>& c) noexcept;

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    operator+(const xcomplex<CTR, CTI, B>& rhs) noexcept;

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    operator-(const xcomplex<CTR, CTI, B>& rhs) noexcept;

    template <class CTR1, class CTI1, bool B1, class CTR2, class CTI2, bool B2>
    common_xcomplex_t<CTR1, CTI1, B1, CTR2, CTI2, B2>
    operator+(const xcomplex<CTR1, CTI1, B1>& lhs, const xcomplex<CTR2, CTI2, B2>& rhs) noexcept;

    template <class CTR, class CTI, bool B, class T>
    enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    operator+(const xcomplex<CTR, CTI, B>& lhs, const T& rhs) noexcept;

    template <class CTR, class CTI, bool B, class T>
    enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    operator+(const T& lhs, const xcomplex<CTR, CTI, B>& rhs) noexcept;

    template <class CTR1, class CTI1, bool B1, class CTR2, class CTI2, bool B2>
    common_xcomplex_t<CTR1, CTI1, B1, CTR2, CTI2, B2>
    operator-(const xcomplex<CTR1, CTI1, B1>& lhs, const xcomplex<CTR2, CTI2, B2>& rhs) noexcept;

    template <class CTR, class CTI, bool B, class T>
    enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    operator-(const xcomplex<CTR, CTI, B>& lhs, const T& rhs) noexcept;

    template <class CTR, class CTI, bool B, class T>
    enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    operator-(const T& lhs, const xcomplex<CTR, CTI, B>& rhs) noexcept;

    template <class CTR1, class CTI1, bool B1, class CTR2, class CTI2, bool B2>
    common_xcomplex_t<CTR1, CTI1, B1, CTR2, CTI2, B2>
    operator*(const xcomplex<CTR1, CTI1, B1>& lhs, const xcomplex<CTR2, CTI2, B2>& rhs) noexcept;

    template <class CTR, class CTI, bool B, class T>
    enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    operator*(const xcomplex<CTR, CTI, B>& lhs, const T& rhs) noexcept;

    template <class CTR, class CTI, bool B, class T>
    enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    operator*(const T& lhs, const xcomplex<CTR, CTI, B>& rhs) noexcept;

    template <class CTR1, class CTI1, bool B1, class CTR2, class CTI2, bool B2>
    common_xcomplex_t<CTR1, CTI1, B1, CTR2, CTI2, B2>
    operator/(const xcomplex<CTR1, CTI1, B1>& lhs, const xcomplex<CTR2, CTI2, B2>& rhs) noexcept;

    template <class CTR, class CTI, bool B, class T>
    enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    operator/(const xcomplex<CTR, CTI, B>& lhs, const T& rhs) noexcept;

    template <class CTR, class CTI, bool B, class T>
    enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    operator/(const T& lhs, const xcomplex<CTR, CTI, B>& rhs) noexcept;

    /*****************
     * real and imag *
     *****************/

    template <class E>
    decltype(auto) real(E&& e) noexcept;

    template <class E>
    decltype(auto) imag(E&& e) noexcept;

    /***************************
     * xcomplex free functions *
     ***************************/

    template <class CTR, class CTI, bool B>
    typename xcomplex<CTR, CTI, B>::value_type
    abs(const xcomplex<CTR, CTI, B>& rhs);

    template <class CTR, class CTI, bool B>
    typename xcomplex<CTR, CTI, B>::value_type
    arg(const xcomplex<CTR, CTI, B>& rhs);

    template <class CTR, class CTI, bool B>
    typename xcomplex<CTR, CTI, B>::value_type
    norm(const xcomplex<CTR, CTI, B>& rhs);

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    conj(const xcomplex<CTR, CTI, B>& rhs);

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    proj(const xcomplex<CTR, CTI, B>& rhs);

    /**********************************
     * xcomplex exponential functions *
     **********************************/

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    exp(const xcomplex<CTR, CTI, B>& rhs);

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    log(const xcomplex<CTR, CTI, B>& rhs);

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    log10(const xcomplex<CTR, CTI, B>& rhs);

    /****************************
     * xcomplex power functions *
     ****************************/

    template <class CTR1, class CTI1, bool B1, class CTR2, class CTI2, bool B2>
    common_xcomplex_t<CTR1, CTI1, B1, CTR2, CTI2, B2>
    pow(const xcomplex<CTR1, CTI1, B1>& x, const xcomplex<CTR2, CTI2, B2>& y);

    template <class CTR, class CTI, bool B, class T>
    enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    pow(const xcomplex<CTR, CTI, B>& x, const T& y);

    template <class CTR, class CTI, bool B, class T>
    enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    pow(const T& x, const xcomplex<CTR, CTI, B>& y);

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    sqrt(const xcomplex<CTR, CTI, B>& x);

    /************************************
     * xcomplex trigonometric functions *
     ************************************/

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    sin(const xcomplex<CTR, CTI, B>& x);

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    cos(const xcomplex<CTR, CTI, B>& x);

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    tan(const xcomplex<CTR, CTI, B>& x);

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    asin(const xcomplex<CTR, CTI, B>& x);

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    acos(const xcomplex<CTR, CTI, B>& x);

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    atan(const xcomplex<CTR, CTI, B>& x);

    /*********************************
     * xcomplex hyperbolic functions *
     *********************************/

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    sinh(const xcomplex<CTR, CTI, B>& x);

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    cosh(const xcomplex<CTR, CTI, B>& x);

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    tanh(const xcomplex<CTR, CTI, B>& x);

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    asinh(const xcomplex<CTR, CTI, B>& x);

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    acosh(const xcomplex<CTR, CTI, B>& x);

    template <class CTR, class CTI, bool B>
    temporary_xcomplex_t<CTR, CTI, B>
    atanh(const xcomplex<CTR, CTI, B>& x);

    /***************************
     * xcomplex implementation *
     ***************************/

    template <class CTR, class CTI, bool B>
    template <class OCTR>
    inline auto xcomplex<CTR, CTI, B>::operator=(OCTR&& rhs) noexcept -> disable_xcomplex<OCTR, self_type&>
    {
        m_real = std::forward<OCTR>(rhs);
        m_imag = std::decay_t<CTI>();
        return *this;
    }

    template <class CTR, class CTI, bool B>
    template <class OCTR, class OCTI, bool OB>
    inline auto xcomplex<CTR, CTI, B>::operator=(const xcomplex<OCTR, OCTI, OB>& rhs) noexcept -> self_type&
    {
        m_real = rhs.m_real;
        m_imag = rhs.m_imag;
        return *this;
    }

    template <class CTR, class CTI, bool B>
    template <class OCTR, class OCTI, bool OB>
    inline auto xcomplex<CTR, CTI, B>::operator=(xcomplex<OCTR, OCTI, OB>&& rhs) noexcept -> self_type&
    {
        m_real = std::move(rhs.m_real);
        m_imag = std::move(rhs.m_imag);
        return *this;
    }

    template <class CTR, class CTI, bool B>
    inline xcomplex<CTR, CTI, B>::operator std::complex<std::decay_t<CTR>>() const noexcept
    {
        return std::complex<std::decay_t<CTR>>(m_real, m_imag);
    }

    template <class CTR, class CTI, bool B>
    template <class OCTR, class OCTI, bool OB>
    inline auto xcomplex<CTR, CTI, B>::operator+=(const xcomplex<OCTR, OCTI, OB>& rhs) noexcept -> self_type&
    {
        m_real += rhs.m_real;
        m_imag += rhs.m_imag;
        return *this;
    }

    template <class CTR, class CTI, bool B>
    template <class OCTR, class OCTI, bool OB>
    inline auto xcomplex<CTR, CTI, B>::operator-=(const xcomplex<OCTR, OCTI, OB>& rhs) noexcept -> self_type&
    {
        m_real -= rhs.m_real;
        m_imag -= rhs.m_imag;
        return *this;
    }

    namespace detail
    {
        template <bool ieee_compliant>
        struct xcomplex_multiplier
        {
            template <class CTR1, class CTI1, bool B1, class CTR2, class CTI2, bool B2>
            static auto mul(const xcomplex<CTR1, CTI1, B1>& lhs, const xcomplex<CTR2, CTI2, B2>& rhs)
            {
                using return_type = temporary_xcomplex_t<CTR1, CTI1, B1>;
                using value_type = typename return_type::value_type;
                value_type a = lhs.real();
                value_type b = lhs.imag();
                value_type c = rhs.real();
                value_type d = rhs.imag();
                return return_type(a*c - b*d, a*d + b*c);
            }

            template <class CTR1, class CTI1, bool B1, class CTR2, class CTI2, bool B2>
            static auto div(const xcomplex<CTR1, CTI1, B1>& lhs, const xcomplex<CTR2, CTI2, B2>& rhs)
            {
                using return_type = temporary_xcomplex_t<CTR1, CTI1, B1>;
                using value_type = typename return_type::value_type;
                value_type a = lhs.real();
                value_type b = lhs.imag();
                value_type c = rhs.real();
                value_type d = rhs.imag();
                value_type e = c*c + d*d;
                return return_type((c*a + d*b) / e, (c*b - d*a) / e);
            }
        };

        template <>
        struct xcomplex_multiplier<true>
        {
            template <class CTR1, class CTI1, bool B1, class CTR2, class CTI2, bool B2>
            static auto mul(const xcomplex<CTR1, CTI1, B1>& lhs, const xcomplex<CTR2, CTI2, B2>& rhs)
            {
                using return_type = temporary_xcomplex_t<CTR1, CTI1, B1>;
                using value_type = typename return_type::value_type;
                value_type a = lhs.real();
                value_type b = lhs.imag();
                value_type c = rhs.real();
                value_type d = rhs.imag();
                value_type ac = a * c;
                value_type bd = b * d;
                value_type ad = a * d;
                value_type bc = b * c;
                value_type x = ac - bd;
                value_type y = ad + bc;
                if (std::isnan(x) && std::isnan(y))
                {
                    bool recalc = false;
                    if (std::isinf(a) || std::isinf(b))
                    {
                        a = copysign(std::isinf(a) ? value_type(1) : value_type(0), a);
                        b = copysign(std::isinf(b) ? value_type(1) : value_type(0), b);
                        if (std::isnan(c))
                        {
                            c = copysign(value_type(0), c);
                        }
                        if (std::isnan(d))
                        {
                            d = copysign(value_type(0), d);
                        }
                        recalc = true;
                    }
                    if (std::isinf(c) || std::isinf(d))
                    {
                        c = copysign(std::isinf(c) ? value_type(1) : value_type(0), c);
                        d = copysign(std::isinf(c) ? value_type(1) : value_type(0), d);
                        if (std::isnan(a))
                        {
                            a = copysign(value_type(0), a);
                        }
                        if (std::isnan(b))
                        {
                            b = copysign(value_type(0), b);
                        }
                        recalc = true;
                    }
                    if (!recalc && (std::isinf(ac) || std::isinf(bd) || std::isinf(ad) || std::isinf(bc)))
                    {
                        if (std::isnan(a))
                        {
                            a = copysign(value_type(0), a);
                        }
                        if (std::isnan(b))
                        {
                            b = copysign(value_type(0), b);
                        }
                        if (std::isnan(c))
                        {
                            c = copysign(value_type(0), c);
                        }
                        if (std::isnan(d))
                        {
                            d = copysign(value_type(0), d);
                        }
                        recalc = true;
                    }
                    if (recalc)
                    {
                        x = std::numeric_limits<value_type>::infinity() * (a * c - b * d);
                        y = std::numeric_limits<value_type>::infinity() * (a * d + b * c);
                    }
                }
                return return_type(x, y);
            }

            template <class CTR1, class CTI1, bool B1, class CTR2, class CTI2, bool B2>
            static auto div(const xcomplex<CTR1, CTI1, B1>& lhs, const xcomplex<CTR2, CTI2, B2>& rhs)
            {
                using return_type = temporary_xcomplex_t<CTR1, CTI1, B1>;
                using value_type = typename return_type::value_type;
                value_type a = lhs.real();
                value_type b = lhs.imag();
                value_type c = rhs.real();
                value_type d = rhs.imag();
                value_type logbw = std::logb(std::fmax(std::fabs(c), std::fabs(d)));
                int ilogbw = 0;
                if (std::isfinite(logbw))
                {
                    ilogbw = static_cast<int>(logbw);
                    c = std::scalbn(c, -ilogbw);
                    d = std::scalbn(d, -ilogbw);
                }
                value_type denom = c*c + d*d;
                value_type x = std::scalbn((a*c + b*d) / denom, -ilogbw);
                value_type y = std::scalbn((b*c - a*d) / denom, -ilogbw);
                if (std::isnan(x) && std::isnan(y))
                {
                    if ((denom == value_type(0)) && (!std::isnan(a) || !std::isnan(b)))
                    {
                        x = copysign(std::numeric_limits<value_type>::infinity(), c) * a;
                        y = copysign(std::numeric_limits<value_type>::infinity(), c) * b;
                    }
                    else if ((std::isinf(a) || std::isinf(b)) && std::isfinite(c) && std::isfinite(d))
                    {
                        a = copysign(std::isinf(a) ? value_type(1) : value_type(0), a);
                        b = copysign(std::isinf(b) ? value_type(1) : value_type(0), b);
                        x = std::numeric_limits<value_type>::infinity() * (a*c + b*d);
                        y = std::numeric_limits<value_type>::infinity() * (b*c - a*d);
                    }
                    else if (std::isinf(logbw) && logbw > value_type(0) && std::isfinite(a) && std::isfinite(b))
                    {
                        c = copysign(std::isinf(c) ? value_type(1) : value_type(0), c);
                        d = copysign(std::isinf(d) ? value_type(1) : value_type(0), d);
                        x = value_type(0) * (a*c + b*d);
                        y = value_type(0) * (b*c - a*d);
                    }
                }
                return std::complex<value_type>(x, y);
            }
        };
    }

    template <class CTR, class CTI, bool B>
    template <class OCTR, class OCTI, bool OB>
    inline auto xcomplex<CTR, CTI, B>::operator*=(const xcomplex<OCTR, OCTI, OB>& rhs) noexcept -> self_type&
    {
        *this = detail::xcomplex_multiplier<B || OB>::mul(*this, rhs);
        return *this;
    }

    template <class CTR, class CTI, bool B>
    template <class OCTR, class OCTI, bool OB>
    inline auto xcomplex<CTR, CTI, B>::operator/=(const xcomplex<OCTR, OCTI, OB>& rhs) noexcept -> self_type&
    {
        *this = detail::xcomplex_multiplier<B || OB>::div(*this, rhs);
        return *this;
    }

    template <class CTR, class CTI, bool B>
    template <class T>
    inline auto xcomplex<CTR, CTI, B>::operator+=(const T& rhs) noexcept -> disable_xcomplex<T, self_type&>
    {
        m_real += rhs;
        return *this;
    }

    template <class CTR, class CTI, bool B>
    template <class T>
    inline auto xcomplex<CTR, CTI, B>::operator-=(const T& rhs) noexcept -> disable_xcomplex<T, self_type&>
    {
        m_real -= rhs;
        return *this;
    }

    template <class CTR, class CTI, bool B>
    template <class T>
    inline auto xcomplex<CTR, CTI, B>::operator*=(const T& rhs) noexcept -> disable_xcomplex<T, self_type&>
    {
        m_real *= rhs;
        m_imag *= rhs;
        return *this;
    }

    template <class CTR, class CTI, bool B>
    template <class T>
    inline auto xcomplex<CTR, CTI, B>::operator/=(const T& rhs) noexcept -> disable_xcomplex<T, self_type&>
    {
        m_real /= rhs;
        m_imag /= rhs;
        return *this;
    }

    template <class CTR, class CTI, bool B>
    auto xcomplex<CTR, CTI, B>::real() & noexcept -> real_reference
    {
        return m_real;
    }

    template <class CTR, class CTI, bool B>
    auto xcomplex<CTR, CTI, B>::real() && noexcept -> real_rvalue_reference
    {
        return m_real;
    }

    template <class CTR, class CTI, bool B>
    constexpr auto xcomplex<CTR, CTI, B>::real() const & noexcept -> real_const_reference
    {
        return m_real;
    }

    template <class CTR, class CTI, bool B>
    constexpr auto xcomplex<CTR, CTI, B>::real() const && noexcept -> real_rvalue_const_reference
    {
        return m_real;
    }

    template <class CTR, class CTI, bool B>
    auto xcomplex<CTR, CTI, B>::imag() & noexcept -> imag_reference
    {
        return m_imag;
    }

    template <class CTR, class CTI, bool B>
    auto xcomplex<CTR, CTI, B>::imag() && noexcept -> imag_rvalue_reference
    {
        return m_imag;
    }

    template <class CTR, class CTI, bool B>
    constexpr auto xcomplex<CTR, CTI, B>::imag() const & noexcept -> imag_const_reference
    {
        return m_imag;
    }

    template <class CTR, class CTI, bool B>
    constexpr auto xcomplex<CTR, CTI, B>::imag() const && noexcept -> imag_rvalue_const_reference
    {
        return m_imag;
    }

    template <class CTR, class CTI, bool B>
    inline auto xcomplex<CTR, CTI, B>::operator&() & noexcept -> xclosure_pointer<self_type&>
    {
        return xclosure_pointer<self_type&>(*this);
    }

    template <class CTR, class CTI, bool B>
    inline auto xcomplex<CTR, CTI, B>::operator&() const & noexcept -> xclosure_pointer<const self_type&>
    {
        return xclosure_pointer<const self_type&>(*this);
    }

    template <class CTR, class CTI, bool B>
    inline auto xcomplex<CTR, CTI, B>::operator&() && noexcept -> xclosure_pointer<self_type>
    {
        return xclosure_pointer<self_type>(std::move(*this));
    }

    /*************************************
     * xcomplex operators implementation *
     *************************************/

    template <class CTR1, class CTI1, bool B1, class CTR2, class CTI2, bool B2>
    inline bool operator==(const xcomplex<CTR1, CTI1, B1>& lhs, const xcomplex<CTR2, CTI2, B2>& rhs) noexcept
    {
        return lhs.real() == rhs.real() && lhs.imag() == rhs.imag();
    }

    template <class CTR1, class CTI1, bool B1, class CTR2, class CTI2, bool B2>
    inline bool operator!=(const xcomplex<CTR1, CTI1, B1>& lhs, const xcomplex<CTR2, CTI2, B2>& rhs) noexcept
    {
        return !(lhs == rhs);
    }

    template <class OC, class OT, class CTR, class CTI, bool B>
    inline std::basic_ostream<OC, OT>& operator<<(std::basic_ostream<OC, OT>& out, const xcomplex<CTR, CTI, B>& c) noexcept
    {
        out << "(" << c.real() << "," << c.imag() << ")";
        return out;
    }

#ifdef __CLING__
    template <class CTR, class CTI, bool B>
    nlohmann::json mime_bundle_repr(const xcomplex<CTR, CTI, B>& c)
    {
        auto bundle = nlohmann::json::object();
        std::stringstream tmp;
        tmp << c;
        bundle["text/plain"] = tmp.str();
        return bundle;
    }
#endif

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    operator+(const xcomplex<CTR, CTI, B>& rhs) noexcept
    {
        return rhs;
    }

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    operator-(const xcomplex<CTR, CTI, B>& rhs) noexcept
    {
        return temporary_xcomplex_t<CTR, CTI, B>(-rhs.real(), -rhs.imag());
    }

    template <class CTR1, class CTI1, bool B1, class CTR2, class CTI2, bool B2>
    inline common_xcomplex_t<CTR1, CTI1, B1, CTR2, CTI2, B2>
    operator+(const xcomplex<CTR1, CTI1, B1>& lhs, const xcomplex<CTR2, CTI2, B2>& rhs) noexcept
    {
        common_xcomplex_t<CTR1, CTI1, B1, CTR2, CTI2, B2> res(lhs);
        res += rhs;
        return res;
    }

    template <class CTR, class CTI, bool B, class T>
    inline enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    operator+(const xcomplex<CTR, CTI, B>& lhs, const T& rhs) noexcept
    {
        temporary_xcomplex_t<CTR, CTI, B> res(lhs);
        res += rhs;
        return res;
    }

    template <class CTR, class CTI, bool B, class T>
    inline enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    operator+(const T& lhs, const xcomplex<CTR, CTI, B>& rhs) noexcept
    {
        temporary_xcomplex_t<CTR, CTI, B> res(lhs);
        res += rhs;
        return res;
    }

    template <class CTR1, class CTI1, bool B1, class CTR2, class CTI2, bool B2>
    inline common_xcomplex_t<CTR1, CTI1, B1, CTR2, CTI2, B2>
    operator-(const xcomplex<CTR1, CTI1, B1>& lhs, const xcomplex<CTR2, CTI2, B2>& rhs) noexcept
    {
        common_xcomplex_t<CTR1, CTI1, B1, CTR2, CTI2, B2> res(lhs);
        res -= rhs;
        return res;
    }

    template <class CTR, class CTI, bool B, class T>
    inline enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    operator-(const xcomplex<CTR, CTI, B>& lhs, const T& rhs) noexcept
    {
        temporary_xcomplex_t<CTR, CTI, B> res(lhs);
        res -= rhs;
        return res;
    }

    template <class CTR, class CTI, bool B, class T>
    inline enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    operator-(const T& lhs, const xcomplex<CTR, CTI, B>& rhs) noexcept
    {
        temporary_xcomplex_t<CTR, CTI, B> res(lhs);
        res -= rhs;
        return res;
    }

    template <class CTR1, class CTI1, bool B1, class CTR2, class CTI2, bool B2>
    inline common_xcomplex_t<CTR1, CTI1, B1, CTR2, CTI2, B2>
    operator*(const xcomplex<CTR1, CTI1, B1>& lhs, const xcomplex<CTR2, CTI2, B2>& rhs) noexcept
    {
        common_xcomplex_t<CTR1, CTI1, B1, CTR2, CTI2, B2> res(lhs);
        res *= rhs;
        return res;
    }

    template <class CTR, class CTI, bool B, class T>
    inline enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    operator*(const xcomplex<CTR, CTI, B>& lhs, const T& rhs) noexcept
    {
        temporary_xcomplex_t<CTR, CTI, B> res(lhs);
        res *= rhs;
        return res;
    }

    template <class CTR, class CTI, bool B, class T>
    inline enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    operator*(const T& lhs, const xcomplex<CTR, CTI, B>& rhs) noexcept
    {
        temporary_xcomplex_t<CTR, CTI, B> res(lhs);
        res *= rhs;
        return res;
    }

    template <class CTR1, class CTI1, bool B1, class CTR2, class CTI2, bool B2>
    inline common_xcomplex_t<CTR1, CTI1, B1, CTR2, CTI2, B2>
    operator/(const xcomplex<CTR1, CTI1, B1>& lhs, const xcomplex<CTR2, CTI2, B2>& rhs) noexcept
    {
        common_xcomplex_t<CTR1, CTI1, B1, CTR2, CTI2, B2> res(lhs);
        res /= rhs;
        return res;
    }

    template <class CTR, class CTI, bool B, class T>
    inline enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    operator/(const xcomplex<CTR, CTI, B>& lhs, const T& rhs) noexcept
    {
        temporary_xcomplex_t<CTR, CTI, B> res(lhs);
        res /= rhs;
        return res;
    }

    template <class CTR, class CTI, bool B, class T>
    inline enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    operator/(const T& lhs, const xcomplex<CTR, CTI, B>& rhs) noexcept
    {
        temporary_xcomplex_t<CTR, CTI, B> res(lhs);
        res /= rhs;
        return res;
    }

    /***************************
     * xcomplex free functions *
     ***************************/

    template <class CTR, class CTI, bool B>
    inline typename xcomplex<CTR, CTI, B>::value_type
    abs(const xcomplex<CTR, CTI, B>& rhs)
    {
        using value_type = typename xcomplex<CTR, CTI, B>::value_type;
        return std::abs(std::complex<value_type>(rhs));
    }

    template <class CTR, class CTI, bool B>
    inline typename xcomplex<CTR, CTI, B>::value_type
    arg(const xcomplex<CTR, CTI, B>& rhs)
    {
        using value_type = typename xcomplex<CTR, CTI, B>::value_type;
        return std::arg(std::complex<value_type>(rhs));
    }

    template <class CTR, class CTI, bool B>
    inline typename xcomplex<CTR, CTI, B>::value_type
    norm(const xcomplex<CTR, CTI, B>& rhs)
    {
        using value_type = typename xcomplex<CTR, CTI, B>::value_type;
        return std::norm(std::complex<value_type>(rhs));
    }

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    conj(const xcomplex<CTR, CTI, B>& rhs)
    {
        using value_type = typename xcomplex<CTR, CTI, B>::value_type;
        return std::conj(std::complex<value_type>(rhs));
    }

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    proj(const xcomplex<CTR, CTI, B>& rhs)
    {
        using value_type = typename xcomplex<CTR, CTI, B>::value_type;
        return std::proj(std::complex<value_type>(rhs));
    }

    /*************************************************
     * xcomplex exponential functions implementation *
     *************************************************/

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    exp(const xcomplex<CTR, CTI, B>& rhs)
    {
        using value_type = typename xcomplex<CTR, CTI, B>::value_type;
        return std::exp(std::complex<value_type>(rhs));
    }

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    log(const xcomplex<CTR, CTI, B>& rhs)
    {
        using value_type = typename xcomplex<CTR, CTI, B>::value_type;
        return std::log(std::complex<value_type>(rhs));
    }

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    log10(const xcomplex<CTR, CTI, B>& rhs)
    {
        using value_type = typename xcomplex<CTR, CTI, B>::value_type;
        return std::log10(std::complex<value_type>(rhs));
    }

    /*******************************************
     * xcomplex power functions implementation *
     *******************************************/

    template <class CTR1, class CTI1, bool B1, class CTR2, class CTI2, bool B2>
    inline common_xcomplex_t<CTR1, CTI1, B1, CTR2, CTI2, B2>
    pow(const xcomplex<CTR1, CTI1, B1>& x, const xcomplex<CTR2, CTI2, B2>& y)
    {
        using value_type1 = typename xcomplex<CTR1, CTI1, B1>::value_type;
        using value_type2 = typename xcomplex<CTR2, CTI2, B2>::value_type;
        return std::pow(std::complex<value_type1>(x), std::complex<value_type2>(y));
    }

    template <class CTR, class CTI, bool B, class T>
    inline enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    pow(const xcomplex<CTR, CTI, B>& x, const T& y)
    {
        using value_type = typename xcomplex<CTR, CTI, B>::value_type;
        return std::pow(std::complex<value_type>(x), y);
    }

    template <class CTR, class CTI, bool B, class T>
    inline enable_scalar<T, temporary_xcomplex_t<CTR, CTI, B>>
    pow(const T& x, const xcomplex<CTR, CTI, B>& y)
    {
        using value_type = typename xcomplex<CTR, CTI, B>::value_type;
        return std::pow(x, std::complex<value_type>(y));
    }

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    sqrt(const xcomplex<CTR, CTI, B>& x)
    {
        using value_type = typename xcomplex<CTR, CTI, B>::value_type;
        return std::sqrt(std::complex<value_type>(x));
    }

    /***************************************************
     * xcomplex trigonometric functions implementation *
     ***************************************************/

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    sin(const xcomplex<CTR, CTI, B>& x)
    {
        using value_type = typename temporary_xcomplex_t<CTR, CTI, B>::value_type;
        return std::sin(std::complex<value_type>(x));
    }

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    cos(const xcomplex<CTR, CTI, B>& x)
    {
        using value_type = typename temporary_xcomplex_t<CTR, CTI, B>::value_type;
        return std::cos(std::complex<value_type>(x));
    }

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    tan(const xcomplex<CTR, CTI, B>& x)
    {
        using value_type = typename temporary_xcomplex_t<CTR, CTI, B>::value_type;
        return std::tan(std::complex<value_type>(x));
    }

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    asin(const xcomplex<CTR, CTI, B>& x)
    {
        using value_type = typename temporary_xcomplex_t<CTR, CTI, B>::value_type;
        return std::asin(std::complex<value_type>(x));
    }

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    acos(const xcomplex<CTR, CTI, B>& x)
    {
        using value_type = typename temporary_xcomplex_t<CTR, CTI, B>::value_type;
        return std::acos(std::complex<value_type>(x));
    }

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    atan(const xcomplex<CTR, CTI, B>& x)
    {
        using value_type = typename temporary_xcomplex_t<CTR, CTI, B>::value_type;
        return std::atan(std::complex<value_type>(x));
    }

    /************************************************
     * xcomplex hyperbolic functions implementation *
     ************************************************/

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    sinh(const xcomplex<CTR, CTI, B>& x)
    {
        using value_type = typename temporary_xcomplex_t<CTR, CTI, B>::value_type;
        return std::sinh(std::complex<value_type>(x));
    }

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    cosh(const xcomplex<CTR, CTI, B>& x)
    {
        using value_type = typename temporary_xcomplex_t<CTR, CTI, B>::value_type;
        return std::cosh(std::complex<value_type>(x));
    }

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    tanh(const xcomplex<CTR, CTI, B>& x)
    {
        using value_type = typename temporary_xcomplex_t<CTR, CTI, B>::value_type;
        return std::tanh(std::complex<value_type>(x));
    }

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    asinh(const xcomplex<CTR, CTI, B>& x)
    {
        using value_type = typename temporary_xcomplex_t<CTR, CTI, B>::value_type;
        return std::asinh(std::complex<value_type>(x));
    }

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    acosh(const xcomplex<CTR, CTI, B>& x)
    {
        using value_type = typename temporary_xcomplex_t<CTR, CTI, B>::value_type;
        return std::acosh(std::complex<value_type>(x));
    }

    template <class CTR, class CTI, bool B>
    inline temporary_xcomplex_t<CTR, CTI, B>
    atanh(const xcomplex<CTR, CTI, B>& x)
    {
        using value_type = typename temporary_xcomplex_t<CTR, CTI, B>::value_type;
        return std::atanh(std::complex<value_type>(x));
    }

    /*********************************
     * forward_offset implementation *
     *********************************/

    namespace detail
    {

        template <class T, class M>
        struct forward_type
        {
            using type = apply_cv_t<T, M>;
        };

        template <class T, class M>
        struct forward_type<T&, M>
        {
            using type = apply_cv_t<T, M>&;
        };

        template <class T, class M>
        using forward_type_t = typename forward_type<T, M>::type;
    }

    template <class M, std::size_t I, class T>
    constexpr detail::forward_type_t<T, M> forward_offset(T&& v) noexcept
    {
        using forward_type = detail::forward_type_t<T, M>;
        using cv_value_type = std::remove_reference_t<forward_type>;
        using byte_type = apply_cv_t<std::remove_reference_t<T>, char>;

        return static_cast<forward_type>(
            *reinterpret_cast<cv_value_type*>(
                reinterpret_cast<byte_type*>(&v) + I
            )
        );
    }

    /**********************************************
     * forward_real & forward_imag implementation *
     **********************************************/

    // forward_real

    template <class T>
    auto forward_real(T&& v)
        -> std::enable_if_t<!is_gen_complex<T>::value, detail::forward_type_t<T, T>>  // real case -> forward
    {
        return static_cast<detail::forward_type_t<T, T>>(v);
    }

    template <class T>
    auto forward_real(T&& v)
        -> std::enable_if_t<is_complex<T>::value, detail::forward_type_t<T, typename std::decay_t<T>::value_type>>  // complex case -> forward the real part
    {
        return forward_offset<typename std::decay_t<T>::value_type, 0>(v);
    }

    template <class T>
    auto forward_real(T&& v)
        -> std::enable_if_t<is_xcomplex<T>::value, decltype(std::forward<T>(v).real())>
    {
        return std::forward<T>(v).real();
    }

    // forward_imag

    template <class T>
    auto forward_imag(T &&)
        -> std::enable_if_t<!is_gen_complex<T>::value, std::decay_t<T>>  // real case -> always return 0 by value
    {
        return 0;
    }

    template <class T>
    auto forward_imag(T&& v)
        -> std::enable_if_t<is_complex<T>::value, detail::forward_type_t<T, typename std::decay_t<T>::value_type>>  // complex case -> forwards the imaginary part
    {
        using real_type = typename std::decay_t<T>::value_type;
        return forward_offset<real_type, sizeof(real_type)>(v);
    }

    template <class T>
    auto forward_imag(T&& v)
        -> std::enable_if_t<is_xcomplex<T>::value, decltype(std::forward<T>(v).imag())>
    {
        return std::forward<T>(v).imag();
    }

    /******************************
     * real & imag implementation *
     ******************************/

    template <class E>
    inline decltype(auto) real(E&& e) noexcept
    {
        return forward_real(std::forward<E>(e));
    }

    template <class E>
    inline decltype(auto) imag(E&& e) noexcept
    {
        return forward_imag(std::forward<E>(e));
    }

    /**********************
     * complex_value_type *
     **********************/

    template <class T>
    struct complex_value_type
    {
        using type = T;
    };

    template <class T>
    struct complex_value_type<std::complex<T>>
    {
        using type = T;
    };

    template <class CTR, class CTI, bool B>
    struct complex_value_type<xcomplex<CTR, CTI, B>>
    {
        using type = xcomplex<CTR, CTI, B>;
    };

    template <class T>
    using complex_value_type_t = typename complex_value_type<T>::type;

    /******************************************************
     * operator overloads for complex and closure wrapper *
     *****************************************************/

    template <class C, class T, std::enable_if_t<!xtl::is_complex<T>::value, int> = 0>
    std::complex<C> operator+(const std::complex<C>& c, const T& t)
    {
        std::complex<C> result(c);
        result += t;
        return result;
    }

    template <class C, class T, std::enable_if_t<!xtl::is_complex<T>::value, int> = 0>
    std::complex<C> operator+(const T& t, const std::complex<C>& c)
    {
        std::complex<C> result(t);
        result += c;
        return result;
    }

    template <class C, class T, std::enable_if_t<!xtl::is_complex<T>::value, int> = 0>
    std::complex<C> operator-(const std::complex<C>& c, const T& t)
    {
        std::complex<C> result(c);
        result -= t;
        return result;
    }

    template <class C, class T, std::enable_if_t<!xtl::is_complex<T>::value, int> = 0>
    std::complex<C> operator-(const T& t, const std::complex<C>& c)
    {
        std::complex<C> result(t);
        result -= c;
        return result;
    }

    template <class C, class T, std::enable_if_t<!xtl::is_complex<T>::value, int> = 0>
    std::complex<C> operator*(const std::complex<C>& c, const T& t)
    {
        std::complex<C> result(c);
        result *= t;
        return result;
    }

    template <class C, class T, std::enable_if_t<!xtl::is_complex<T>::value, int> = 0>
    std::complex<C> operator*(const T& t, const std::complex<C>& c)
    {
        std::complex<C> result(t);
        result *= c;
        return result;
    }

    template <class C, class T, std::enable_if_t<!xtl::is_complex<T>::value, int> = 0>
    std::complex<C> operator/(const std::complex<C>& c, const T& t)
    {
        std::complex<C> result(c);
        result /= t;
        return result;
    }

    template <class C, class T, std::enable_if_t<!xtl::is_complex<T>::value, int> = 0>
    std::complex<C> operator/(const T& t, const std::complex<C>& c)
    {
        std::complex<C> result(t);
        result /= c;
        return result;
    }
}

#endif
