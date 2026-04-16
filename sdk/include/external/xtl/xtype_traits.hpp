/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_TYPE_TRAITS_HPP
#define XTL_TYPE_TRAITS_HPP

#include <complex>
#include <chrono>
#include <type_traits>

#include "xtl_config.hpp"

namespace xtl
{
    /************************************
     * std proxy traits                 *
     ************************************/

    template <class T>
    struct is_scalar : std::is_scalar<T>
    {
    };

    template <class T>
    struct is_arithmetic : std::is_arithmetic<T>
    {
    };

    template <class T>
    struct is_fundamental : std::is_fundamental<T>
    {
    };

    template <class T>
    struct is_signed : std::is_signed<T>
    {
    };

    template <class T>
    struct is_floating_point : std::is_floating_point<T>
    {
    };

    template <class T>
    struct is_integral : std::is_integral<T>
    {
    };

    /************************************
     * arithmetic type promotion traits *
     ************************************/

    /**
     * Traits class for the result type of mixed arithmetic expressions.
     * For example, <tt>promote_type<unsigned char, unsigned char>::type</tt> tells
     * the user that <tt>unsigned char + unsigned char => int</tt>.
     */
    template <class... T>
    struct promote_type;

    template <>
    struct promote_type<>
    {
        using type = void;
    };

    template <class T>
    struct promote_type<T>
    {
        using type = typename promote_type<T, T>::type;
    };

    template <class C, class D1, class D2>
    struct promote_type<std::chrono::time_point<C, D1>, std::chrono::time_point<C, D2>>
    {
        using type = std::chrono::time_point<C, typename promote_type<D1, D2>::type>;
    };

    template <class T0, class T1>
    struct promote_type<T0, T1>
    {
        using type = decltype(std::declval<std::decay_t<T0>>() + std::declval<std::decay_t<T1>>());
    };

    template <>
    struct promote_type<bool>
    {
        using type = bool;
    };

    template <class T>
    struct promote_type<bool, T>
    {
        using type = T;
    };

    template <class T>
    struct promote_type<bool, std::complex<T>>
    {
        using type = std::complex<T>;
    };

    template <class T1, class T2>
    struct promote_type<T1, std::complex<T2>>
    {
        using type = std::complex<typename promote_type<T1, T2>::type>;
    };

    template <class T1, class T2>
    struct promote_type<std::complex<T1>, T2>
        : promote_type<T2, std::complex<T1>>
    {
    };

    template <class T>
    struct promote_type<std::complex<T>, std::complex<T>>
    {
        using type = std::complex<T>;
    };

    template <class T1, class T2>
    struct promote_type<std::complex<T1>, std::complex<T2>>
    {
        using type = std::complex<typename promote_type<T1, T2>::type>;
    };

    template <class T, class... REST>
    struct promote_type<T, REST...>
    {
        using type = typename promote_type<T, typename promote_type<REST...>::type>::type;
    };

    template <class T0, class T1, class... REST>
    struct promote_type<std::complex<T0>, std::complex<T1>, REST...>
    {
        using type = std::complex<typename promote_type<T0, T1, REST...>::type>;
    };

    /**
     * Abbreviation of 'typename promote_type<T>::type'.
     */
    template <class... T>
    using promote_type_t = typename promote_type<T...>::type;

    /**
     * Traits class to find the biggest type of the same kind.
     *
     * For example, <tt>big_promote_type<unsigned char>::type</tt> is <tt>unsigned long long</tt>.
     * The default implementation only supports built-in types and <tt>std::complex</tt>. All
     * other types remain unchanged unless <tt>big_promote_type</tt> gets specialized for them.
     */
    template <class T>
    struct big_promote_type
    {
    private:

        using V = std::decay_t<T>;
        static constexpr bool is_arithmetic = xtl::is_arithmetic<V>::value;
        static constexpr bool is_signed = xtl::is_signed<V>::value;
        static constexpr bool is_integral = xtl::is_integral<V>::value;
        static constexpr bool is_long_double = std::is_same<V, long double>::value;

    public:

        using type = std::conditional_t<is_arithmetic,
                        std::conditional_t<is_integral,
                            std::conditional_t<is_signed, long long, unsigned long long>,
                            std::conditional_t<is_long_double, long double, double>
                        >,
                        V
                     >;
    };

    template <class T>
    struct big_promote_type<std::complex<T>>
    {
        using type = std::complex<typename big_promote_type<T>::type>;
    };

    /**
     * Abbreviation of 'typename big_promote_type<T>::type'.
     */
    template <class T>
    using big_promote_type_t = typename big_promote_type<T>::type;

    namespace traits_detail
    {
        using std::sqrt;

        template <class T>
        using real_promote_type_t = decltype(sqrt(std::declval<std::decay_t<T>>()));
    }

    /**
     * Result type of algebraic expressions.
     *
     * For example, <tt>real_promote_type<int>::type</tt> tells the
     * user that <tt>sqrt(int) => double</tt>.
     */
    template <class T>
    struct real_promote_type
    {
        using type = traits_detail::real_promote_type_t<T>;
    };

    /**
     * Abbreviation of 'typename real_promote_type<T>::type'.
     */
    template <class T>
    using real_promote_type_t = typename real_promote_type<T>::type;

    /**
     * Traits class to replace 'bool' with 'uint8_t' and keep everything else.
     *
     * This is useful for scientific computing, where a boolean mask array is
     * usually implemented as an array of bytes.
     */
    template <class T>
    struct bool_promote_type
    {
        using type = typename std::conditional<std::is_same<T, bool>::value, uint8_t, T>::type;
    };

    /**
     * Abbreviation for typename bool_promote_type<T>::type
     */
    template <class T>
    using bool_promote_type_t = typename bool_promote_type<T>::type;

    /************
     * apply_cv *
     ************/

    namespace detail
    {
        template <class T, class U, bool = std::is_const<std::remove_reference_t<T>>::value,
                  bool = std::is_volatile<std::remove_reference_t<T>>::value>
        struct apply_cv_impl
        {
            using type = U;
        };

        template <class T, class U>
        struct apply_cv_impl<T, U, true, false>
        {
            using type = const U;
        };

        template <class T, class U>
        struct apply_cv_impl<T, U, false, true>
        {
            using type = volatile U;
        };

        template <class T, class U>
        struct apply_cv_impl<T, U, true, true>
        {
            using type = const volatile U;
        };

        template <class T, class U>
        struct apply_cv_impl<T&, U, false, false>
        {
            using type = U&;
        };

        template <class T, class U>
        struct apply_cv_impl<T&, U, true, false>
        {
            using type = const U&;
        };

        template <class T, class U>
        struct apply_cv_impl<T&, U, false, true>
        {
            using type = volatile U&;
        };

        template <class T, class U>
        struct apply_cv_impl<T&, U, true, true>
        {
            using type = const volatile U&;
        };
    }

    template <class T, class U>
    struct apply_cv
    {
        using type = typename detail::apply_cv_impl<T, U>::type;
    };

    template <class T, class U>
    using apply_cv_t = typename apply_cv<T, U>::type;

    /************
     * concepts *
     ************/

    template <class... C>
    constexpr bool xtl_requires = std::conjunction<C...>::value;

    template <class... C>
    constexpr bool either = std::disjunction<C...>::value;

    template <class... C>
    constexpr bool disallow = std::negation<std::conjunction<C...>>::value;

    template <class... C>
    constexpr bool disallow_one = std::negation<std::disjunction<C...>>::value;

    template <class... C>
    using check_requires = std::enable_if_t<xtl_requires<C...>, int>;

    template <class... C>
    using check_either = std::enable_if_t<either<C...>, int>;

    template <class... C>
    using check_disallow = std::enable_if_t<disallow<C...>, int>;

    template <class... C>
    using check_disallow_one = std::enable_if_t<disallow_one<C...>, int>;


#define XTL_REQUIRES_IMPL(...) xtl::check_requires<__VA_ARGS__>
#define XTL_REQUIRES(...) XTL_REQUIRES_IMPL(__VA_ARGS__) = 0

#define XTL_EITHER_IMPL(...) xtl::check_either<__VA_ARGS__>
#define XTL_EITHER(...) XTL_EITHER_IMPL(__VA_ARGS__) = 0

#define XTL_DISALLOW_IMPL(...) xtl::check_disallow<__VA_ARGS__>
#define XTL_DISALLOW(...) XTL_DISALLOW_IMPL(__VA_ARGS__) = 0

#define XTL_DISALLOW_ONE_IMPL(...) xtl::check_disallow_one<__VA_ARGS__>
#define XTL_DISALLOW_ONE(...) XTL_DISALLOW_ONE_IMPL(__VA_ARGS__) = 0

    // For backward compatibility
    template <class... C>
    using check_concept = check_requires<C...>;

    /**************
     * all_scalar *
     **************/

    template <class... Args>
    struct all_scalar : std::conjunction<std::is_scalar<Args>...>
    {
    };

    /************
     * constify *
     ************/

    // Adds const to the underlying type of a reference or pointer, or to the type itself
    // if it's not a reference nor a pointer

    template <class T>
    struct constify
    {
        using type = std::add_const_t<T>;
    };

    template <class T>
    struct constify<T*>
    {
        using type = std::add_const_t<T>*;
    };

    template <class T>
    struct constify<T&>
    {
        using type = std::add_const_t<T>&;
    };

    template <class T>
    using constify_t = typename constify<T>::type;
}

#endif
