/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_UTILS_HPP
#define XTENSOR_UTILS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <xtl/xfunctional.hpp>
#include <xtl/xmeta_utils.hpp>
#include <xtl/xsequence.hpp>
#include <xtl/xtype_traits.hpp>

#include "../core/xtensor_config.hpp"

#if (defined(_MSC_VER) && _MSC_VER >= 1910)
#define NOEXCEPT(T)
#else
#define NOEXCEPT(T) noexcept(T)
#endif

namespace xt
{
    /****************
     * declarations *
     ****************/

    template <class T>
    struct remove_class;

    /*template <class F, class... T>
    void for_each(F&& f, std::tuple<T...>& t) noexcept(implementation_dependent);*/

    /*template <class F, class R, class... T>
    R accumulate(F&& f, R init, const std::tuple<T...>& t) noexcept(implementation_dependent);*/

    template <std::size_t I, class... Args>
    constexpr decltype(auto) argument(Args&&... args) noexcept;

    template <class R, class F, class... S>
    R apply(std::size_t index, F&& func, const std::tuple<S...>& s) NOEXCEPT(noexcept(func(std::get<0>(s))));

    template <class T, class S>
    void nested_copy(T&& iter, const S& s);

    template <class T, class S>
    void nested_copy(T&& iter, std::initializer_list<S> s);

    template <class C>
    bool resize_container(C& c, typename C::size_type size);

    template <class T, std::size_t N>
    bool resize_container(std::array<T, N>& a, typename std::array<T, N>::size_type size);

    template <std::size_t... I>
    class fixed_shape;

    template <std::size_t... I>
    bool resize_container(fixed_shape<I...>& a, std::size_t size);

    template <class X, class C>
    struct rebind_container;

    template <class X, class C>
    using rebind_container_t = typename rebind_container<X, C>::type;

    std::size_t normalize_axis(std::size_t dim, std::ptrdiff_t axis);

    // gcc 4.9 is affected by C++14 defect CGW 1558
    // see http://open-std.org/JTC1/SC22/WG21/docs/cwg_defects.html#1558
    template <class... T>
    struct make_void
    {
        using type = void;
    };

    template <class... T>
    using void_t = typename make_void<T...>::type;

    // This is used for non existent types (e.g. storage for some expressions
    // like generators)
    struct invalid_type
    {
    };

    template <class... T>
    struct make_invalid_type
    {
        using type = invalid_type;
    };

    template <class T, class R>
    using disable_integral_t = std::enable_if_t<!xtl::is_integral<T>::value, R>;

    /********************************
     * meta identity implementation *
     ********************************/

    template <class T>
    struct meta_identity
    {
        using type = T;
    };

    /***************************************
     * is_specialization_of implementation *
     ***************************************/

    template <template <class...> class TT, class T>
    struct is_specialization_of : std::false_type
    {
    };

    template <template <class...> class TT, class... Ts>
    struct is_specialization_of<TT, TT<Ts...>> : std::true_type
    {
    };

    /*******************************
     * remove_class implementation *
     *******************************/

    template <class T>
    struct remove_class
    {
    };

    template <class C, class R, class... Args>
    struct remove_class<R (C::*)(Args...)>
    {
        typedef R type(Args...);
    };

    template <class C, class R, class... Args>
    struct remove_class<R (C::*)(Args...) const>
    {
        typedef R type(Args...);
    };

    template <class T>
    using remove_class_t = typename remove_class<T>::type;

    /***************************
     * for_each implementation *
     ***************************/

    namespace detail
    {
        template <std::size_t I, class F, class... T>
        inline typename std::enable_if<I == sizeof...(T), void>::type
        for_each_impl(F&& /*f*/, std::tuple<T...>& /*t*/) noexcept
        {
        }

        template <std::size_t I, class F, class... T>
            inline typename std::enable_if < I<sizeof...(T), void>::type
            for_each_impl(F&& f, std::tuple<T...>& t) noexcept(noexcept(f(std::get<I>(t))))
        {
            f(std::get<I>(t));
            for_each_impl<I + 1, F, T...>(std::forward<F>(f), t);
        }
    }

    template <class F, class... T>
    inline void for_each(F&& f, std::tuple<T...>& t) noexcept(
        noexcept(detail::for_each_impl<0, F, T...>(std::forward<F>(f), t))
    )
    {
        detail::for_each_impl<0, F, T...>(std::forward<F>(f), t);
    }

    namespace detail
    {
        template <std::size_t I, class F, class... T>
        inline typename std::enable_if<I == sizeof...(T), void>::type
        for_each_impl(F&& /*f*/, const std::tuple<T...>& /*t*/) noexcept
        {
        }

        template <std::size_t I, class F, class... T>
            inline typename std::enable_if < I<sizeof...(T), void>::type
            for_each_impl(F&& f, const std::tuple<T...>& t) noexcept(noexcept(f(std::get<I>(t))))
        {
            f(std::get<I>(t));
            for_each_impl<I + 1, F, T...>(std::forward<F>(f), t);
        }
    }

    template <class F, class... T>
    inline void for_each(F&& f, const std::tuple<T...>& t) noexcept(
        noexcept(detail::for_each_impl<0, F, T...>(std::forward<F>(f), t))
    )
    {
        detail::for_each_impl<0, F, T...>(std::forward<F>(f), t);
    }

    /*****************************
     * accumulate implementation *
     *****************************/

    /// @cond DOXYGEN_INCLUDE_NOEXCEPT

    namespace detail
    {
        template <std::size_t I, class F, class R, class... T>
        inline std::enable_if_t<I == sizeof...(T), R>
        accumulate_impl(F&& /*f*/, R init, const std::tuple<T...>& /*t*/) noexcept
        {
            return init;
        }

        template <std::size_t I, class F, class R, class... T>
            inline std::enable_if_t < I<sizeof...(T), R>
            accumulate_impl(F&& f, R init, const std::tuple<T...>& t) noexcept(noexcept(f(init, std::get<I>(t))))
        {
            R res = f(init, std::get<I>(t));
            return accumulate_impl<I + 1, F, R, T...>(std::forward<F>(f), res, t);
        }
    }

    template <class F, class R, class... T>
    inline R accumulate(F&& f, R init, const std::tuple<T...>& t) noexcept(
        noexcept(detail::accumulate_impl<0, F, R, T...>(std::forward<F>(f), init, t))
    )
    {
        return detail::accumulate_impl<0, F, R, T...>(std::forward<F>(f), init, t);
    }

    /// @endcond

    /***************************
     * argument implementation *
     ***************************/

    namespace detail
    {
        template <std::size_t I>
        struct getter
        {
            template <class Arg, class... Args>
            static constexpr decltype(auto) get(Arg&& /*arg*/, Args&&... args) noexcept
            {
                return getter<I - 1>::get(std::forward<Args>(args)...);
            }
        };

        template <>
        struct getter<0>
        {
            template <class Arg, class... Args>
            static constexpr Arg&& get(Arg&& arg, Args&&... /*args*/) noexcept
            {
                return std::forward<Arg>(arg);
            }
        };
    }

    template <std::size_t I, class... Args>
    constexpr decltype(auto) argument(Args&&... args) noexcept
    {
        static_assert(I < sizeof...(Args), "I should be lesser than sizeof...(Args)");
        return detail::getter<I>::get(std::forward<Args>(args)...);
    }

    /************************
     * apply implementation *
     ************************/

    template <class R, class F, class... S>
    inline R apply(std::size_t index, F&& func, const std::tuple<S...>& s)
        NOEXCEPT(noexcept(func(std::get<0>(s))))
    {
        XTENSOR_ASSERT(sizeof...(S) > index);
        return std::apply(
            [&](const S&... args) -> R
            {
                auto f_impl = [&](auto&& self, auto&& i, auto&& h, auto&&... t) -> R
                {
                    if (i == index)
                    {
                        return static_cast<R>(func(h));
                    }
                    if constexpr (sizeof...(t) > 0)
                    {
                        return self(self, std::size_t{i + 1}, t...);
                    }
                    return R{};
                };
                return f_impl(f_impl, std::size_t{0}, args...);
            },
            s
        );
    }

    /***************************
     * nested_initializer_list *
     ***************************/

    template <class T, std::size_t I>
    struct nested_initializer_list
    {
        using type = std::initializer_list<typename nested_initializer_list<T, I - 1>::type>;
    };

    template <class T>
    struct nested_initializer_list<T, 0>
    {
        using type = T;
    };

    template <class T, std::size_t I>
    using nested_initializer_list_t = typename nested_initializer_list<T, I>::type;

    /******************************
     * nested_copy implementation *
     ******************************/

    template <class T, class S>
    inline void nested_copy(T&& iter, const S& s)
    {
        *iter++ = s;
    }

    template <class T, class S>
    inline void nested_copy(T&& iter, std::initializer_list<S> s)
    {
        for (auto it = s.begin(); it != s.end(); ++it)
        {
            nested_copy(std::forward<T>(iter), *it);
        }
    }

    /***********************************
     * resize_container implementation *
     ***********************************/
    template <class C>
    inline bool resize_container(C& c, typename C::size_type size)
    {
        c.resize(size);
        return true;
    }

    template <class T, std::size_t N>
    inline bool resize_container(std::array<T, N>& /*a*/, typename std::array<T, N>::size_type size)
    {
        return size == N;
    }

    template <std::size_t... I>
    inline bool resize_container(xt::fixed_shape<I...>&, std::size_t size)
    {
        return sizeof...(I) == size;
    }

    /*********************************
     * normalize_axis implementation *
     *********************************/

    // scalar normalize axis
    inline std::size_t normalize_axis(std::size_t dim, std::ptrdiff_t axis)
    {
        return axis < 0 ? static_cast<std::size_t>(static_cast<std::ptrdiff_t>(dim) + axis)
                        : static_cast<std::size_t>(axis);
    }

    template <class E, class C>
    inline std::enable_if_t<
        !xtl::is_integral<std::decay_t<C>>::value && xtl::is_signed<typename std::decay_t<C>::value_type>::value,
        rebind_container_t<std::size_t, std::decay_t<C>>>
    normalize_axis(E& expr, C&& axes)
    {
        rebind_container_t<std::size_t, std::decay_t<C>> res;
        resize_container(res, axes.size());

        for (std::size_t i = 0; i < axes.size(); ++i)
        {
            res[i] = normalize_axis(expr.dimension(), axes[i]);
        }

        XTENSOR_ASSERT(std::all_of(
            res.begin(),
            res.end(),
            [&expr](auto ax_el)
            {
                return ax_el < expr.dimension();
            }
        ));

        return res;
    }

    template <class C, class E>
    inline std::enable_if_t<
        !xtl::is_integral<std::decay_t<C>>::value && std::is_unsigned<typename std::decay_t<C>::value_type>::value,
        C&&>
    normalize_axis(E& expr, C&& axes)
    {
        static_cast<void>(expr);
        XTENSOR_ASSERT(std::all_of(
            axes.begin(),
            axes.end(),
            [&expr](auto ax_el)
            {
                return ax_el < expr.dimension();
            }
        ));
        return std::forward<C>(axes);
    }

    template <class R, class E, class C>
    inline auto forward_normalize(E& expr, C&& axes)
        -> std::enable_if_t<xtl::is_signed<std::decay_t<decltype(*std::begin(axes))>>::value, R>
    {
        R res;
        xt::resize_container(res, std::size(axes));
        auto dim = expr.dimension();
        std::transform(
            std::begin(axes),
            std::end(axes),
            std::begin(res),
            [&dim](auto ax_el)
            {
                return normalize_axis(dim, ax_el);
            }
        );

        XTENSOR_ASSERT(std::all_of(
            res.begin(),
            res.end(),
            [&expr](auto ax_el)
            {
                return ax_el < expr.dimension();
            }
        ));

        return res;
    }

    template <class R, class E, class C>
    inline auto forward_normalize(E& expr, C&& axes) -> std::enable_if_t<
        !xtl::is_signed<std::decay_t<decltype(*std::begin(axes))>>::value && !std::is_same<R, std::decay_t<C>>::value,
        R>
    {
        static_cast<void>(expr);

        R res;
        xt::resize_container(res, std::size(axes));
        std::copy(std::begin(axes), std::end(axes), std::begin(res));
        XTENSOR_ASSERT(std::all_of(
            res.begin(),
            res.end(),
            [&expr](auto ax_el)
            {
                return ax_el < expr.dimension();
            }
        ));
        return res;
    }

    template <class R, class E, class C>
    inline auto forward_normalize(E& expr, C&& axes) -> std::enable_if_t<
        !xtl::is_signed<std::decay_t<decltype(*std::begin(axes))>>::value && std::is_same<R, std::decay_t<C>>::value,
        R&&>
    {
        static_cast<void>(expr);
        XTENSOR_ASSERT(std::all_of(
            std::begin(axes),
            std::end(axes),
            [&expr](auto ax_el)
            {
                return ax_el < expr.dimension();
            }
        ));
        return std::move(axes);
    }

    /******************
     * get_value_type *
     ******************/

    template <class T, class = void_t<>>
    struct get_value_type
    {
        using type = T;
    };

    template <class T>
    struct get_value_type<T, void_t<typename T::value_type>>
    {
        using type = typename T::value_type;
    };

    template <class T>
    using get_value_type_t = typename get_value_type<T>::type;

    /**********************
     * get implementation *
     **********************/

    // When subclassing from std::tuple not all compilers are able to correctly instantiate get
    // See here: https://stackoverflow.com/a/37188019/2528668
    template <std::size_t I, template <typename... Args> class T, typename... Args>
    decltype(auto) get(T<Args...>&& v)
    {
        return std::get<I>(static_cast<std::tuple<Args...>&&>(v));
    }

    template <std::size_t I, template <typename... Args> class T, typename... Args>
    decltype(auto) get(T<Args...>& v)
    {
        return std::get<I>(static_cast<std::tuple<Args...>&>(v));
    }

    template <std::size_t I, template <typename... Args> class T, typename... Args>
    decltype(auto) get(const T<Args...>& v)
    {
        return std::get<I>(static_cast<const std::tuple<Args...>&>(v));
    }

    /**************************
     * to_array implementation *
     ***************************/

    namespace detail
    {
        template <class T, std::size_t N, std::size_t... I>
        constexpr std::array<std::remove_cv_t<T>, N> to_array_impl(T (&a)[N], std::index_sequence<I...>)
        {
            return {{a[I]...}};
        }
    }

    template <class T, std::size_t N>
    constexpr std::array<std::remove_cv_t<T>, N> to_array(T (&a)[N])
    {
        return detail::to_array_impl(a, std::make_index_sequence<N>{});
    }

    /***********************************
     * has_storage_type implementation *
     ***********************************/

    template <class T, class = void>
    struct has_storage_type : std::false_type
    {
    };

    template <class T>
    struct xcontainer_inner_types;

    template <class T>
    struct has_storage_type<T, void_t<typename xcontainer_inner_types<T>::storage_type>>
        : std::negation<
              std::is_same<typename std::remove_cv<typename xcontainer_inner_types<T>::storage_type>::type, invalid_type>>
    {
    };

    /*************************************
     * has_data_interface implementation *
     *************************************/

    template <class E, class = void>
    struct has_data_interface : std::false_type
    {
    };

    template <class E>
    struct has_data_interface<E, void_t<decltype(std::declval<E>().data())>> : std::true_type
    {
    };

    template <class E>
    concept has_data_interface_concept = has_data_interface<E>::value;

    template <class E, class = void>
    struct has_strides : std::false_type
    {
    };

    template <class E>
    struct has_strides<E, void_t<decltype(std::declval<E>().strides())>> : std::true_type
    {
    };

    template <class E, class = void>
    struct has_iterator_interface : std::false_type
    {
    };

    template <class E>
    struct has_iterator_interface<E, void_t<decltype(std::declval<E>().begin())>> : std::true_type
    {
    };

    template <class E>
    concept has_iterator_interface_concept = has_iterator_interface<E>::value;

    /******************************
     * is_iterator implementation *
     ******************************/

    template <class E, class = void>
    struct is_iterator : std::false_type
    {
    };

    template <class E>
    struct is_iterator<
        E,
        void_t<
            decltype(*std::declval<const E>(), std::declval<const E>() == std::declval<const E>(), std::declval<const E>() != std::declval<const E>(), ++(*std::declval<E*>()), (*std::declval<E*>())++, std::true_type())>>
        : std::true_type
    {
    };

    template <typename E>
    concept iterator_concept = is_iterator<E>::value;

    /********************************************
     * xtrivial_default_construct implemenation *
     ********************************************/

#if defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE >= 7
// has_trivial_default_constructor has not been available since libstdc++-7.
#define XTENSOR_GLIBCXX_USE_CXX11_ABI 1
#else
#if defined(_GLIBCXX_USE_CXX11_ABI)
#if _GLIBCXX_USE_CXX11_ABI || (defined(_GLIBCXX_USE_DUAL_ABI) && !_GLIBCXX_USE_DUAL_ABI)
#define XTENSOR_GLIBCXX_USE_CXX11_ABI 1
#endif
#endif
#endif

#if !defined(__GNUG__) || defined(_LIBCPP_VERSION) || defined(XTENSOR_GLIBCXX_USE_CXX11_ABI)

    template <class T>
    using xtrivially_default_constructible = std::is_trivially_default_constructible<T>;

#else

    template <class T>
    using xtrivially_default_constructible = std::has_trivial_default_constructor<T>;

#endif
#undef XTENSOR_GLIBCXX_USE_CXX11_ABI

    /*************************
     * conditional type cast *
     *************************/

    template <bool condition, class T>
    struct conditional_cast_functor;

    template <class T>
    struct conditional_cast_functor<false, T> : public xtl::identity
    {
    };

    template <class T>
    struct conditional_cast_functor<true, T>
    {
        template <class U>
        inline auto operator()(U&& u) const
        {
            return static_cast<T>(std::forward<U>(u));
        }
    };

    /**
     * @brief Perform a type cast when a condition is true.
     * If <tt>condition</tt> is true, return <tt>static_cast<T>(u)</tt>,
     * otherwise return <tt>u</tt> unchanged. This is useful when an unconditional
     * static_cast would force undesired type conversions in some situations where
     * an error or warning would be desired. The condition determines when the
     * explicit cast is ok.
     */
    template <bool condition, class T, class U>
    inline auto conditional_cast(U&& u)
    {
        return conditional_cast_functor<condition, T>()(std::forward<U>(u));
    }

    /**********************
     * tracking allocator *
     **********************/

    namespace alloc_tracking
    {
        inline bool& enabled()
        {
            static bool enabled;
            return enabled;
        }

        inline void enable()
        {
            enabled() = true;
        }

        inline void disable()
        {
            enabled() = false;
        }

        enum policy
        {
            print,
            assert
        };
    }

    template <class T, class A, alloc_tracking::policy P>
    struct tracking_allocator : private A
    {
        using base_type = A;
        using value_type = typename A::value_type;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = typename std::allocator_traits<A>::pointer;
        using const_pointer = typename std::allocator_traits<A>::const_pointer;
        using size_type = typename std::allocator_traits<A>::size_type;
        using difference_type = typename std::allocator_traits<A>::difference_type;

        tracking_allocator() = default;

        T* allocate(std::size_t n)
        {
            if (alloc_tracking::enabled())
            {
                if (P == alloc_tracking::print)
                {
                    std::cout << "xtensor allocating: " << n << "" << std::endl;
                }
                else if (P == alloc_tracking::assert)
                {
                    XTENSOR_THROW(
                        std::runtime_error,
                        "xtensor allocation of " + std::to_string(n) + " elements detected"
                    );
                }
            }
            return base_type::allocate(n);
        }

        using base_type::deallocate;

// Construct and destroy are removed in --std=c++-20
#if ((defined(__cplusplus) && __cplusplus < 202002L) || (defined(_MSVC_LANG) && _MSVC_LANG < 202002L))
        using base_type::construct;
        using base_type::destroy;
#endif

        template <class U>
        struct rebind
        {
            using traits = std::allocator_traits<A>;
            using other = tracking_allocator<U, typename traits::template rebind_alloc<U>, P>;
        };
    };

    template <class T, class AT, alloc_tracking::policy PT, class U, class AU, alloc_tracking::policy PU>
    inline bool operator==(const tracking_allocator<T, AT, PT>&, const tracking_allocator<U, AU, PU>&)
    {
        return std::is_same<AT, AU>::value;
    }

    template <class T, class AT, alloc_tracking::policy PT, class U, class AU, alloc_tracking::policy PU>
    inline bool operator!=(const tracking_allocator<T, AT, PT>& a, const tracking_allocator<U, AU, PU>& b)
    {
        return !(a == b);
    }

    /*****************
     * has_assign_to *
     *****************/

    template <class E1, class E2, class = void>
    struct has_assign_to : std::false_type
    {
    };

    template <class E1, class E2>
    struct has_assign_to<E1, E2, void_t<decltype(std::declval<const E2&>().assign_to(std::declval<E1&>()))>>
        : std::true_type
    {
    };

    template <class E1, class E2>
    constexpr bool has_assign_to_v = has_assign_to<E1, E2>::value;

    /*************************************
     * overlapping_memory_checker_traits *
     *************************************/

    template <class T, class Enable = void>
    struct has_memory_address : std::false_type
    {
    };

    template <class T>
    struct has_memory_address<T, void_t<decltype(std::addressof(*std::declval<T>().begin()))>> : std::true_type
    {
    };

    template <typename T>
    concept with_memory_address_concept = has_memory_address<std::decay_t<T>>::value;
    template <typename T>
    concept without_memory_address_concept = !has_memory_address<std::decay_t<T>>::value;

    struct memory_range
    {
        // Checking pointer overlap is more correct in integer values,
        // for more explanation check https://devblogs.microsoft.com/oldnewthing/20170927-00/?p=97095
        const uintptr_t m_first = 0;
        const uintptr_t m_last = 0;

        explicit memory_range() = default;

        template <class T>
        explicit memory_range(T* first, T* last)
            : m_first(reinterpret_cast<uintptr_t>(last < first ? last : first))
            , m_last(reinterpret_cast<uintptr_t>(last < first ? first : last))
        {
        }

        template <class T>
        bool overlaps(T* first, T* last) const
        {
            if (first <= last)
            {
                return reinterpret_cast<uintptr_t>(first) <= m_last
                       && reinterpret_cast<uintptr_t>(last) >= m_first;
            }
            else
            {
                return reinterpret_cast<uintptr_t>(last) <= m_last
                       && reinterpret_cast<uintptr_t>(first) >= m_first;
            }
        }
    };

    template <class E, class Enable = void>
    struct overlapping_memory_checker_traits
    {
        static bool check_overlap(const E&, const memory_range&)
        {
            return true;
        }
    };

    template <class E>
    struct overlapping_memory_checker_traits<E, std::enable_if_t<has_memory_address<E>::value>>
    {
        static bool check_overlap(const E& expr, const memory_range& dst_range)
        {
            if (expr.size() == 0)
            {
                return false;
            }
            else
            {
                return dst_range.overlaps(std::addressof(*expr.begin()), std::addressof(*expr.rbegin()));
            }
        }
    };

    struct overlapping_memory_checker_base
    {
        memory_range m_dst_range;

        explicit overlapping_memory_checker_base() = default;

        explicit overlapping_memory_checker_base(memory_range dst_memory_range)
            : m_dst_range(std::move(dst_memory_range))
        {
        }

        template <class E>
        bool check_overlap(const E& expr) const
        {
            if (!m_dst_range.m_first || !m_dst_range.m_last)
            {
                return false;
            }
            else
            {
                return overlapping_memory_checker_traits<E>::check_overlap(expr, m_dst_range);
            }
        }
    };

    template <class Dst, class Enable = void>
    struct overlapping_memory_checker : overlapping_memory_checker_base
    {
        explicit overlapping_memory_checker(const Dst&)
            : overlapping_memory_checker_base()
        {
        }
    };

    template <class Dst>
    struct overlapping_memory_checker<Dst, std::enable_if_t<has_memory_address<Dst>::value>>
        : overlapping_memory_checker_base
    {
        explicit overlapping_memory_checker(const Dst& aDst)
            : overlapping_memory_checker_base(
                [&]()
                {
                    if (aDst.size() == 0)
                    {
                        return memory_range();
                    }
                    else
                    {
                        return memory_range(std::addressof(*aDst.begin()), std::addressof(*aDst.rbegin()));
                    }
                }()
            )
        {
        }
    };

    template <class Dst>
    auto make_overlapping_memory_checker(const Dst& a_dst)
    {
        return overlapping_memory_checker<Dst>(a_dst);
    }

    /********************
     * rebind_container *
     ********************/

    template <class X, template <class, class> class C, class T, class A>
    struct rebind_container<X, C<T, A>>
    {
        using traits = std::allocator_traits<A>;
        using allocator = typename traits::template rebind_alloc<X>;
        using type = C<X, allocator>;
    };

// Workaround for rebind_container problems when C++17 feature is enabled
#ifdef __cpp_template_template_args
    template <class X, class T, std::size_t N>
    struct rebind_container<X, std::array<T, N>>
    {
        using type = std::array<X, N>;
    };
#else
    template <class X, template <class, std::size_t> class C, class T, std::size_t N>
    struct rebind_container<X, C<T, N>>
    {
        using type = C<X, N>;
    };
#endif

    /********************
     * get_strides_type *
     ********************/

    template <class S>
    struct get_strides_type
    {
        using type = typename rebind_container<std::ptrdiff_t, S>::type;
    };

    template <std::size_t... I>
    struct get_strides_type<fixed_shape<I...>>
    {
        // TODO we could compute the strides statically here.
        //  But we'll need full constexpr support to have a
        //  homogenous ``compute_strides`` method
        using type = std::array<std::ptrdiff_t, sizeof...(I)>;
    };

    template <class CP, class O, class A>
    class xbuffer_adaptor;

    template <class CP, class O, class A>
    struct get_strides_type<xbuffer_adaptor<CP, O, A>>
    {
        // In bindings this mapping is called by reshape_view with an inner shape of type
        // xbuffer_adaptor.
        // Since we cannot create a buffer adaptor holding data, we map it to an std::vector.
        using type = std::vector<
            typename xbuffer_adaptor<CP, O, A>::value_type,
            typename xbuffer_adaptor<CP, O, A>::allocator_type>;
    };


    template <class C>
    using get_strides_t = typename get_strides_type<C>::type;

    /*******************
     * inner_reference *
     *******************/

    template <class ST>
    struct inner_reference
    {
        using storage_type = std::decay_t<ST>;
        using type = std::conditional_t<
            std::is_const<std::remove_reference_t<ST>>::value,
            typename storage_type::const_reference,
            typename storage_type::reference>;
    };

    template <class ST>
    using inner_reference_t = typename inner_reference<ST>::type;

    /************
     * get_rank *
     ************/

    template <class E, typename = void>
    struct get_rank
    {
        static constexpr std::size_t value = SIZE_MAX;
    };

    template <class E>
    struct get_rank<E, decltype((void) E::rank, void())>
    {
        static constexpr std::size_t value = E::rank;
    };

    /******************
     * has_fixed_rank *
     ******************/

    template <class E>
    struct has_fixed_rank
    {
        using type = std::integral_constant<bool, get_rank<std::decay_t<E>>::value != SIZE_MAX>;
    };

    template <class E>
    using has_fixed_rank_t = typename has_fixed_rank<std::decay_t<E>>::type;

    /************
     * has_rank *
     ************/

    template <class E, size_t N>
    struct has_rank
    {
        using type = std::integral_constant<bool, get_rank<std::decay_t<E>>::value == N>;
    };

    template <class E, size_t N>
    using has_rank_t = typename has_rank<std::decay_t<E>, N>::type;

}

#endif
