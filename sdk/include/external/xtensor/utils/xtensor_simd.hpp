/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_SIMD_HPP
#define XTENSOR_SIMD_HPP

#include <vector>

#include <xtl/xdynamic_bitset.hpp>

#include "../utils/xutils.hpp"

#ifdef XTENSOR_USE_XSIMD

#include <xsimd/xsimd.hpp>
// #include <xsimd/memory/xsimd_load_store.hpp>

#if defined(_MSV_VER) && (_MSV_VER < 1910)
template <class T, class A>
inline xsimd::batch_bool<T, A> isnan(const xsimd::batch<T, A>& b)
{
    return xsimd::isnan(b);
}
#endif

namespace xt_simd
{
    template <class T, std::size_t A>
    using aligned_allocator = xsimd::aligned_allocator<T, A>;

    using aligned_mode = xsimd::aligned_mode;
    using unaligned_mode = xsimd::unaligned_mode;

    template <class A>
    using allocator_alignment = xsimd::allocator_alignment<A>;

    template <class A>
    using allocator_alignment_t = xsimd::allocator_alignment_t<A>;

    template <class C>
    using container_alignment = xsimd::container_alignment<C>;

    template <class C>
    using container_alignment_t = xsimd::container_alignment_t<C>;

    template <class T>
    using simd_traits = xsimd::simd_traits<T>;

    template <class T>
    using revert_simd_traits = xsimd::revert_simd_traits<T>;

    template <class T>
    using simd_type = xsimd::simd_type<T>;

    template <class T>
    using simd_bool_type = xsimd::simd_bool_type<T>;

    template <class T>
    using revert_simd_type = xsimd::revert_simd_type<T>;

    template <class T1, class T2>
    using simd_return_type = xsimd::simd_return_type<T1, T2>;

    using xsimd::broadcast_as;
    using xsimd::get_alignment_offset;
    using xsimd::load_as;
    using xsimd::select;
    using xsimd::store_as;

    template <class V>
    using is_batch_bool = xsimd::is_batch_bool<V>;

    template <class V>
    using is_batch_complex = xsimd::is_batch_complex<V>;

    template <class T1, class T2>
    using simd_condition = xsimd::detail::simd_condition<T1, T2>;
}

#else  // XTENSOR_USE_XSIMD

namespace xt_simd
{
    template <class T, std::size_t A>
    class aligned_allocator;

    struct aligned_mode
    {
    };

    struct unaligned_mode
    {
    };

    template <class A>
    struct allocator_alignment
    {
        using type = unaligned_mode;
    };

    template <class A>
    using allocator_alignment_t = typename allocator_alignment<A>::type;

    template <class C>
    struct container_alignment
    {
        using type = unaligned_mode;
    };

    template <class C>
    using container_alignment_t = typename container_alignment<C>::type;

    template <class T>
    struct simd_traits
    {
        using type = T;
        using bool_type = bool;
        using batch_bool = bool;
        static constexpr std::size_t size = 1;
    };

    template <class T>
    struct revert_simd_traits
    {
        using type = T;
        static constexpr std::size_t size = simd_traits<type>::size;
    };

    template <class T>
    using simd_type = typename simd_traits<T>::type;

    template <class T>
    using simd_bool_type = typename simd_traits<T>::bool_type;

    template <class T>
    using revert_simd_type = typename revert_simd_traits<T>::type;

    template <class R, class T>
    inline simd_type<R> broadcast_as(const T& value)
    {
        return value;
    }

    template <class R, class T>
    inline simd_type<R> load_as(const T* src, aligned_mode)
    {
        return *src;
    }

    template <class R, class T>
    inline simd_type<R> load_as(const T* src, unaligned_mode)
    {
        return *src;
    }

    template <class R, class T>
    inline void store_as(R* dst, const simd_type<T>& src, aligned_mode)
    {
        *dst = src;
    }

    template <class R, class T>
    inline void store_as(R* dst, const simd_type<T>& src, unaligned_mode)
    {
        *dst = src;
    }

    template <class T>
    inline T select(bool cond, const T& t1, const T& t2)
    {
        return cond ? t1 : t2;
    }

    template <class T>
    inline std::size_t get_alignment_offset(const T* /*p*/, std::size_t size, std::size_t /*block_size*/)
    {
        return size;
    }

    template <class T1, class T2>
    using simd_return_type = simd_type<T2>;

    template <class V>
    struct is_batch_bool : std::false_type
    {
    };

    template <class V>
    struct is_batch_complex : std::false_type
    {
    };

    template <class T1, class T2>
    struct simd_condition : std::true_type
    {
    };
}

#endif  // XTENSOR_USE_XSIMD

namespace xt
{
    using xt_simd::aligned_mode;
    using xt_simd::unaligned_mode;

    struct inner_aligned_mode
    {
    };

    namespace detail
    {
        template <class A1, class A2>
        struct driven_align_mode_impl
        {
            using type = std::conditional_t<std::is_same<A1, A2>::value, A1, ::xt_simd::unaligned_mode>;
        };

        template <class A>
        struct driven_align_mode_impl<inner_aligned_mode, A>
        {
            using type = A;
        };
    }

    template <class A1, class A2>
    struct driven_align_mode
    {
        using type = typename detail::driven_align_mode_impl<A1, A2>::type;
    };

    template <class A1, class A2>
    using driven_align_mode_t = typename detail::driven_align_mode_impl<A1, A2>::type;

    namespace detail
    {
        template <class E, class T, class = void>
        struct has_load_simd : std::false_type
        {
        };

        template <class E, class T>
        struct has_load_simd<
            E,
            T,
            void_t<decltype(std::declval<E>().template load_simd<aligned_mode, T>(typename E::size_type(0)))>>
            : std::true_type
        {
        };

        template <class E, class T, bool B = xt_simd::simd_condition<typename E::value_type, T>::value>
        struct has_simd_interface_impl : has_load_simd<E, T>
        {
        };

        template <class E, class T>
        struct has_simd_interface_impl<E, T, false> : std::false_type
        {
        };
    }

    template <class E, class T = typename std::decay_t<E>::value_type>
    struct has_simd_interface : detail::has_simd_interface_impl<E, T>
    {
    };

    template <class E>
    concept has_simd_interface_concept = has_simd_interface<E>::value;

    template <class T>
    struct has_simd_type : std::integral_constant<bool, !std::is_same<T, xt_simd::simd_type<T>>::value>
    {
    };

    namespace detail
    {
        template <class F, class B, class = void>
        struct has_simd_apply_impl : std::false_type
        {
        };

        template <class F, class B>
        struct has_simd_apply_impl<F, B, void_t<decltype(&F::template simd_apply<B>)>> : std::true_type
        {
        };
    }

    template <class F, class B>
    struct has_simd_apply : detail::has_simd_apply_impl<F, B>
    {
    };

    template <class T>
    using bool_load_type = std::conditional_t<std::is_same<T, bool>::value, uint8_t, T>;

    template <class T>
    struct forbid_simd : std::false_type
    {
    };

    template <class A>
    struct forbid_simd<std::vector<bool, A>> : std::true_type
    {
    };

    template <class A>
    struct forbid_simd<const std::vector<bool, A>> : std::true_type
    {
    };

    template <class B, class A>
    struct forbid_simd<xtl::xdynamic_bitset<B, A>> : std::true_type
    {
    };

    template <class B, class A>
    struct forbid_simd<const xtl::xdynamic_bitset<B, A>> : std::true_type
    {
    };

    template <class C, class T1, class T2>
    struct container_simd_return_type
        : std::enable_if<!forbid_simd<C>::value, xt_simd::simd_return_type<T1, bool_load_type<T2>>>
    {
    };

    template <class C, class T1, class T2>
    using container_simd_return_type_t = typename container_simd_return_type<C, T1, T2>::type;
}

#endif
