/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_SEQUENCE_HPP
#define XTL_SEQUENCE_HPP

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtl_config.hpp"
#include "xmeta_utils.hpp"

namespace xtl
{
    template <class S>
    S make_sequence(typename S::size_type size);

    template <class S>
    S make_sequence(typename S::size_type size, typename S::value_type v);

    template <class S>
    S make_sequence(std::initializer_list<typename S::value_type> init);

    template <class R, class A>
    decltype(auto) forward_sequence(A&& s);

    /********************************
     * make_sequence implementation *
     ********************************/

    namespace detail
    {
        template <class S>
        struct sequence_builder
        {
            using value_type = typename S::value_type;
            using size_type = typename S::size_type;

            inline static S make(size_type size)
            {
                return S(size);
            }

            inline static S make(size_type size, value_type v)
            {
                return S(size, v);
            }

            inline static S make(std::initializer_list<value_type> init)
            {
                return S(init);
            }
        };

        template <class T, std::size_t N>
        struct sequence_builder<std::array<T, N>>
        {
            using sequence_type = std::array<T, N>;
            using value_type = typename sequence_type::value_type;
            using size_type = typename sequence_type::size_type;

            inline static sequence_type make(size_type /*size*/)
            {
                return sequence_type();
            }

            inline static sequence_type make(size_type /*size*/, value_type v)
            {
                sequence_type s;
                s.fill(v);
                return s;
            }

            inline static sequence_type make(std::initializer_list<value_type> init)
            {
                sequence_type s;
                std::copy(init.begin(), init.end(), s.begin());
                return s;
            }
        };
    }

    template <class S>
    inline S make_sequence(typename S::size_type size)
    {
        return detail::sequence_builder<S>::make(size);
    }

    template <class S>
    inline S make_sequence(typename S::size_type size, typename S::value_type v)
    {
        return detail::sequence_builder<S>::make(size, v);
    }

    template <class S>
    inline S make_sequence(std::initializer_list<typename S::value_type> init)
    {
        return detail::sequence_builder<S>::make(init);
    }

    /***********************************
     * forward_sequence implementation *
     ***********************************/

    namespace detail
    {
        template <class R, class A, class E = void>
        struct sequence_forwarder_impl
        {
            template <class T>
            static inline R forward(const T& r)
            {
                R ret;
                std::copy(std::begin(r), std::end(r), std::begin(ret));
                return ret;
            }
        };

        template <class R, class A>
        struct sequence_forwarder_impl<R, A, std::void_t<decltype(std::declval<R>().resize(
              std::declval<std::size_t>()))>>
        {
            template <class T>
            static inline auto forward(const T& r)
            {
                return R(std::begin(r), std::end(r));
            }
        };

        template <class R, class A>
        struct sequence_forwarder
            : sequence_forwarder_impl<R, A>
        {
        };

        template <class R>
        struct sequence_forwarder<R, R>
        {
            template <class T>
            static inline T&& forward(T&& t) noexcept
            {
                return std::forward<T>(t);
            }
        };

        template <class R, class A>
        using forwarder_type = detail::sequence_forwarder<
            std::decay_t<R>,
            std::remove_cv_t<std::remove_reference_t<A>>
        >;
    }

    template <class R, class A>
    inline decltype(auto) forward_sequence(typename std::remove_reference<A>::type& s)
    {
        using forwarder = detail::forwarder_type<R, A>;
        return forwarder::forward(std::forward<A>(s));
    }

    template <class R, class A>
    inline decltype(auto) forward_sequence(typename std::remove_reference<A>::type&& s)
    {
        using forwarder = detail::forwarder_type<R, A>;
        static_assert(!std::is_lvalue_reference<A>::value,
                      "Can not forward an rvalue as an lvalue.");
        return forwarder::forward(std::move(s));
    }

    /****************************
     * are_equivalent_sequences *
     ****************************/

    template <class E1, class E2>
    inline bool are_equivalent_sequences(const E1& e1, const E2& e2)
    {
        return std::equal(e1.cbegin(), e1.cend(), e2.cbegin(), e2.cend());
    }
}

#endif
