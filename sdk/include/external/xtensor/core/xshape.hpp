/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_XSHAPE_HPP
#define XTENSOR_XSHAPE_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>

#include "../containers/xstorage.hpp"
#include "../core/xlayout.hpp"
#include "../core/xtensor_forward.hpp"

namespace xt
{
    template <class T>
    using dynamic_shape = svector<T, 4>;

    template <class T, std::size_t N>
    using static_shape = std::array<T, N>;

    template <std::size_t... X>
    class fixed_shape;

    using xindex = dynamic_shape<std::size_t>;

    template <class S1, class S2>
    bool same_shape(const S1& s1, const S2& s2) noexcept;

    template <class U>
    struct initializer_dimension;

    template <class R, class T>
    constexpr R shape(T t);

    template <class R = std::size_t, class T, std::size_t N>
    xt::static_shape<R, N> shape(const T (&aList)[N]);

    template <class S>
    struct static_dimension;

    template <layout_type L, class S>
    struct select_layout;

    template <class... S>
    struct promote_shape;

    template <class... S>
    struct promote_strides;

    template <class S>
    struct index_from_shape;
}

namespace xtl
{
    namespace detail
    {
        template <class S>
        struct sequence_builder;

        template <std::size_t... I>
        struct sequence_builder<xt::fixed_shape<I...>>
        {
            using sequence_type = xt::fixed_shape<I...>;
            using value_type = typename sequence_type::value_type;

            inline static sequence_type make(std::size_t /*size*/)
            {
                return sequence_type{};
            }

            inline static sequence_type make(std::size_t /*size*/, value_type /*v*/)
            {
                return sequence_type{};
            }
        };
    }
}

namespace xt
{
    /**
     * @defgroup xt_xshape Support functions to get/check a shape array.
     */

    /**************
     * same_shape *
     **************/

    /**
     * Check if two objects have the same shape.
     *
     * @ingroup xt_xshape
     * @param s1 an array
     * @param s2 an array
     * @return bool
     */
    template <class S1, class S2>
    inline bool same_shape(const S1& s1, const S2& s2) noexcept
    {
        return s1.size() == s2.size() && std::equal(s1.begin(), s1.end(), s2.begin());
    }

    /*************
     * has_shape *
     *************/

    /**
     * Check if an object has a certain shape.
     *
     * @ingroup xt_xshape
     * @param a an array
     * @param shape the shape to test
     * @return bool
     */
    template <class E, class S>
    inline bool has_shape(const E& e, std::initializer_list<S> shape) noexcept
    {
        return e.shape().size() == shape.size()
               && std::equal(e.shape().cbegin(), e.shape().cend(), shape.begin());
    }

    /**
     * Check if an object has a certain shape.
     *
     * @ingroup has_shape
     * @param a an array
     * @param shape the shape to test
     * @return bool
     */
    template <class E, class S, class = typename std::enable_if_t<has_iterator_interface<S>::value>>
    inline bool has_shape(const E& e, const S& shape)
    {
        return e.shape().size() == shape.size()
               && std::equal(e.shape().cbegin(), e.shape().cend(), shape.begin());
    }

    /*************************
     * initializer_dimension *
     *************************/

    namespace detail
    {
        template <class U>
        struct initializer_depth_impl
        {
            static constexpr std::size_t value = 0;
        };

        template <class T>
        struct initializer_depth_impl<std::initializer_list<T>>
        {
            static constexpr std::size_t value = 1 + initializer_depth_impl<T>::value;
        };
    }

    template <class U>
    struct initializer_dimension
    {
        static constexpr std::size_t value = detail::initializer_depth_impl<U>::value;
    };

    /*********************
     * initializer_shape *
     *********************/

    namespace detail
    {
        template <std::size_t I>
        struct initializer_shape_impl
        {
            template <class T>
            static constexpr std::size_t value(T t)
            {
                return t.size() == 0 ? 0 : initializer_shape_impl<I - 1>::value(*t.begin());
            }
        };

        template <>
        struct initializer_shape_impl<0>
        {
            template <class T>
            static constexpr std::size_t value(T t)
            {
                return t.size();
            }
        };

        template <class R, class U, std::size_t... I>
        constexpr R initializer_shape(U t, std::index_sequence<I...>)
        {
            using size_type = typename R::value_type;
            return {size_type(initializer_shape_impl<I>::value(t))...};
        }
    }

    template <class R, class T>
    constexpr R shape(T t)
    {
        return detail::initializer_shape<R, decltype(t)>(
            t,
            std::make_index_sequence<initializer_dimension<decltype(t)>::value>()
        );
    }

    /** @brief Generate an xt::static_shape of the given size. */
    template <class R, class T, std::size_t N>
    xt::static_shape<R, N> shape(const T (&list)[N])
    {
        xt::static_shape<R, N> shape;
        std::copy(std::begin(list), std::end(list), std::begin(shape));
        return shape;
    }

    /********************
     * static_dimension *
     ********************/

    namespace detail
    {
        template <class T, class E = void>
        struct static_dimension_impl
        {
            static constexpr std::ptrdiff_t value = -1;
        };

        template <class T>
        struct static_dimension_impl<T, void_t<decltype(std::tuple_size<T>::value)>>
        {
            static constexpr std::ptrdiff_t value = static_cast<std::ptrdiff_t>(std::tuple_size<T>::value);
        };
    }

    template <class S>
    struct static_dimension
    {
        static constexpr std::ptrdiff_t value = detail::static_dimension_impl<S>::value;
    };

    /**
     * Compute a layout based on a layout and a shape type.
     *
     * The main functionality of this function is that it reduces vectors to
     * ``xt::layout_type::any`` so that assigning a row major 1D container to another
     * row_major container becomes free.
     *
     * @ingroup xt_xshape
     */
    template <layout_type L, class S>
    struct select_layout
    {
        static constexpr std::ptrdiff_t static_dimension = xt::static_dimension<S>::value;
        static constexpr bool is_any = static_dimension != -1 && static_dimension <= 1
                                       && L != layout_type::dynamic;
        static constexpr layout_type value = is_any ? layout_type::any : L;
    };

    /*************************************
     * promote_shape and promote_strides *
     *************************************/

    namespace detail
    {
        template <class T1, class T2>
        constexpr std::common_type_t<T1, T2> imax(const T1& a, const T2& b)
        {
            return a > b ? a : b;
        }

        // Variadic meta-function returning the maximal size of std::arrays.
        template <class... T>
        struct max_array_size;

        template <>
        struct max_array_size<>
        {
            static constexpr std::size_t value = 0;
        };

        template <class T, class... Ts>
        struct max_array_size<T, Ts...>
            : std::integral_constant<std::size_t, imax(std::tuple_size<T>::value, max_array_size<Ts...>::value)>
        {
        };

        // Broadcasting for fixed shapes
        template <std::size_t IDX, std::size_t... X>
        struct at
        {
            static constexpr std::size_t arr[sizeof...(X)] = {X...};
            static constexpr std::size_t value = (IDX < sizeof...(X)) ? arr[IDX] : 0;
        };

        template <class S1, class S2>
        struct broadcast_fixed_shape;

        template <class IX, class A, class B>
        struct broadcast_fixed_shape_impl;

        template <std::size_t IX, class A, class B>
        struct broadcast_fixed_shape_cmp_impl;

        template <std::size_t JX, std::size_t... I, std::size_t... J>
        struct broadcast_fixed_shape_cmp_impl<JX, fixed_shape<I...>, fixed_shape<J...>>
        {
            // We line the shapes up from the last index
            // IX may underflow, thus being a very large number
            static constexpr std::size_t IX = JX - (sizeof...(J) - sizeof...(I));

            // Out of bounds access gives value 0
            static constexpr std::size_t I_v = at<IX, I...>::value;
            static constexpr std::size_t J_v = at<JX, J...>::value;

            // we're statically checking if the broadcast shapes are either one on either of them or equal
            static_assert(!I_v || I_v == 1 || J_v == 1 || J_v == I_v, "broadcast shapes do not match.");

            static constexpr std::size_t ordinate = (I_v > J_v) ? I_v : J_v;
            static constexpr bool value = (I_v == J_v);
        };

        template <std::size_t... JX, std::size_t... I, std::size_t... J>
        struct broadcast_fixed_shape_impl<std::index_sequence<JX...>, fixed_shape<I...>, fixed_shape<J...>>
        {
            static_assert(sizeof...(J) >= sizeof...(I), "broadcast shapes do not match.");

            using type = xt::fixed_shape<
                broadcast_fixed_shape_cmp_impl<JX, fixed_shape<I...>, fixed_shape<J...>>::ordinate...>;
            static constexpr bool value = std::conjunction<
                broadcast_fixed_shape_cmp_impl<JX, fixed_shape<I...>, fixed_shape<J...>>...>::value;
        };

        /* broadcast_fixed_shape<fixed_shape<I...>, fixed_shape<J...>>
         * Just like a call to broadcast_shape(cont S1& input, S2& output),
         * except that the result shape is alised as type, and the returned
         * bool is the member value. Asserts on an illegal broadcast, including
         * the case where pack I is strictly longer than pack J. */

        template <std::size_t... I, std::size_t... J>
        struct broadcast_fixed_shape<fixed_shape<I...>, fixed_shape<J...>>
            : broadcast_fixed_shape_impl<std::make_index_sequence<sizeof...(J)>, fixed_shape<I...>, fixed_shape<J...>>
        {
        };

        // Simple is_array and only_array meta-functions
        template <class S>
        struct is_array
        {
            static constexpr bool value = false;
        };

        template <class T, std::size_t N>
        struct is_array<std::array<T, N>>
        {
            static constexpr bool value = true;
        };

        template <class S>
        struct is_fixed : std::false_type
        {
        };

        template <std::size_t... N>
        struct is_fixed<fixed_shape<N...>> : std::true_type
        {
        };

        template <class S>
        struct is_scalar_shape
        {
            static constexpr bool value = false;
        };

        template <class T>
        struct is_scalar_shape<std::array<T, 0>>
        {
            static constexpr bool value = true;
        };

        template <class... S>
        using only_array = std::conjunction<std::disjunction<is_array<S>, is_fixed<S>>...>;

        // test that at least one argument is a fixed shape. If yes, then either argument has to be fixed or
        // scalar
        template <class... S>
        using only_fixed = std::integral_constant<
            bool,
            std::disjunction<is_fixed<S>...>::value
                && std::conjunction<std::disjunction<is_fixed<S>, is_scalar_shape<S>>...>::value>;

        template <class... S>
        using all_fixed = std::conjunction<is_fixed<S>...>;

        // The promote_index meta-function returns std::vector<promoted_value_type> in the
        // general case and an array of the promoted value type and maximal size if all
        // arguments are of type std::array

        template <class... S>
        struct promote_array
        {
            using type = std::
                array<typename std::common_type<typename S::value_type...>::type, max_array_size<S...>::value>;
        };

        template <>
        struct promote_array<>
        {
            using type = std::array<std::size_t, 0>;
        };

        template <class S>
        struct filter_scalar
        {
            using type = S;
        };

        template <class T>
        struct filter_scalar<std::array<T, 0>>
        {
            using type = fixed_shape<1>;
        };

        template <class S>
        using filter_scalar_t = typename filter_scalar<S>::type;

        template <class... S>
        struct promote_fixed : promote_fixed<filter_scalar_t<S>...>
        {
        };

        template <std::size_t... I>
        struct promote_fixed<fixed_shape<I...>>
        {
            using type = fixed_shape<I...>;
            static constexpr bool value = true;
        };

        template <std::size_t... I, std::size_t... J, class... S>
        struct promote_fixed<fixed_shape<I...>, fixed_shape<J...>, S...>
        {
        private:

            using intermediate = std::conditional_t<
                (sizeof...(I) > sizeof...(J)),
                broadcast_fixed_shape<fixed_shape<J...>, fixed_shape<I...>>,
                broadcast_fixed_shape<fixed_shape<I...>, fixed_shape<J...>>>;
            using result = promote_fixed<typename intermediate::type, S...>;

        public:

            using type = typename result::type;
            static constexpr bool value = std::conjunction<intermediate, result>::value;
        };

        template <bool all_index, bool all_array, class... S>
        struct select_promote_index;

        template <class... S>
        struct select_promote_index<true, true, S...> : promote_fixed<S...>
        {
        };

        template <>
        struct select_promote_index<true, true>
        {
            // todo correct? used in xvectorize
            using type = dynamic_shape<std::size_t>;
        };

        template <class... S>
        struct select_promote_index<false, true, S...> : promote_array<S...>
        {
        };

        template <class... S>
        struct select_promote_index<false, false, S...>
        {
            using type = dynamic_shape<typename std::common_type<typename S::value_type...>::type>;
        };

        template <class... S>
        struct promote_index : select_promote_index<only_fixed<S...>::value, only_array<S...>::value, S...>
        {
        };

        template <class T>
        struct index_from_shape_impl
        {
            using type = T;
        };

        template <std::size_t... N>
        struct index_from_shape_impl<fixed_shape<N...>>
        {
            using type = std::array<std::size_t, sizeof...(N)>;
        };
    }

    template <typename T>
    concept fixed_shape_container_concept = detail::is_fixed<typename std::decay_t<T>::shape_type>::value;

    template <class... S>
    struct promote_shape
    {
        using type = typename detail::promote_index<S...>::type;
    };

    /**
     * @ingroup xt_xshape
     */
    template <class... S>
    using promote_shape_t = typename promote_shape<S...>::type;

    template <class... S>
    struct promote_strides
    {
        using type = typename detail::promote_index<S...>::type;
    };

    /**
     * @ingroup xt_xshape
     */
    template <class... S>
    using promote_strides_t = typename promote_strides<S...>::type;

    template <class S>
    struct index_from_shape
    {
        using type = typename detail::index_from_shape_impl<S>::type;
    };

    /**
     * @ingroup xt_xshape
     */
    template <class S>
    using index_from_shape_t = typename index_from_shape<S>::type;

    /**********************
     * filter_fixed_shape *
     **********************/

    namespace detail
    {
        template <class S>
        struct filter_fixed_shape_impl
        {
            using type = S;
        };

        template <std::size_t... N>
        struct filter_fixed_shape_impl<fixed_shape<N...>>
        {
            using type = std::array<std::size_t, sizeof...(N)>;
        };
    }

    template <class S>
    struct filter_fixed_shape : detail::filter_fixed_shape_impl<S>
    {
    };

    /**
     * @ingroup xt_xshape
     */
    template <class S>
    using filter_fixed_shape_t = typename filter_fixed_shape<S>::type;
}

#endif
