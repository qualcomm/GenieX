/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_XMETA_UTILS_HPP
#define XTL_XMETA_UTILS_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "xfunctional.hpp"
#include "xtl_config.hpp"

namespace xtl
{
    namespace mpl
    {
        /*************
         * mpl types *
         *************/

        template <class... T>
        struct vector
        {
        };

        template <bool B>
        using bool_ = std::integral_constant<bool, B>;

        template <std::size_t S>
        using size_t_ = std::integral_constant<std::size_t, S>;

        /*******
         * if_ *
         *******/

        template <bool B, class T, class F>
        struct if_c : std::conditional<B, T, F>
        {
        };

        template <bool B, class T, class F>
        using if_c_t = typename if_c<B, T, F>::type;

        template <class B, class T, class F>
        struct if_ : if_c<B::value, T, F>
        {
        };

        template <class B, class T, class F>
        using if_t = typename if_<B, T, F>::type;

        /***********
         * eval_if *
         ***********/

        template <bool B, class T, class F>
        struct eval_if_c
        {
            using type = typename T::type;
        };

        template <class T, class F>
        struct eval_if_c<false, T, F>
        {
            using type = typename F::type;
        };

        template <class B, class T, class F>
        struct eval_if : eval_if_c<B::value, T, F>
        {
        };

        template <class B, class T, class F>
        using eval_if_t = typename eval_if<B, T, F>::type;

        /********
         * cast *
         ********/

        namespace detail
        {
            template <class A, template <class...> class B>
            struct cast_impl;

            template <template <class...> class A, class... T, template <class...> class B>
            struct cast_impl<A<T...>, B>
            {
                using type = B<T...>;
            };
        }

        template <class A, template <class...> class B>
        struct cast : detail::cast_impl<A, B>
        {
        };

        template <class A, template <class...> class B>
        using cast_t = typename cast<A, B>::type;

        /********
         * size *
         ********/

        namespace detail
        {
            template <class L>
            struct size_impl;

            template <template <class...> class F, class... T>
            struct size_impl<F<T...>> : size_t_<sizeof...(T)>
            {
            };
        }

        template <class L>
        struct size : detail::size_impl<L>
        {
        };

        /*********
         * empty *
         *********/

        namespace detail
        {
            template <class L>
            struct empty_impl;

            template <template <class...> class F, class... T>
            struct empty_impl<F<T...>> : bool_<sizeof...(T) == std::size_t(0)>
            {
            };
        }

        template <class L>
        struct empty : detail::empty_impl<L>
        {
        };

        template <class L>
        using empty_t = typename empty<L>::type;

        /********
         * plus *
         ********/

        namespace detail
        {
            template <class... T>
            struct plus_impl;

            template <>
            struct plus_impl<> : size_t_<0>
            {
            };

            template <class T1, class... T>
            struct plus_impl<T1, T...> : size_t_<T1::value + plus_impl<T...>::value>
            {
            };
        }

        template <class... T>
        struct plus : detail::plus_impl<T...>
        {
        };

        /*********
         * count *
         *********/

        namespace detail
        {
            template <class L, class V>
            struct count_impl;

            template <template <class...> class L, class... T, class V>
            struct count_impl<L<T...>, V> : plus<std::is_same<T, V>...>
            {
            };
        }

        template <class L, class V>
        struct count : detail::count_impl<L, V>
        {
        };

        /************
         * count_if *
         ************/

        namespace detail
        {
            template <class L, template <class> class P>
            struct count_if_impl;

            template <template <class...> class L, class... T, template <class> class P>
            struct count_if_impl<L<T...>, P> : plus<P<T>...>
            {
            };
        }

        template <class L, template <class> class P>
        struct count_if : detail::count_if_impl<L, P>
        {
        };

        /************
         * index_of *
         ************/

        namespace detail
        {
            template <class L, class V>
            struct index_of_impl;

            template <template <class...> class L, class V>
            struct index_of_impl<L<>, V>
            {
                static constexpr size_t value = SIZE_MAX;
            };

            template <template <class...> class L, class... T, class V>
            struct index_of_impl<L<V, T...>, V>
            {
                static constexpr size_t value = 0u;
            };

            template <template <class...> class L, class U, class... T, class V>
            struct index_of_impl<L<U, T...>, V>
            {
                static constexpr size_t tmp = index_of_impl<L<T...>, V>::value;
                static constexpr size_t value = tmp == SIZE_MAX ? SIZE_MAX : 1u + tmp;
            };
        }

        template <class L, class T>
        struct index_of : detail::index_of_impl<L, T>
        {
        };

        /************
         * contains *
         ************/

        namespace detail
        {
            template <class L, class V>
            struct contains_impl;

            template <template <class...> class L, class V>
            struct contains_impl<L<>, V> : std::false_type
            {
            };

            template <template <class...> class L, class... T, class V>
            struct contains_impl<L<V, T...>, V> : std::true_type
            {
            };

            template <template <class...> class L, class U, class... T, class V>
            struct contains_impl<L<U, T...>, V> : contains_impl<L<T...>, V>
            {
            };
        }

        template <class L, class V>
        struct contains : detail::contains_impl<L, V>
        {
        };

        /*********
         * front *
         *********/

        namespace detail
        {
            template <class L>
            struct front_impl;

            template <template <class...> class L, class T, class... U>
            struct front_impl<L<T, U...>>
            {
                using type = T;
            };
        }

        template <class L>
        struct front : detail::front_impl<L>
        {
        };

        template <class L>
        using front_t = typename front<L>::type;

        /********
         * back *
         ********/

        namespace detail
        {
            template <class L>
            struct back_impl;

            template <template <class...> class L, class T>
            struct back_impl<L<T>>
            {
                using type = T;
            };

            // Compilation time improvement
            template <template <class...> class L, class T1, class T2>
            struct back_impl<L<T1, T2>>
            {
                using type = T2;
            };

            template <template <class...> class L, class T1, class T2, class T3>
            struct back_impl<L<T1, T2, T3>>
            {
                using type = T3;
            };

            template <template <class...> class L, class T1, class T2, class T3, class T4>
            struct back_impl<L<T1, T2, T3, T4>>
            {
                using type = T4;
            };

            template <template <class...> class L, class T, class... U>
            struct back_impl<L<T, U...>> : back_impl<L<U...>>
            {
            };
        }

        template <class L>
        struct back : detail::back_impl<L>
        {
        };

        template <class L>
        using back_t = typename back<L>::type;

        /**************
         * push_front *
         **************/

        namespace detail
        {
            template <class L, class... T>
            struct push_front_impl;

            template <template <class...> class L, class... U, class... T>
            struct push_front_impl<L<U...>, T...>
            {
                using type = L<T..., U...>;
            };
        }

        template <class L, class... T>
        struct push_front : detail::push_front_impl<L, T...>
        {
        };

        template <class L, class... T>
        using push_front_t = typename push_front<L, T...>::type;

        /*************
         * push_back *
         *************/

        namespace detail
        {
            template <class L, class... T>
            struct push_back_impl;

            template <template <class...> class L, class... U, class... T>
            struct push_back_impl<L<U...>, T...>
            {
                using type = L<U..., T...>;
            };
        }

        template <class L, class... T>
        struct push_back : detail::push_back_impl<L, T...>
        {
        };

        template <class L, class... T>
        using push_back_t = typename push_back<L, T...>::type;

        /*************
         * pop_front *
         *************/

        namespace detail
        {
            template <class L>
            struct pop_front_impl;

            template <template <class...> class L, class T, class... U>
            struct pop_front_impl<L<T, U...>>
            {
                using type = L<U...>;
            };
        }

        template <class L>
        struct pop_front : detail::pop_front_impl<L>
        {
        };

        template <class L>
        using pop_front_t = typename pop_front<L>::type;

        /*************
         * transform *
         *************/

        namespace detail
        {
            template <template <class...> class F, class L>
            struct transform_impl;

            template <template <class...> class F, template <class...> class L, class... T>
            struct transform_impl<F, L<T...>>
            {
                using type = L<F<T>...>;
            };
        }

        template <template <class...> class F, class L>
        struct transform : detail::transform_impl<F, L>
        {
        };

        template <template <class...> class F, class L>
        using transform_t = typename transform<F, L>::type;

        /*************
         * merge_set *
         *************/

        namespace detail
        {
            template <class S1, class S2>
            struct merge_set_impl;

            template <template <class...> class L, class... T>
            struct merge_set_impl<L<T...>, L<>>
            {
                using type = L<T...>;
            };

            template <template <class...> class L, class... T, class U1, class... U>
            struct merge_set_impl<L<T...>, L<U1, U...>>
            {
                using type = typename merge_set_impl<if_t<contains<L<T...>, U1>,
                                                          L<T...>,
                                                          L<T..., U1>>,
                                                     L<U...>>::type;
            };
        }

        template <class S1, class S2>
        struct merge_set : detail::merge_set_impl<S1, S2>
        {
        };

        template <class S1, class S2>
        using merge_set_t = typename merge_set<S1, S2>::type;

        /***********
         * find_if *
         ***********/

        template <template <class> class Test, class L>
        struct find_if;

        namespace detail
        {
            template <template <class> class Test, std::size_t I, class... T>
            struct find_if_impl;

            template <template <class> class Test, std::size_t I>
            struct find_if_impl<Test, I> : size_t_<I>
            {
            };

            template <template <class> class Test, std::size_t I, class T0, class... T>
            struct find_if_impl<Test, I, T0, T...> : std::conditional_t<Test<T0>::value,
                                                                        size_t_<I>,
                                                                        find_if_impl<Test, I + 1, T...>>
            {
            };
        }

        template <template <class> class Test, template <class...> class L, class... T>
        struct find_if<Test, L<T...>> : detail::find_if_impl<Test, 0, T...>
        {
        };

        /*********
         * split *
         *********/

        namespace detail
        {
            template <std::size_t N, class L1, class L2>
            struct transfer
            {
                using new_l1 = push_back_t<L1, front_t<L2>>;
                using new_l2 = pop_front_t<L2>;
                using new_transfer = transfer<N - 1, new_l1, new_l2>;
                using first_type = typename new_transfer::first_type;
                using second_type = typename new_transfer::second_type;
            };

            template <class L1, class L2>
            struct transfer<0, L1, L2>
            {
                using first_type = L1;
                using second_type = L2;
            };

            template <std::size_t N, class L>
            struct split_impl
            {
                using tr_type = transfer<N, vector<>, L>;
                using first_type = typename tr_type::first_type;
                using second_type = typename tr_type::second_type;
            };
        }

        template <std::size_t N, class L>
        struct split : detail::split_impl<N, L>
        {
        };

        /**********
         * unique *
         **********/

        namespace detail
        {
            template <class L>
            struct unique_impl;

            template <template <class...> class L, class... T>
            struct unique_impl<L<T...>>
            {
                using type = merge_set_t<L<>, L<T...>>;
            };
        }

        template <class L>
        struct unique : detail::unique_impl<L>
        {
        };

        template <class L>
        using unique_t = typename unique<L>::type;

        /***********
         * switch_ *
         ***********/

        using default_t = std::true_type;

        namespace detail
        {
            template <class... T>
            struct switch_impl;

            template <class C, class T, class... U>
            struct switch_impl<C, T, U...>
                : std::conditional<C::value, T, typename switch_impl<U...>::type>
            {
            };

            template <class C, class T, class U>
            struct switch_impl<C, T, default_t, U>
                : std::conditional<C::value, T, U>
            {
            };
        }

        template <class... T>
        struct switch_ : detail::switch_impl<T...>
        {
        };

        template <class... T>
        using switch_t = typename switch_<T...>::type;
    }
}

#endif
