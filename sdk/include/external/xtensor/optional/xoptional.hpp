/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_OPTIONAL_HPP
#define XTENSOR_OPTIONAL_HPP

#include <type_traits>
#include <utility>

#include <xtl/xoptional.hpp>
#include <xtl/xoptional_sequence.hpp>

#include "../containers/xarray.hpp"
#include "../containers/xscalar.hpp"
#include "../containers/xtensor.hpp"
#include "../generators/xgenerator.hpp"
#include "../reducers/xreducer.hpp"
#include "../views/xbroadcast.hpp"
#include "../views/xdynamic_view.hpp"
#include "../views/xfunctor_view.hpp"
#include "../views/xindex_view.hpp"
#include "../views/xrepeat.hpp"
#include "../views/xstrided_view.hpp"
#include "../views/xview.hpp"

namespace xt
{

    /****************************************************
     * Metafunction for splitting xoptional expressions *
     ****************************************************/

    namespace extension
    {

        /**************************************
         * get_expression_tag specializations *
         **************************************/

        template <class T, class B>
        struct get_expression_tag<xtl::xoptional<T, B>>
        {
            using type = xoptional_expression_tag;
        };

        /************************
         * xoptional_empty_base *
         ************************/

        template <class D>
        class xoptional_empty_base
        {
        public:

            using expression_tag = xoptional_expression_tag;

        protected:

            D& derived_cast() noexcept;
            const D& derived_cast() const noexcept;
        };
    }

    namespace detail
    {
        /*****************************
         * split_optional_expression *
         *****************************/

        template <class T, class Tag>
        struct split_optional_expression_impl
        {
            using value_expression = T;
            using flag_expression = decltype(ones<bool>(std::declval<T>().shape()));

            template <class U>
            inline static U&& value(U&& arg)
            {
                return std::forward<U>(arg);
            }

            template <class U>
            inline static flag_expression has_value(U&& arg)
            {
                return ones<bool>(arg.shape());
            }
        };

        template <class T, class Tag>
        struct split_optional_expression_impl<xscalar<T>, Tag>
        {
            using value_expression = xscalar<T>;
            using flag_expression = xscalar<bool>;

            template <class U>
            inline static U&& value(U&& arg)
            {
                return std::forward<U>(arg);
            }

            template <class U>
            inline static flag_expression has_value(U&&)
            {
                return xscalar<bool>(true);
            }
        };

        template <class T>
        struct split_optional_expression_impl_base
        {
            static constexpr bool is_const = std::is_const<std::remove_reference_t<T>>::value;
            using decay_type = std::decay_t<T>;

            using value_expression = std::conditional_t<
                is_const,
                typename decay_type::const_value_expression,
                typename decay_type::value_expression>;
            using flag_expression = std::
                conditional_t<is_const, typename decay_type::const_flag_expression, typename decay_type::flag_expression>;

            template <class U>
            inline static value_expression value(U&& arg)
            {
                return arg.value();
            }

            template <class U>
            inline static flag_expression has_value(U&& arg)
            {
                return arg.has_value();
            }
        };

        template <class T>
        struct split_optional_expression_impl<T, xoptional_expression_tag>
            : split_optional_expression_impl_base<T>
        {
        };

        template <class T>
        struct split_optional_expression_impl<xscalar<T>, xoptional_expression_tag>
            : split_optional_expression_impl_base<xscalar<T>>
        {
        };

        template <class T>
        struct split_optional_expression
            : split_optional_expression_impl<T, xexpression_tag_t<std::decay_t<T>>>
        {
        };

        template <class T>
        using value_expression_t = typename split_optional_expression<T>::value_expression;

        template <class T>
        using flag_expression_t = typename split_optional_expression<T>::flag_expression;

        /********************
         * optional_bitwise *
         ********************/

        template <class T = bool>
        class optional_bitwise
        {
        public:

            using return_type = T;
            using first_argument_type = T;
            using second_argument_type = T;
            using result_type = T;
            using simd_value_type = bool;
            using simd_result_type = bool;

            template <class... Args>
            constexpr result_type operator()(const Args&... args) const
            {
                return apply_impl(args...);
            }

            template <class B, class... Args>
            constexpr B simd_apply(const B& b, const Args&... args) const
            {
                return simd_apply_impl(b, args...);
            }

        private:

            constexpr result_type apply_impl() const
            {
                return true;
            }

            template <class U, class... Args>
            constexpr result_type apply_impl(const U& t, const Args&... args) const
            {
                return t & apply_impl(args...);
            }

            template <class B>
            constexpr B simd_apply_impl(const B& b) const
            {
                return b;
            }

            template <class B1, class B2, class... Args>
            constexpr B1 simd_apply_impl(const B1& b1, const B2& b2, const Args&... args) const
            {
                return b1 & simd_apply_impl(b2, args...);
            }
        };

        /*********************************
         * optional const_value rebinder *
         *********************************/

        template <class T, class B>
        struct const_value_rebinder<xtl::xoptional<T, B>, T>
        {
            static const_value<T> run(const const_value<xtl::xoptional<T, B>>& src)
            {
                return const_value<T>(src.m_value.value());
            }
        };

        /**************************
         * xreducer types helpers *
         **************************/

        template <class T, class B>
        struct xreducer_size_type<xtl::xoptional<T, B>>
        {
            using type = xtl::xoptional<std::size_t, bool>;
        };

        template <class T, class B>
        struct xreducer_temporary_type<xtl::xoptional<T, B>>
        {
            using type = xtl::xoptional<std::decay_t<T>, bool>;
            ;
        };
    }

    /**********************
     * optional functions *
     **********************/

    template <class T, class B>
    auto sign(const xtl::xoptional<T, B>& e);

    /*template <class E, XTL_REQUIRES(is_xexpression<E>)>
    detail::value_expression_t<E> value(E&&);

    template <class E, XTL_REQUIRES(is_xexpression<E>)>
    detail::flag_expression_t<E> has_value(E&&);*/

    template <>
    class xexpression_assigner_base<xoptional_expression_tag>
    {
    public:

        template <class E1, class E2>
        static void assign_data(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial);
    };

    /**********************************
     * xscalar extension for optional *
     **********************************/

    namespace extension
    {
        template <class CT>
        struct xscalar_optional_traits
        {
            using closure_type = CT;
            static constexpr bool is_ref = std::is_reference<closure_type>::value;
            using unref_closure_type = std::remove_reference_t<closure_type>;
            static constexpr bool is_const = std::is_const<unref_closure_type>::value;
            using raw_closure_type = std::decay_t<CT>;

            using raw_value_closure = typename raw_closure_type::value_closure;
            using raw_flag_closure = typename raw_closure_type::flag_closure;
            using const_raw_value_closure = std::add_const_t<raw_value_closure>;
            using const_raw_flag_closure = std::add_const_t<raw_flag_closure>;

            using value_closure = std::conditional_t<is_ref, std::add_lvalue_reference_t<raw_value_closure>, raw_value_closure>;
            using flag_closure = std::conditional_t<is_ref, std::add_lvalue_reference_t<raw_flag_closure>, raw_flag_closure>;
            using const_value_closure = std::
                conditional_t<is_ref, std::add_lvalue_reference_t<const_raw_value_closure>, raw_value_closure>;
            using const_flag_closure = std::
                conditional_t<is_ref, std::add_lvalue_reference_t<const_raw_flag_closure>, raw_flag_closure>;

            using value_expression = xscalar<std::conditional_t<is_const, const_value_closure, value_closure>>;
            using flag_expression = xscalar<std::conditional_t<is_const, const_flag_closure, flag_closure>>;
            using const_value_expression = xscalar<const_value_closure>;
            using const_flag_expression = xscalar<const_flag_closure>;
        };

        template <class CT>
        class xscalar_optional_base : public xoptional_empty_base<xscalar<CT>>
        {
        public:

            using traits = xscalar_optional_traits<CT>;
            using value_expression = typename traits::value_expression;
            using flag_expression = typename traits::flag_expression;
            using const_value_expression = typename traits::const_value_expression;
            using const_flag_expression = typename traits::const_flag_expression;
            using expression_tag = xoptional_expression_tag;

            value_expression value();
            const_value_expression value() const;

            flag_expression has_value();
            const_flag_expression has_value() const;
        };

        template <class CT>
        struct xscalar_base_impl<xoptional_expression_tag, CT>
        {
            using type = xscalar_optional_base<CT>;
        };
    }

    /*************************************
     * xcontainer extension for optional *
     *************************************/

    namespace extension
    {
        template <class T>
        class xcontainer_optional_base : public xoptional_empty_base<typename T::derived_type>
        {
        public:

            using traits = T;
            using value_expression = typename traits::value_expression;
            using flag_expression = typename traits::flag_expression;
            using const_value_expression = typename traits::const_value_expression;
            using const_flag_expression = typename traits::const_flag_expression;
            using expression_tag = xoptional_expression_tag;

            value_expression value();
            const_value_expression value() const;

            flag_expression has_value();
            const_flag_expression has_value() const;
        };
    }

    /*******************************************
     * xarray_container extension for optional *
     *******************************************/

    namespace extension
    {
        template <class EC, layout_type L, class SC>
        struct xarray_optional_traits
        {
            using value_container = typename std::remove_reference_t<EC>::base_container_type;
            using flag_container = typename std::remove_reference_t<EC>::flag_container_type;
            using value_expression = xarray_adaptor<value_container&, L, SC>;
            using flag_expression = xarray_adaptor<flag_container&, L, SC>;
            using const_value_expression = xarray_adaptor<const value_container&, L, SC>;
            using const_flag_expression = xarray_adaptor<const flag_container&, L, SC>;
        };

        template <class EC, layout_type L, class SC>
        struct xarray_container_optional_traits : xarray_optional_traits<EC, L, SC>
        {
            using derived_type = xarray_container<EC, L, SC, xoptional_expression_tag>;
        };

        template <class EC, layout_type L, class SC>
        struct xarray_container_base<EC, L, SC, xoptional_expression_tag>
        {
            using traits = xarray_container_optional_traits<EC, L, SC>;
            using type = xcontainer_optional_base<traits>;
        };
    }

    /*****************************************
     * xarray_adaptor extension for optional *
     *****************************************/

    namespace extension
    {
        template <class EC, layout_type L, class SC>
        struct xarray_adaptor_optional_traits : xarray_optional_traits<EC, L, SC>
        {
            using derived_type = xarray_adaptor<EC, L, SC, xoptional_expression_tag>;
        };

        template <class EC, layout_type L, class SC>
        struct xarray_adaptor_base<EC, L, SC, xoptional_expression_tag>
        {
            using traits = xarray_adaptor_optional_traits<EC, L, SC>;
            using type = xcontainer_optional_base<traits>;
        };
    }

    /********************************************
     * xtensor_container extension for optional *
     ********************************************/

    namespace extension
    {
        template <class EC, std::size_t N, layout_type L>
        struct xtensor_optional_traits
        {
            using value_container = typename std::remove_reference_t<EC>::base_container_type;
            using flag_container = typename std::remove_reference_t<EC>::flag_container_type;
            using value_expression = xtensor_adaptor<value_container&, N, L>;
            using flag_expression = xtensor_adaptor<flag_container&, N, L>;
            using const_value_expression = xtensor_adaptor<const value_container&, N, L>;
            using const_flag_expression = xtensor_adaptor<const flag_container&, N, L>;
        };

        template <class EC, std::size_t N, layout_type L>
        struct xtensor_container_optional_traits : xtensor_optional_traits<EC, N, L>
        {
            using derived_type = xtensor_container<EC, N, L, xoptional_expression_tag>;
        };

        template <class EC, std::size_t N, layout_type L>
        struct xtensor_container_base<EC, N, L, xoptional_expression_tag>
        {
            using traits = xtensor_container_optional_traits<EC, N, L>;
            using type = xcontainer_optional_base<traits>;
        };
    }

    /******************************************
     * xtensor_adaptor extension for optional *
     ******************************************/

    namespace extension
    {
        template <class EC, std::size_t N, layout_type L>
        struct xtensor_adaptor_optional_traits : xtensor_optional_traits<EC, N, L>
        {
            using derived_type = xtensor_adaptor<EC, N, L, xoptional_expression_tag>;
        };

        template <class EC, std::size_t N, layout_type L>
        struct xtensor_adaptor_base<EC, N, L, xoptional_expression_tag>
        {
            using traits = xtensor_adaptor_optional_traits<EC, N, L>;
            using type = xcontainer_optional_base<traits>;
        };
    }

    /***************************************
     * xtensor_view extension for optional *
     ***************************************/

    namespace extension
    {
        template <class EC, std::size_t N, layout_type L>
        struct xtensor_view_optional_traits : xtensor_optional_traits<EC, N, L>
        {
            using derived_type = xtensor_view<EC, N, L, xoptional_expression_tag>;
        };

        template <class EC, std::size_t N, layout_type L>
        struct xtensor_view_base<EC, N, L, xoptional_expression_tag>
        {
            using traits = xtensor_view_optional_traits<EC, N, L>;
            using type = xcontainer_optional_base<traits>;
        };
    }

    /************************************************
     * xfunction extension for optional expressions *
     ************************************************/

    namespace extension
    {
        template <class F, class... CT>
        class xfunction_optional_base : public xoptional_empty_base<xfunction<F, CT...>>
        {
        public:

            using expression_tag = xoptional_expression_tag;
            using value_functor = F;
            using flag_functor = xt::detail::optional_bitwise<bool>;

            using value_expression = xfunction<value_functor, xt::detail::value_expression_t<CT>...>;
            using flag_expression = xfunction<flag_functor, xt::detail::flag_expression_t<CT>...>;
            using const_value_expression = value_expression;
            using const_flag_expression = flag_expression;

            const_value_expression value() const;
            const_flag_expression has_value() const;

        private:

            template <std::size_t... I>
            const_value_expression value_impl(std::index_sequence<I...>) const;

            template <std::size_t... I>
            const_flag_expression has_value_impl(std::index_sequence<I...>) const;
        };

        template <class F, class... CT>
        struct xfunction_base_impl<xoptional_expression_tag, F, CT...>
        {
            using type = xfunction_optional_base<F, CT...>;
        };
    }

    /****************************************************
     * xdynamic_view extension for optional expressions *
     ****************************************************/

    namespace extension
    {
        template <class CT, class S, layout_type L, class FST>
        class xdynamic_view_optional : public xoptional_empty_base<xdynamic_view<CT, S, L, FST>>
        {
        public:

            using expression_tag = xoptional_expression_tag;
            using uvt = typename std::decay_t<CT>::value_expression;
            using uft = typename std::decay_t<CT>::flag_expression;
            using ucvt = typename std::decay_t<CT>::const_value_expression;
            using ucft = typename std::decay_t<CT>::const_flag_expression;
            using value_expression = xdynamic_view<uvt, S, L, typename FST::template rebind_t<uvt>>;
            using flag_expression = xdynamic_view<uft, S, L, typename FST::template rebind_t<uft>>;
            using const_value_expression = xdynamic_view<ucvt, S, L, typename FST::template rebind_t<ucvt>>;
            using const_flag_expression = xdynamic_view<ucft, S, L, typename FST::template rebind_t<ucft>>;

            value_expression value();
            const_value_expression value() const;

            flag_expression has_value();
            const_flag_expression has_value() const;
        };

        template <class CT, class S, layout_type L, class FST>
        struct xdynamic_view_base_impl<xoptional_expression_tag, CT, S, L, FST>
        {
            using type = xdynamic_view_optional<CT, S, L, FST>;
        };
    }

    /*************************************************
     * xbroadcast extension for optional expressions *
     *************************************************/

    namespace extension
    {
        template <class CT, class X>
        class xbroadcast_optional : public xoptional_empty_base<xbroadcast<CT, X>>
        {
        public:

            using expression_tag = xoptional_expression_tag;
            using value_expression = xbroadcast<xt::detail::value_expression_t<CT>, X>;
            using flag_expression = xbroadcast<xt::detail::flag_expression_t<CT>, X>;
            using const_value_expression = value_expression;
            using const_flag_expression = flag_expression;

            const_value_expression value() const;
            const_flag_expression has_value() const;
        };

        template <class CT, class X>
        struct xbroadcast_base_impl<xoptional_expression_tag, CT, X>
        {
            using type = xbroadcast_optional<CT, X>;
        };
    }

    /***************************************************
     * xfunctor_view extension for optional expression *
     ***************************************************/

    namespace extension
    {
        template <class F, class CT>
        class xfunctor_view_optional : public xoptional_empty_base<xfunctor_view<F, CT>>
        {
        public:

            using expression_tag = xoptional_expression_tag;
            using uvt = typename std::decay_t<CT>::value_expression;
            using uft = typename std::decay_t<CT>::flag_expression;
            using ucvt = typename std::decay_t<CT>::const_value_expression;
            using ucft = typename std::decay_t<CT>::const_flag_expression;
            using value_expression = xfunctor_view<F, uvt>;
            using flag_expression = uft;
            using const_value_expression = xfunctor_view<F, ucvt>;
            using const_flag_expression = ucft;

            value_expression value();
            const_value_expression value() const;

            flag_expression has_value();
            const_flag_expression has_value() const;
        };

        template <class F, class CT>
        struct xfunctor_view_base_impl<xoptional_expression_tag, F, CT>
        {
            using type = xfunctor_view_optional<F, CT>;
        };
    }

    /**************************************************
     * xindex_view extension for optional expressions *
     **************************************************/

    namespace extension
    {
        template <class CT, class I>
        class xindex_view_optional : public xoptional_empty_base<xindex_view<CT, I>>
        {
        public:

            using expression_tag = xoptional_expression_tag;
            using uvt = typename std::decay_t<CT>::value_expression;
            using ucvt = typename std::decay_t<CT>::const_value_expression;
            using uft = typename std::decay_t<CT>::flag_expression;
            using ucft = typename std::decay_t<CT>::const_flag_expression;
            using value_expression = xindex_view<uvt, I>;
            using flag_expression = xindex_view<uft, I>;
            using const_value_expression = xindex_view<ucvt, I>;
            using const_flag_expression = xindex_view<ucft, I>;

            value_expression value();
            const_value_expression value() const;

            flag_expression has_value();
            const_flag_expression has_value() const;
        };

        template <class CT, class I>
        struct xindex_view_base_impl<xoptional_expression_tag, CT, I>
        {
            using type = xindex_view_optional<CT, I>;
        };
    }

    /***********************************************
     * xreducer extension for optional expressions *
     ***********************************************/

    namespace extension
    {
        template <class F, class CT, class X, class O>
        class xreducer_optional : public xoptional_empty_base<xreducer<F, CT, X, O>>
        {
        public:

            using expression_tag = xoptional_expression_tag;
            using result_type = typename F::init_value_type;

            using rebound_result_type = typename result_type::value_type;
            using rebound_functors_type = typename F::template rebind_t<rebound_result_type>;
            using rebound_reduce_options_values = typename O::template rebind_t<rebound_result_type>;
            using rebound_reduce_options_flag = typename O::template rebind_t<bool>;

            using flag_reducer = xreducer_functors<xt::detail::optional_bitwise<bool>, xt::const_value<bool>>;
            using flag_expression = xreducer<flag_reducer, xt::detail::flag_expression_t<CT>, X, rebound_reduce_options_flag>;
            using value_expression = xreducer<rebound_functors_type, xt::detail::value_expression_t<CT>, X, rebound_reduce_options_values>;

            using const_value_expression = value_expression;
            using const_flag_expression = flag_expression;

            const_value_expression value() const;
            const_flag_expression has_value() const;
        };

        template <class F, class CT, class X, class O>
        struct xreducer_base_impl<xoptional_expression_tag, F, CT, X, O>
        {
            using type = xreducer_optional<F, CT, X, O>;
        };
    }

    /**********************************************
     * xrepeat extension for optional expressions *
     **********************************************/

    namespace extension
    {
        template <class CT, class X>
        class xrepeat_optional : public xoptional_empty_base<xrepeat<CT, X>>
        {
        public:

            using expression_tag = xoptional_expression_tag;
            using value_expression = xbroadcast<xt::detail::value_expression_t<CT>, X>;
            using flag_expression = xbroadcast<xt::detail::flag_expression_t<CT>, X>;
            using const_value_expression = value_expression;
            using const_flag_expression = flag_expression;

            const_value_expression value() const;
            const_flag_expression has_value() const;
        };

        template <class CT, class X>
        struct xrepeat_base_impl<xoptional_expression_tag, CT, X>
        {
            using type = xrepeat_optional<CT, X>;
        };
    }

    /****************************************************
     * xstrided_view extension for optional expressions *
     ****************************************************/

    namespace extension
    {
        template <class CT, class S, layout_type L, class FST>
        class xstrided_view_optional : public xoptional_empty_base<xstrided_view<CT, S, L, FST>>
        {
        public:

            using expression_tag = xoptional_expression_tag;
            using uvt = typename std::decay_t<CT>::value_expression;
            using uft = typename std::decay_t<CT>::flag_expression;
            using ucvt = typename std::decay_t<CT>::const_value_expression;
            using ucft = typename std::decay_t<CT>::const_flag_expression;
            using value_expression = xstrided_view<uvt, S, L, typename FST::template rebind_t<uvt>>;
            using flag_expression = xstrided_view<uft, S, L, typename FST::template rebind_t<uft>>;
            using const_value_expression = xstrided_view<ucvt, S, L, typename FST::template rebind_t<ucvt>>;
            using const_flag_expression = xstrided_view<ucft, S, L, typename FST::template rebind_t<ucft>>;

            value_expression value();
            const_value_expression value() const;

            flag_expression has_value();
            const_flag_expression has_value() const;
        };

        template <class CT, class S, layout_type L, class FST>
        struct xstrided_view_base_impl<xoptional_expression_tag, CT, S, L, FST>
        {
            using type = xstrided_view_optional<CT, S, L, FST>;
        };
    }

    /********************************************
     * xview extension for optional expressions *
     ********************************************/

    namespace extension
    {
        template <class CT, class... S>
        class xview_optional : public xoptional_empty_base<xview<CT, S...>>
        {
        public:

            using expression_tag = xoptional_expression_tag;
            using uvt = typename std::decay_t<CT>::value_expression;
            using uft = typename std::decay_t<CT>::flag_expression;
            using ucvt = typename std::decay_t<CT>::const_value_expression;
            using ucft = typename std::decay_t<CT>::const_flag_expression;
            using value_expression = xview<uvt, S...>;
            using flag_expression = xview<uft, S...>;
            using const_value_expression = xview<ucvt, S...>;
            using const_flag_expression = xview<ucft, S...>;

            value_expression value();
            const_value_expression value() const;

            flag_expression has_value();
            const_flag_expression has_value() const;
        };

        template <class CT, class... S>
        struct xview_base_impl<xoptional_expression_tag, CT, S...>
        {
            using type = xview_optional<CT, S...>;
        };
    }

    /*************************************************
     * xgenerator extension for generator expression *
     *************************************************/

    namespace extension
    {
        namespace detail
        {
            template <class F, class = void_t<int>>
            struct value_functor
            {
                using type = F;

                static type get(const F& f)
                {
                    return f;
                }
            };

            template <class F>
            struct value_functor<F, void_t<typename F::value_functor_type>>
            {
                using type = typename F::value_functor_type;

                static type get(const F& f)
                {
                    return f.value_functor();
                }
            };

            template <class F>
            using value_functor_t = typename value_functor<F>::type;

            struct always_true
            {
                template <class... T>
                bool operator()(T...) const
                {
                    return true;
                }
            };

            template <class F, class = void_t<int>>
            struct flag_functor
            {
                using type = always_true;

                static type get(const F&)
                {
                    return type();
                }
            };

            template <class F>
            struct flag_functor<F, void_t<typename F::flag_functor_type>>
            {
                using type = typename F::flag_functor_type;

                static type get(const F& f)
                {
                    return f.flag_functor();
                }
            };

            template <class F>
            using flag_functor_t = typename flag_functor<F>::type;
        }

        template <class F, class R, class S>
        class xgenerator_optional : public xoptional_empty_base<xgenerator<F, R, S>>
        {
        public:

            using expression_tag = xoptional_expression_tag;
            using value_closure = typename R::value_closure;
            using flag_closure = typename R::flag_closure;
            using value_functor = detail::value_functor_t<F>;
            using flag_functor = detail::flag_functor_t<F>;
            using value_expression = xgenerator<value_functor, value_closure, S>;
            using flag_expression = xgenerator<flag_functor, flag_closure, S>;
            using const_value_expression = value_expression;
            using const_flag_expression = flag_expression;

            const_value_expression value() const;
            const_flag_expression has_value() const;
        };

        template <class F, class R, class S>
        struct xgenerator_base_impl<xoptional_expression_tag, F, R, S>
        {
            using type = xgenerator_optional<F, R, S>;
        };
    }

    /***************************************
     * xoptional_empty_base implementation *
     ***************************************/

    namespace extension
    {
        template <class D>
        inline D& xoptional_empty_base<D>::derived_cast() noexcept
        {
            return *static_cast<D*>(this);
        }

        template <class D>
        inline const D& xoptional_empty_base<D>::derived_cast() const noexcept
        {
            return *static_cast<const D*>(this);
        }
    }

    /****************************************
     * xscalar_optional_base implementation *
     ****************************************/

    namespace extension
    {
        template <class CT>
        inline auto xscalar_optional_base<CT>::value() -> value_expression
        {
            return this->derived_cast().expression().value();
        }

        template <class CT>
        inline auto xscalar_optional_base<CT>::value() const -> const_value_expression
        {
            return this->derived_cast().expression().value();
        }

        template <class CT>
        inline auto xscalar_optional_base<CT>::has_value() -> flag_expression
        {
            return this->derived_cast().expression().has_value();
        }

        template <class CT>
        inline auto xscalar_optional_base<CT>::has_value() const -> const_flag_expression
        {
            return this->derived_cast().expression().has_value();
        }
    }

    /*******************************************
     * xcontainer_optional_base implementation *
     *******************************************/

    namespace extension
    {
        template <class T>
        inline auto xcontainer_optional_base<T>::value() -> value_expression
        {
            return value_expression(this->derived_cast().storage().value(), this->derived_cast().shape());
        }

        template <class T>
        inline auto xcontainer_optional_base<T>::value() const -> const_value_expression
        {
            return const_value_expression(this->derived_cast().storage().value(), this->derived_cast().shape());
        }

        template <class T>
        inline auto xcontainer_optional_base<T>::has_value() -> flag_expression
        {
            return flag_expression(this->derived_cast().storage().has_value(), this->derived_cast().shape());
        }

        template <class T>
        inline auto xcontainer_optional_base<T>::has_value() const -> const_flag_expression
        {
            return const_flag_expression(this->derived_cast().storage().has_value(), this->derived_cast().shape());
        }
    }

    /******************************************
     * xfunction_optional_base implementation *
     ******************************************/

    namespace extension
    {
        template <class F, class... CT>
        inline auto xfunction_optional_base<F, CT...>::value() const -> const_value_expression
        {
            return value_impl(std::make_index_sequence<sizeof...(CT)>());
        }

        template <class F, class... CT>
        inline auto xfunction_optional_base<F, CT...>::has_value() const -> const_flag_expression
        {
            return has_value_impl(std::make_index_sequence<sizeof...(CT)>());
        }

        template <class F, class... CT>
        template <std::size_t... I>
        inline auto xfunction_optional_base<F, CT...>::value_impl(std::index_sequence<I...>) const
            -> const_value_expression
        {
            return value_expression(
                value_functor(),
                xt::detail::split_optional_expression<CT>::value(std::get<I>(this->derived_cast().arguments()))...
            );
        }

        template <class F, class... CT>
        template <std::size_t... I>
        inline auto xfunction_optional_base<F, CT...>::has_value_impl(std::index_sequence<I...>) const
            -> const_flag_expression
        {
            return flag_expression(
                flag_functor(),
                xt::detail::split_optional_expression<CT>::has_value(
                    std::get<I>(this->derived_cast().arguments())
                )...
            );
        }
    }

    /*****************************************
     * xdynamic_view_optional implementation *
     *****************************************/

    namespace extension
    {
        template <class CT, class S, layout_type L, class FST>
        inline auto xdynamic_view_optional<CT, S, L, FST>::value() -> value_expression
        {
            return this->derived_cast().build_view(this->derived_cast().expression().value());
        }

        template <class CT, class S, layout_type L, class FST>
        inline auto xdynamic_view_optional<CT, S, L, FST>::value() const -> const_value_expression
        {
            return this->derived_cast().build_view(this->derived_cast().expression().value());
        }

        template <class CT, class S, layout_type L, class FST>
        inline auto xdynamic_view_optional<CT, S, L, FST>::has_value() -> flag_expression
        {
            return this->derived_cast().build_view(this->derived_cast().expression().has_value());
        }

        template <class CT, class S, layout_type L, class FST>
        inline auto xdynamic_view_optional<CT, S, L, FST>::has_value() const -> const_flag_expression
        {
            return this->derived_cast().build_view(this->derived_cast().expression().has_value());
        }
    }

    /**************************************
     * xbroadcast_optional implementation *
     **************************************/

    namespace extension
    {
        template <class CT, class X>
        inline auto xbroadcast_optional<CT, X>::value() const -> const_value_expression
        {
            return this->derived_cast().build_broadcast(this->derived_cast().expression().value());
        }

        template <class CT, class X>
        inline auto xbroadcast_optional<CT, X>::has_value() const -> const_flag_expression
        {
            return this->derived_cast().build_broadcast(this->derived_cast().expression().has_value());
        }
    }

    /*****************************************
     * xfunctor_view_optional implementation *
     *****************************************/

    namespace extension
    {
        template <class F, class CT>
        inline auto xfunctor_view_optional<F, CT>::value() -> value_expression
        {
            return this->derived_cast().build_functor_view(this->derived_cast().expression().value());
        }

        template <class F, class CT>
        inline auto xfunctor_view_optional<F, CT>::value() const -> const_value_expression
        {
            return this->derived_cast().build_functor_view(this->derived_cast().expression().value());
        }

        template <class F, class CT>
        inline auto xfunctor_view_optional<F, CT>::has_value() -> flag_expression
        {
            return this->derived_cast().expression().has_value();
        }

        template <class F, class CT>
        inline auto xfunctor_view_optional<F, CT>::has_value() const -> const_flag_expression
        {
            return this->derived_cast().expression().has_value();
        }
    }

    /***************************************
     * xindex_view_optional implementation *
     ***************************************/

    namespace extension
    {
        template <class CT, class I>
        inline auto xindex_view_optional<CT, I>::value() -> value_expression
        {
            return this->derived_cast().build_index_view(this->derived_cast().expression().value());
        };

        template <class CT, class I>
        inline auto xindex_view_optional<CT, I>::value() const -> const_value_expression
        {
            return this->derived_cast().build_index_view(this->derived_cast().expression().value());
        };

        template <class CT, class I>
        inline auto xindex_view_optional<CT, I>::has_value() -> flag_expression
        {
            return this->derived_cast().build_index_view(this->derived_cast().expression().has_value());
        };

        template <class CT, class I>
        inline auto xindex_view_optional<CT, I>::has_value() const -> const_flag_expression
        {
            return this->derived_cast().build_index_view(this->derived_cast().expression().has_value());
        };
    }

    /************************************
     * xreducer_optional implementation *
     ************************************/

    namespace extension
    {
        template <class F, class CT, class X, class O>
        inline auto xreducer_optional<F, CT, X, O>::value() const -> const_value_expression
        {
            auto func = this->derived_cast().functors();
            auto opts = this->derived_cast().options().template rebind<rebound_result_type>(
                this->derived_cast().options().initial_value.value(),
                this->derived_cast().options()
            );

            return this->derived_cast().build_reducer(
                this->derived_cast().expression().value(),
                func.template rebind<rebound_result_type>(),
                std::move(opts)
            );
        }

        template <class F, class CT, class X, class O>
        inline auto xreducer_optional<F, CT, X, O>::has_value() const -> const_flag_expression
        {
            auto opts = this->derived_cast().options().rebind(
                this->derived_cast().options().initial_value.has_value(),
                this->derived_cast().options()
            );

            return this->derived_cast().build_reducer(
                this->derived_cast().expression().has_value(),
                make_xreducer_functor(xt::detail::optional_bitwise<bool>(), xt::const_value<bool>(true)),
                std::move(opts)
            );
        }
    }

    /*****************************************
     * xstrided_view_optional implementation *
     *****************************************/

    namespace extension
    {
        template <class CT, class S, layout_type L, class FST>
        inline auto xstrided_view_optional<CT, S, L, FST>::value() -> value_expression
        {
            return this->derived_cast().build_view(this->derived_cast().expression().value());
        }

        template <class CT, class S, layout_type L, class FST>
        inline auto xstrided_view_optional<CT, S, L, FST>::value() const -> const_value_expression
        {
            return this->derived_cast().build_view(this->derived_cast().expression().value());
        }

        template <class CT, class S, layout_type L, class FST>
        inline auto xstrided_view_optional<CT, S, L, FST>::has_value() -> flag_expression
        {
            return this->derived_cast().build_view(this->derived_cast().expression().has_value());
        }

        template <class CT, class S, layout_type L, class FST>
        inline auto xstrided_view_optional<CT, S, L, FST>::has_value() const -> const_flag_expression
        {
            return this->derived_cast().build_view(this->derived_cast().expression().has_value());
        }
    }

    /*********************************
     * xview_optional implementation *
     *********************************/

    namespace extension
    {
        template <class CT, class... S>
        inline auto xview_optional<CT, S...>::value() -> value_expression
        {
            return this->derived_cast().build_view(this->derived_cast().expression().value());
        }

        template <class CT, class... S>
        inline auto xview_optional<CT, S...>::value() const -> const_value_expression
        {
            return this->derived_cast().build_view(this->derived_cast().expression().value());
        }

        template <class CT, class... S>
        inline auto xview_optional<CT, S...>::has_value() -> flag_expression
        {
            return this->derived_cast().build_view(this->derived_cast().expression().has_value());
        }

        template <class CT, class... S>
        inline auto xview_optional<CT, S...>::has_value() const -> const_flag_expression
        {
            return this->derived_cast().build_view(this->derived_cast().expression().has_value());
        }
    }

    /**************************************
     * xgenerator_optional implementation *
     **************************************/

    namespace extension
    {
        template <class F, class R, class S>
        inline auto xgenerator_optional<F, R, S>::value() const -> const_value_expression
        {
            return this->derived_cast().template build_generator<value_closure>(
                detail::value_functor<F>::get(this->derived_cast().functor())
            );
        }

        template <class F, class R, class S>
        inline auto xgenerator_optional<F, R, S>::has_value() const -> const_flag_expression
        {
            return this->derived_cast().template build_generator<flag_closure>(
                detail::flag_functor<F>::get(this->derived_cast().functor())
            );
        }
    }

    /********************************
     * sign function implementation *
     ********************************/

    namespace math
    {
        template <class T, class B>
        struct sign_impl<xtl::xoptional<T, B>>
        {
            static constexpr auto run(const xtl::xoptional<T, B>& x)
            {
                return sign(x);  // use overload declared above
            }
        };
    }

    template <class T, class B>
    inline auto sign(const xtl::xoptional<T, B>& e)
    {
        using value_type = std::decay_t<T>;
        return e.has_value() ? math::sign_impl<value_type>::run(e.value()) : xtl::missing<value_type>();
    }

    /******************************************
     * value() and has_value() implementation *
     ******************************************/

    template <class E, XTL_REQUIRES(is_xexpression<E>)>
    inline auto value(E&& e) -> detail::value_expression_t<E>
    {
        return detail::split_optional_expression<E>::value(std::forward<E>(e));
    }

    template <class E, XTL_REQUIRES(is_xexpression<E>)>
    inline auto has_value(E&& e) -> detail::flag_expression_t<E>
    {
        return detail::split_optional_expression<E>::has_value(std::forward<E>(e));
    }

    namespace detail
    {
        template <class T1, class T2>
        struct assign_data_impl
        {
            template <class E1, class E2>
            static void run(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial)
            {
                E1& de1 = e1.derived_cast();
                const E2& de2 = e2.derived_cast();

                decltype(auto) bde1 = xt::value(de1);
                decltype(auto) hde1 = xt::has_value(de1);
                xexpression_assigner_base<xtensor_expression_tag>::assign_data(bde1, xt::value(de2), trivial);
                xexpression_assigner_base<xtensor_expression_tag>::assign_data(hde1, xt::has_value(de2), trivial);
            }
        };

        template <class T>
        struct xarray_assigner
        {
            template <class E1, class E2>
            static void assign(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial)
            {
                E1& de1 = e1.derived_cast();
                const E2& de2 = e2.derived_cast();
                xarray<bool> mask = xt::full_like(de2, true);

                decltype(auto) bde1 = xt::value(de1);
                decltype(auto) hde1 = xt::has_value(de1);
                xexpression_assigner_base<xtensor_expression_tag>::assign_data(bde1, e2, trivial);
                xexpression_assigner_base<xtensor_expression_tag>::assign_data(hde1, mask, trivial);
            }
        };

        template <class T>
        struct xarray_assigner<xtl::xoptional<T>>
        {
            template <class E1, class E2>
            static void assign(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial)
            {
                xexpression_assigner_base<xtensor_expression_tag>::assign_data(e1, e2, trivial);
            }
        };

        template <>
        struct assign_data_impl<xoptional_expression_tag, xtensor_expression_tag>
        {
            template <class E1, class E2>
            static void run(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial)
            {
                xarray_assigner<typename E2::value_type>::assign(e1, e2, trivial);
            }
        };

        template <>
        struct assign_data_impl<xtensor_expression_tag, xoptional_expression_tag>
        {
            template <class E1, class E2>
            static void run(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial)
            {
                xexpression_assigner_base<xtensor_expression_tag>::assign_data(e1, e2, trivial);
            }
        };
    }

    template <class E1, class E2>
    inline void xexpression_assigner_base<xoptional_expression_tag>::assign_data(
        xexpression<E1>& e1,
        const xexpression<E2>& e2,
        bool trivial
    )
    {
        detail::assign_data_impl<typename E1::expression_tag, typename E2::expression_tag>::run(e1, e2, trivial);
    }
}

#endif
