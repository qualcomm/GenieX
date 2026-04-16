/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_FUNCTOR_VIEW_HPP
#define XTENSOR_FUNCTOR_VIEW_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <xtl/xproxy_wrapper.hpp>

#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xaccessible.hpp"
#include "../core/xexpression.hpp"
#include "../core/xiterator.hpp"
#include "../core/xsemantic.hpp"
#include "../utils/xutils.hpp"

namespace xt
{
    /**
     * @defgroup xt_xfunctor_view
     *
     * Chunked array container.
     * Defined in ``xtensor/xfunctor_view.hpp``
     */

    /************************************************
     * xfunctor_view and xfunctor_adaptor extension *
     ************************************************/

    namespace extension
    {
        template <class Tag, class F, class CT>
        struct xfunctor_view_base_impl;

        template <class F, class CT>
        struct xfunctor_view_base_impl<xtensor_expression_tag, F, CT>
        {
            using type = xtensor_empty_base;
        };

        template <class F, class CT>
        struct xfunctor_view_base : xfunctor_view_base_impl<xexpression_tag_t<CT>, F, CT>
        {
        };

        template <class F, class CT>
        using xfunctor_view_base_t = typename xfunctor_view_base<F, CT>::type;
    }

    /*************************************
     * xfunctor_applier_base declaration *
     *************************************/

    template <class F, class IT>
    class xfunctor_iterator;

    template <class F, class ST>
    class xfunctor_stepper;

    template <class D>
    class xfunctor_applier_base : private xaccessible<D>
    {
    public:

        using self_type = xfunctor_applier_base<D>;
        using inner_types = xcontainer_inner_types<D>;
        using xexpression_type = typename inner_types::xexpression_type;
        using undecay_expression = typename inner_types::undecay_expression;
        using functor_type = typename inner_types::functor_type;
        using accessible_base = xaccessible<D>;

        using extension_base = extension::xfunctor_view_base_t<functor_type, undecay_expression>;
        using expression_tag = typename extension_base::expression_tag;

        using value_type = typename functor_type::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = typename functor_type::pointer;
        using const_pointer = typename functor_type::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using shape_type = typename xexpression_type::shape_type;
        using strides_type = xtl::mpl::eval_if_t<
            has_strides<xexpression_type>,
            detail::expr_strides_type<xexpression_type>,
            get_strides_type<shape_type>>;
        using backstrides_type = xtl::mpl::eval_if_t<
            has_strides<xexpression_type>,
            detail::expr_backstrides_type<xexpression_type>,
            get_strides_type<shape_type>>;

        using inner_shape_type = typename xexpression_type::inner_shape_type;
        using inner_strides_type = xtl::mpl::eval_if_t<
            has_strides<xexpression_type>,
            detail::expr_inner_strides_type<xexpression_type>,
            get_strides_type<shape_type>>;
        using inner_backstrides_type = xtl::mpl::eval_if_t<
            has_strides<xexpression_type>,
            detail::expr_inner_backstrides_type<xexpression_type>,
            get_strides_type<shape_type>>;

        using bool_load_type = xt::bool_load_type<value_type>;

        static constexpr layout_type static_layout = xexpression_type::static_layout;
        static constexpr bool contiguous_layout = xexpression_type::contiguous_layout;

        using stepper = xfunctor_stepper<functor_type, typename xexpression_type::stepper>;
        using const_stepper = xfunctor_stepper<const functor_type, typename xexpression_type::const_stepper>;

        template <layout_type L>
        using layout_iterator = xfunctor_iterator<functor_type, typename xexpression_type::template layout_iterator<L>>;
        template <layout_type L>
        using const_layout_iterator = xfunctor_iterator<
            const functor_type,
            typename xexpression_type::template const_layout_iterator<L>>;

        template <layout_type L>
        using reverse_layout_iterator = xfunctor_iterator<
            functor_type,
            typename xexpression_type::template reverse_layout_iterator<L>>;
        template <layout_type L>
        using const_reverse_layout_iterator = xfunctor_iterator<
            const functor_type,
            typename xexpression_type::template const_reverse_layout_iterator<L>>;

        template <class S, layout_type L>
        using broadcast_iterator = xfunctor_iterator<functor_type, xiterator<typename xexpression_type::stepper, S, L>>;
        template <class S, layout_type L>
        using const_broadcast_iterator = xfunctor_iterator<
            functor_type,
            xiterator<typename xexpression_type::const_stepper, S, L>>;

        template <class S, layout_type L>
        using reverse_broadcast_iterator = xfunctor_iterator<
            functor_type,
            typename xexpression_type::template reverse_broadcast_iterator<S, L>>;
        template <class S, layout_type L>
        using const_reverse_broadcast_iterator = xfunctor_iterator<
            functor_type,
            typename xexpression_type::template const_reverse_broadcast_iterator<S, L>>;

        using linear_iterator = xfunctor_iterator<functor_type, typename xexpression_type::linear_iterator>;
        using const_linear_iterator = xfunctor_iterator<const functor_type, typename xexpression_type::const_linear_iterator>;
        using reverse_linear_iterator = xfunctor_iterator<functor_type, typename xexpression_type::reverse_linear_iterator>;
        using const_reverse_linear_iterator = xfunctor_iterator<
            const functor_type,
            typename xexpression_type::const_reverse_linear_iterator>;

        using iterator = xfunctor_iterator<functor_type, typename xexpression_type::iterator>;
        using const_iterator = xfunctor_iterator<const functor_type, typename xexpression_type::const_iterator>;
        using reverse_iterator = xfunctor_iterator<functor_type, typename xexpression_type::reverse_iterator>;
        using const_reverse_iterator = xfunctor_iterator<const functor_type, typename xexpression_type::const_reverse_iterator>;

        explicit xfunctor_applier_base(undecay_expression) noexcept;

        template <class Func, class E>
        xfunctor_applier_base(Func&&, E&&) noexcept;

        size_type size() const noexcept;
        const inner_shape_type& shape() const noexcept;
        const inner_strides_type& strides() const noexcept;
        const inner_backstrides_type& backstrides() const noexcept;
        using accessible_base::dimension;
        using accessible_base::shape;

        layout_type layout() const noexcept;
        bool is_contiguous() const noexcept;

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        reference unchecked(Args... args);

        template <class IT>
        reference element(IT first, IT last);

        template <class... Args>
        const_reference operator()(Args... args) const;

        template <class... Args>
        const_reference unchecked(Args... args) const;

        template <class IT>
        const_reference element(IT first, IT last) const;

        using accessible_base::at;
        using accessible_base::operator[];
        using accessible_base::back;
        using accessible_base::front;
        using accessible_base::periodic;

        using accessible_base::in_bounds;

        xexpression_type& expression() noexcept;
        const xexpression_type& expression() const noexcept;

        template <class S>
        bool broadcast_shape(S& shape, bool reuse_cache = false) const;

        template <class S>
        bool has_linear_assign(const S& strides) const;

        template <class FCT = functor_type>
        auto data_element(size_type i)
            -> decltype(std::declval<FCT>()(std::declval<undecay_expression>().data_element(i)))
        {
            return m_functor(m_e.data_element(i));
        }

        template <class FCT = functor_type>
        auto data_element(size_type i) const
            -> decltype(std::declval<FCT>()(std::declval<const undecay_expression>().data_element(i)))
        {
            return m_functor(m_e.data_element(i));
        }

        template <class FCT = functor_type>
        auto flat(size_type i) -> decltype(std::declval<FCT>()(std::declval<undecay_expression>().flat(i)))
        {
            return m_functor(m_e.flat(i));
        }

        template <class FCT = functor_type>
        auto flat(size_type i) const
            -> decltype(std::declval<FCT>()(std::declval<const undecay_expression>().flat(i)))
        {
            return m_functor(m_e.flat(i));
        }

        // The following functions are defined inline because otherwise signatures
        // don't match on GCC.
        template <
            class align,
            class requested_type = typename xexpression_type::value_type,
            std::size_t N = xt_simd::simd_traits<requested_type>::size,
            class FCT = functor_type>
        auto load_simd(size_type i) const
            -> decltype(std::declval<FCT>().template proxy_simd_load<align, requested_type, N>(
                std::declval<undecay_expression>(),
                i
            ))
        {
            return m_functor.template proxy_simd_load<align, requested_type, N>(m_e, i);
        }

        template <class align, class simd, class FCT = functor_type>
        auto store_simd(size_type i, const simd& e)
            -> decltype(std::declval<FCT>()
                            .template proxy_simd_store<align>(std::declval<undecay_expression>(), i, e))
        {
            return m_functor.template proxy_simd_store<align>(m_e, i, e);
        }

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        auto begin() noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        auto end() noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        auto begin() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        auto end() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        auto cbegin() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        auto cend() const noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        auto rbegin() noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        auto rend() noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        auto rbegin() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        auto rend() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        auto crbegin() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        auto crend() const noexcept;

        template <class S, layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        broadcast_iterator<S, L> begin(const S& shape) noexcept;
        template <class S, layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        broadcast_iterator<S, L> end(const S& shape) noexcept;

        template <class S, layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_broadcast_iterator<S, L> begin(const S& shape) const noexcept;
        template <class S, layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_broadcast_iterator<S, L> end(const S& shape) const noexcept;
        template <class S, layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_broadcast_iterator<S, L> cbegin(const S& shape) const noexcept;
        template <class S, layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_broadcast_iterator<S, L> cend(const S& shape) const noexcept;

        template <class S, layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        reverse_broadcast_iterator<S, L> rbegin(const S& shape) noexcept;
        template <class S, layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        reverse_broadcast_iterator<S, L> rend(const S& shape) noexcept;

        template <class S, layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_reverse_broadcast_iterator<S, L> rbegin(const S& shape) const noexcept;
        template <class S, layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_reverse_broadcast_iterator<S, L> rend(const S& shape) const noexcept;
        template <class S, layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_reverse_broadcast_iterator<S, L> crbegin(const S& shape) const noexcept;
        template <class S, layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_reverse_broadcast_iterator<S, L> crend(const S& shape) const noexcept;

        linear_iterator linear_begin() noexcept;
        linear_iterator linear_end() noexcept;

        const_linear_iterator linear_begin() const noexcept;
        const_linear_iterator linear_end() const noexcept;
        const_linear_iterator linear_cbegin() const noexcept;
        const_linear_iterator linear_cend() const noexcept;

        reverse_linear_iterator linear_rbegin() noexcept;
        reverse_linear_iterator linear_rend() noexcept;

        const_reverse_linear_iterator linear_rbegin() const noexcept;
        const_reverse_linear_iterator linear_rend() const noexcept;
        const_reverse_linear_iterator linear_crbegin() const noexcept;
        const_reverse_linear_iterator linear_crend() const noexcept;

        template <class S>
        stepper stepper_begin(const S& shape) noexcept;
        template <class S>
        stepper stepper_end(const S& shape, layout_type l) noexcept;
        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape, layout_type l) const noexcept;

    protected:

        undecay_expression m_e;
        functor_type m_functor;

    private:

        friend class xaccessible<D>;
        friend class xconst_accessible<D>;
    };

    template <class D, class T>
    struct has_simd_interface<xfunctor_applier_base<D>, T>
        : std::conjunction<
              has_simd_type<T>,
              has_simd_interface<typename xfunctor_applier_base<D>::xexpression_type>,
              detail::has_simd_interface_impl<xfunctor_applier_base<D>, T>>
    {
    };

    /********************************
     * xfunctor_view_temporary_type *
     ********************************/

    namespace detail
    {
        // TODO replace with xexpression_for_shape ...
        template <class F, class S, layout_type L>
        struct functorview_temporary_type_impl
        {
            using type = xarray<typename F::value_type, L>;
        };

        template <class F, class T, std::size_t N, layout_type L>
        struct functorview_temporary_type_impl<F, std::array<T, N>, L>
        {
            using type = xtensor<typename F::value_type, N, L>;
        };
    }

    template <class F, class E>
    struct xfunctor_view_temporary_type
    {
        using type = typename detail::functorview_temporary_type_impl<F, typename E::shape_type, E::static_layout>::type;
    };

    /*****************************
     * xfunctor_view declaration *
     *****************************/

    template <class F, class CT>
    class xfunctor_view;

    template <class F, class CT>
    struct xcontainer_inner_types<xfunctor_view<F, CT>>
    {
        using xexpression_type = std::decay_t<CT>;
        using undecay_expression = CT;
        using functor_type = std::decay_t<F>;
        using reference = decltype(std::declval<F>()(std::declval<xexpression_type>()()));
        using const_reference = decltype(std::declval<F>()(std::declval<const xexpression_type>()()));
        using size_type = typename xexpression_type::size_type;
        using temporary_type = typename xfunctor_view_temporary_type<F, xexpression_type>::type;
    };

    template <class F, class CT, class T>
    struct has_simd_interface<xfunctor_view<F, CT>, T>
        : has_simd_interface<xfunctor_applier_base<xfunctor_view<F, CT>>, T>
    {
    };

    /**
     * View of an xexpression .
     *
     * The xt::xfunctor_view class is an expression addressing its elements by applying a functor to the
     * corresponding element of an underlying expression.
     * Unlike e.g. xgenerator, an xt::xfunctor_view is an lvalue.
     * It is used e.g. to access real and imaginary parts of complex expressions.
     *
     * xt::xfunctor_view has a view semantics and can be used on any expression.
     * For a similar feature with a container semantics, one can use xt::xfunctor_adaptor.
     *
     * xt::xfunctor_view is not meant to be used directly, but through helper functions such
     * as xt::real or xt::imag.
     *
     * @ingroup xt_xfunctor_view
     * @tparam F the functor type to be applied to the elements of specified expression.
     * @tparam CT the closure type of the xt::xexpression type underlying this view
     * @see xt::real, xt::imag
     */
    template <class F, class CT>
    class xfunctor_view : public xfunctor_applier_base<xfunctor_view<F, CT>>,
                          public xview_semantic<xfunctor_view<F, CT>>,
                          public extension::xfunctor_view_base_t<F, CT>
    {
    public:

        using self_type = xfunctor_view<F, CT>;
        using semantic_base = xview_semantic<self_type>;

        // constructors
        using xfunctor_applier_base<self_type>::xfunctor_applier_base;

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

        template <class E>
        using rebind_t = xfunctor_view<F, E>;

        template <class E>
        rebind_t<E> build_functor_view(E&& e) const;

    private:

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        void assign_temporary_impl(temporary_type&& tmp);
        friend class xview_semantic<self_type>;
        friend class xaccessible<self_type>;
    };

    /********************************
     * xfunctor_adaptor declaration *
     ********************************/

    template <class F, class CT>
    class xfunctor_adaptor;

    template <class F, class CT>
    struct xcontainer_inner_types<xfunctor_adaptor<F, CT>>
    {
        using xexpression_type = std::decay_t<CT>;
        using undecay_expression = CT;
        using functor_type = std::decay_t<F>;
        using reference = typename functor_type::reference;
        using const_reference = typename functor_type::const_reference;
        using size_type = typename xexpression_type::size_type;
        using temporary_type = typename xfunctor_view_temporary_type<F, xexpression_type>::type;
    };

    template <class F, class CT, class T>
    struct has_simd_interface<xfunctor_adaptor<F, CT>, T>
        : has_simd_interface<xfunctor_applier_base<xfunctor_adaptor<F, CT>>, T>
    {
    };

    /**
     * Adapt a container with a functor, forwarding methods such as resize / reshape.
     *
     * xt::xfunctor_adaptor has a container semantics and can only be used with containers.
     * For a similar feature with a view semantics, one can use xt::xfunctor_view.
     *
     * @ingroup xt_xfunctor_view
     * @tparam F the functor type to be applied to the elements of specified expression.
     * @tparam CT the closure type of the xt::xexpression type underlying this view
     * @see xt::xfunctor_view
     */
    template <class F, class CT>
    class xfunctor_adaptor : public xfunctor_applier_base<xfunctor_adaptor<F, CT>>,
                             public xcontainer_semantic<xfunctor_adaptor<F, CT>>,
                             public extension::xfunctor_view_base_t<F, CT>
    {
    public:

        using self_type = xfunctor_adaptor<F, CT>;
        using semantic_base = xcontainer_semantic<self_type>;
        using xexpression_type = std::decay_t<CT>;
        using base_type = xfunctor_applier_base<self_type>;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename xexpression_type::strides_type;
        // constructors
        using xfunctor_applier_base<self_type>::xfunctor_applier_base;

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

        template <class S = shape_type>
        auto resize(S&& shape, bool force = false);

        template <class S = shape_type>
        auto resize(S&& shape, layout_type l);

        template <class S = shape_type>
        auto resize(S&& shape, const strides_type& strides);

        template <class S = shape_type>
        auto& reshape(S&& shape, layout_type layout = base_type::static_layout) &;

    private:

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        void assign_temporary_impl(temporary_type&& tmp);
        friend class xcontainer_semantic<self_type>;
        friend class xaccessible<self_type>;
    };

    /*********************************
     * xfunctor_iterator declaration *
     *********************************/

    template <class R>
    struct xproxy_inner_types
    {
        using reference = R;
        using pointer = std::add_pointer_t<std::remove_reference_t<R>>;
    };

    namespace detail
    {
        template <class F, class IT>
        struct xfunctor_invoker
        {
            using type = decltype(std::declval<F>()(*(std::declval<IT>())));
        };

        template <class F, class IT>
        using xfunctor_invoker_t = typename xfunctor_invoker<F, IT>::type;
    }

    template <class F, class IT>
    class xfunctor_iterator : public xtl::xrandom_access_iterator_base<
                                  xfunctor_iterator<F, IT>,
                                  typename std::decay_t<F>::value_type,
                                  typename std::iterator_traits<IT>::difference_type,
                                  typename xproxy_inner_types<detail::xfunctor_invoker_t<F, IT>>::pointer,
                                  typename xproxy_inner_types<detail::xfunctor_invoker_t<F, IT>>::reference>
    {
    public:

        using functor_type = F;
        using subiterator_traits = std::iterator_traits<IT>;

        using proxy_inner = xproxy_inner_types<detail::xfunctor_invoker_t<F, IT>>;
        using value_type = typename functor_type::value_type;
        using reference = typename proxy_inner::reference;
        using pointer = typename proxy_inner::pointer;
        using difference_type = typename subiterator_traits::difference_type;
        using iterator_category = typename subiterator_traits::iterator_category;

        using self_type = xfunctor_iterator<F, IT>;

        xfunctor_iterator(const IT&, functor_type*);

        self_type& operator++();
        self_type& operator--();

        self_type& operator+=(difference_type n);
        self_type& operator-=(difference_type n);

        difference_type operator-(xfunctor_iterator rhs) const;

        reference operator*() const;
        pointer operator->() const;

        bool equal(const xfunctor_iterator& rhs) const;
        bool less_than(const xfunctor_iterator& rhs) const;

    private:

        IT m_it;
        functor_type* p_functor;
    };

    template <class F, class IT>
    bool operator==(const xfunctor_iterator<F, IT>& lhs, const xfunctor_iterator<F, IT>& rhs);

    template <class F, class IT>
    bool operator<(const xfunctor_iterator<F, IT>& lhs, const xfunctor_iterator<F, IT>& rhs);

    /********************************
     * xfunctor_stepper declaration *
     ********************************/

    template <class F, class ST>
    class xfunctor_stepper
    {
    public:

        using functor_type = F;

        using proxy_inner = xproxy_inner_types<detail::xfunctor_invoker_t<F, ST>>;
        using value_type = typename functor_type::value_type;
        using reference = typename proxy_inner::reference;
        using pointer = std::remove_reference_t<reference>*;
        using size_type = typename ST::size_type;
        using difference_type = typename ST::difference_type;

        using shape_type = typename ST::shape_type;

        xfunctor_stepper() = default;
        xfunctor_stepper(const ST&, functor_type*);

        reference operator*() const;

        void step(size_type dim);
        void step_back(size_type dim);
        void step(size_type dim, size_type n);
        void step_back(size_type dim, size_type n);
        void reset(size_type dim);
        void reset_back(size_type dim);

        void to_begin();
        void to_end(layout_type);

    private:

        ST m_stepper;
        functor_type* p_functor;
    };

    /****************************************
     * xfunctor_applier_base implementation *
     ****************************************/

    /**
     * @name Constructors
     */
    //@{

    /**
     * Constructs an xfunctor_applier_base expression wrappering the specified xt::xexpression.
     *
     * @param e the underlying expression
     */
    template <class D>
    inline xfunctor_applier_base<D>::xfunctor_applier_base(undecay_expression e) noexcept
        : m_e(e)
        , m_functor(functor_type())
    {
    }

    /**
     * Constructs an xfunctor_applier_base expression wrappering the specified xt::xexpression.
     *
     * @param func the functor to be applied to the elements of the underlying expression.
     * @param e the underlying expression
     */
    template <class D>
    template <class Func, class E>
    inline xfunctor_applier_base<D>::xfunctor_applier_base(Func&& func, E&& e) noexcept
        : m_e(std::forward<E>(e))
        , m_functor(std::forward<Func>(func))
    {
    }

    //@}

    /**
     * @name Size and shape
     */

    /**
     * Returns the size of the expression.
     */
    template <class D>
    inline auto xfunctor_applier_base<D>::size() const noexcept -> size_type
    {
        return m_e.size();
    }

    /**
     * Returns the shape of the expression.
     */
    template <class D>
    inline auto xfunctor_applier_base<D>::shape() const noexcept -> const inner_shape_type&
    {
        return m_e.shape();
    }

    /**
     * Returns the strides of the expression.
     */
    template <class D>
    inline auto xfunctor_applier_base<D>::strides() const noexcept -> const inner_strides_type&
    {
        return m_e.strides();
    }

    /**
     * Returns the backstrides of the expression.
     */
    template <class D>
    inline auto xfunctor_applier_base<D>::backstrides() const noexcept -> const inner_backstrides_type&
    {
        return m_e.backstrides();
    }

    /**
     * Returns the layout_type of the expression.
     */
    template <class D>
    inline layout_type xfunctor_applier_base<D>::layout() const noexcept
    {
        return m_e.layout();
    }

    template <class D>
    inline bool xfunctor_applier_base<D>::is_contiguous() const noexcept
    {
        return m_e.is_contiguous();
    }

    //@}

    /**
     * @name Data
     */

    /**
     * Returns a reference to the element at the specified position in the expression.
     * @param args a list of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the expression.
     */
    template <class D>
    template <class... Args>
    inline auto xfunctor_applier_base<D>::operator()(Args... args) -> reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        return m_functor(m_e(args...));
    }

    /**
     * Returns a reference to the element at the specified position in the expression.
     * @param args a list of indices specifying the position in the expression. Indices
     * must be unsigned integers, the number of indices must be equal to the number of
     * dimensions of the expression, else the behavior is undefined.
     *
     * @warning This method is meant for performance, for expressions with a dynamic
     * number of dimensions (i.e. not known at compile time). Since it may have
     * undefined behavior (see parameters), operator() should be preferred whenever
     * it is possible.
     * @warning This method is NOT compatible with broadcasting, meaning the following
     * code has undefined behavior:
     * @code{.cpp}
     * xt::xarray<double> a = {{0, 1}, {2, 3}};
     * xt::xarray<double> b = {0, 1};
     * auto fd = a + b;
     * double res = fd.unchecked(0, 1);
     * @endcode
     */
    template <class D>
    template <class... Args>
    inline auto xfunctor_applier_base<D>::unchecked(Args... args) -> reference
    {
        return m_functor(m_e.unchecked(args...));
    }

    /**
     * Returns a reference to the element at the specified position in the expression.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the function.
     */
    template <class D>
    template <class IT>
    inline auto xfunctor_applier_base<D>::element(IT first, IT last) -> reference
    {
        XTENSOR_TRY(check_element_index(shape(), first, last));
        return m_functor(m_e.element(first, last));
    }

    /**
     * Returns a constant reference to the element at the specified position in the expression.
     * @param args a list of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the expression.
     */
    template <class D>
    template <class... Args>
    inline auto xfunctor_applier_base<D>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        return m_functor(m_e(args...));
    }

    /**
     * Returns a constant reference to the element at the specified position in the expression.
     * @param args a list of indices specifying the position in the expression. Indices
     * must be unsigned integers, the number of indices must be equal to the number of
     * dimensions of the expression, else the behavior is undefined.
     *
     * @warning This method is meant for performance, for expressions with a dynamic
     * number of dimensions (i.e. not known at compile time). Since it may have
     * undefined behavior (see parameters), operator() should be preferred whenever
     * it is possible.
     * @warning This method is NOT compatible with broadcasting, meaning the following
     * code has undefined behavior:
     * @code{.cpp}
     * xt::xarray<double> a = {{0, 1}, {2, 3}};
     * xt::xarray<double> b = {0, 1};
     * auto fd = a + b;
     * double res = fd.uncheked(0, 1);
     * @endcode
     */
    template <class D>
    template <class... Args>
    inline auto xfunctor_applier_base<D>::unchecked(Args... args) const -> const_reference
    {
        return m_functor(m_e.unchecked(args...));
    }

    /**
     * Returns a constant reference to the element at the specified position in the expression.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the function.
     */
    template <class D>
    template <class IT>
    inline auto xfunctor_applier_base<D>::element(IT first, IT last) const -> const_reference
    {
        XTENSOR_TRY(check_element_index(shape(), first, last));
        return m_functor(m_e.element(first, last));
    }

    /**
     * Returns a reference to the underlying expression of the view.
     */
    template <class D>
    inline auto xfunctor_applier_base<D>::expression() noexcept -> xexpression_type&
    {
        return m_e;
    }

    /**
     * Returns a consttant reference to the underlying expression of the view.
     */
    template <class D>
    inline auto xfunctor_applier_base<D>::expression() const noexcept -> const xexpression_type&
    {
        return m_e;
    }

    //@}

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the function to the specified parameter.
     * @param shape the result shape
     * @param reuse_cache boolean for reusing a previously computed shape
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class D>
    template <class S>
    inline bool xfunctor_applier_base<D>::broadcast_shape(S& shape, bool reuse_cache) const
    {
        return m_e.broadcast_shape(shape, reuse_cache);
    }

    /**
     * Checks whether the xfunctor_applier_base can be linearly assigned to an expression
     * with the specified strides.
     *
     * @return a boolean indicating whether a linear assign is possible
     */
    template <class D>
    template <class S>
    inline bool xfunctor_applier_base<D>::has_linear_assign(const S& strides) const
    {
        return m_e.has_linear_assign(strides);
    }

    //@}

    /**
     * @name Iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xfunctor_applier_base<D>::begin() noexcept
    {
        return xfunctor_iterator<functor_type, decltype(m_e.template begin<L>())>(
            m_e.template begin<L>(),
            &m_functor
        );
    }

    /**
     * Returns an iterator to the element following the last element
     * of the expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xfunctor_applier_base<D>::end() noexcept
    {
        return xfunctor_iterator<functor_type, decltype(m_e.template end<L>())>(m_e.template end<L>(), &m_functor);
    }

    /**
     * Returns a constant iterator to the first element of the expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xfunctor_applier_base<D>::begin() const noexcept
    {
        return this->template cbegin<L>();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xfunctor_applier_base<D>::end() const noexcept
    {
        return this->template cend<L>();
    }

    /**
     * Returns a constant iterator to the first element of the expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xfunctor_applier_base<D>::cbegin() const noexcept
    {
        return xfunctor_iterator<const functor_type, decltype(m_e.template cbegin<L>())>(
            m_e.template cbegin<L>(),
            &m_functor
        );
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xfunctor_applier_base<D>::cend() const noexcept
    {
        return xfunctor_iterator<const functor_type, decltype(m_e.template cend<L>())>(
            m_e.template cend<L>(),
            &m_functor
        );
    }

    //@}

    /**
     * @name Broadcast iterators
     */
    //@{
    /**
     * Returns a constant iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xfunctor_applier_base<D>::begin(const S& shape) noexcept -> broadcast_iterator<S, L>
    {
        return broadcast_iterator<S, L>(m_e.template begin<S, L>(shape), &m_functor);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xfunctor_applier_base<D>::end(const S& shape) noexcept -> broadcast_iterator<S, L>
    {
        return broadcast_iterator<S, L>(m_e.template end<S, L>(shape), &m_functor);
    }

    /**
     * Returns a constant iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xfunctor_applier_base<D>::begin(const S& shape) const noexcept
        -> const_broadcast_iterator<S, L>
    {
        return cbegin<S, L>(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xfunctor_applier_base<D>::end(const S& shape) const noexcept -> const_broadcast_iterator<S, L>
    {
        return cend<S, L>(shape);
    }

    /**
     * Returns a constant iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xfunctor_applier_base<D>::cbegin(const S& shape) const noexcept
        -> const_broadcast_iterator<S, L>
    {
        return const_broadcast_iterator<S, L>(m_e.template cbegin<S, L>(shape), &m_functor);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xfunctor_applier_base<D>::cend(const S& shape) const noexcept -> const_broadcast_iterator<S, L>
    {
        return const_broadcast_iterator<S, L>(m_e.template cend<S, L>(shape), &m_functor);
    }

    //@}

    /**
     * @name Reverse iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the reversed expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xfunctor_applier_base<D>::rbegin() noexcept
    {
        return xfunctor_iterator<functor_type, decltype(m_e.template rbegin<L>())>(
            m_e.template rbegin<L>(),
            &m_functor
        );
    }

    /**
     * Returns an iterator to the element following the last element
     * of the reversed expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xfunctor_applier_base<D>::rend() noexcept
    {
        return xfunctor_iterator<functor_type, decltype(m_e.template rend<L>())>(
            m_e.template rend<L>(),
            &m_functor
        );
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xfunctor_applier_base<D>::rbegin() const noexcept
    {
        return this->template crbegin<L>();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xfunctor_applier_base<D>::rend() const noexcept
    {
        return this->template crend<L>();
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xfunctor_applier_base<D>::crbegin() const noexcept
    {
        return xfunctor_iterator<const functor_type, decltype(m_e.template crbegin<L>())>(
            m_e.template crbegin<L>(),
            &m_functor
        );
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xfunctor_applier_base<D>::crend() const noexcept
    {
        return xfunctor_iterator<const functor_type, decltype(m_e.template crend<L>())>(
            m_e.template crend<L>(),
            &m_functor
        );
    }

    //@}

    /**
     * @name Reverse broadcast iterators
     */

    /**
     * Returns an iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xfunctor_applier_base<D>::rbegin(const S& shape) noexcept -> reverse_broadcast_iterator<S, L>
    {
        return reverse_broadcast_iterator<S, L>(m_e.template rbegin<S, L>(shape), &m_functor);
    }

    /**
     * Returns an iterator to the element following the last element of the
     * reversed expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xfunctor_applier_base<D>::rend(const S& shape) noexcept -> reverse_broadcast_iterator<S, L>
    {
        return reverse_broadcast_iterator<S, L>(m_e.template rend<S, L>(shape), &m_functor);
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xfunctor_applier_base<D>::rbegin(const S& shape) const noexcept
        -> const_reverse_broadcast_iterator<S, L>
    {
        return crbegin<S, L>(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xfunctor_applier_base<D>::rend(const S& /*shape*/) const noexcept
        -> const_reverse_broadcast_iterator<S, L>
    {
        return crend<S, L>();
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xfunctor_applier_base<D>::crbegin(const S& /*shape*/) const noexcept
        -> const_reverse_broadcast_iterator<S, L>
    {
        return const_reverse_broadcast_iterator<S, L>(m_e.template crbegin<S, L>(), &m_functor);
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xfunctor_applier_base<D>::crend(const S& shape) const noexcept
        -> const_reverse_broadcast_iterator<S, L>
    {
        return const_reverse_broadcast_iterator<S, L>(m_e.template crend<S, L>(shape), &m_functor);
    }

    //@}

    template <class D>
    inline auto xfunctor_applier_base<D>::linear_begin() noexcept -> linear_iterator
    {
        return linear_iterator(m_e.linear_begin(), &m_functor);
    }

    template <class D>
    inline auto xfunctor_applier_base<D>::linear_end() noexcept -> linear_iterator
    {
        return linear_iterator(m_e.linear_end(), &m_functor);
    }

    template <class D>
    inline auto xfunctor_applier_base<D>::linear_begin() const noexcept -> const_linear_iterator
    {
        return const_linear_iterator(m_e.linear_begin(), &m_functor);
    }

    template <class D>
    inline auto xfunctor_applier_base<D>::linear_end() const noexcept -> const_linear_iterator
    {
        return const_linear_iterator(m_e.linear_end(), &m_functor);
    }

    template <class D>
    inline auto xfunctor_applier_base<D>::linear_cbegin() const noexcept -> const_linear_iterator
    {
        return const_linear_iterator(m_e.linear_cbegin(), &m_functor);
    }

    template <class D>
    inline auto xfunctor_applier_base<D>::linear_cend() const noexcept -> const_linear_iterator
    {
        return const_linear_iterator(m_e.linear_cend(), &m_functor);
    }

    template <class D>
    inline auto xfunctor_applier_base<D>::linear_rbegin() noexcept -> reverse_linear_iterator
    {
        return reverse_linear_iterator(m_e.linear_rbegin(), &m_functor);
    }

    template <class D>
    inline auto xfunctor_applier_base<D>::linear_rend() noexcept -> reverse_linear_iterator
    {
        return reverse_linear_iterator(m_e.linear_rend(), &m_functor);
    }

    template <class D>
    inline auto xfunctor_applier_base<D>::linear_rbegin() const noexcept -> const_reverse_linear_iterator
    {
        return const_reverse_linear_iterator(m_e.linear_rbegin(), &m_functor);
    }

    template <class D>
    inline auto xfunctor_applier_base<D>::linear_rend() const noexcept -> const_reverse_linear_iterator
    {
        return const_reverse_linear_iterator(m_e.linear_rend(), &m_functor);
    }

    template <class D>
    inline auto xfunctor_applier_base<D>::linear_crbegin() const noexcept -> const_reverse_linear_iterator
    {
        return const_reverse_linear_iterator(m_e.linear_crbegin(), &m_functor);
    }

    template <class D>
    inline auto xfunctor_applier_base<D>::linear_crend() const noexcept -> const_reverse_linear_iterator
    {
        return const_reverse_linear_iterator(m_e.linear_crend(), &m_functor);
    }

    /***************
     * stepper api *
     ***************/

    template <class D>
    template <class S>
    inline auto xfunctor_applier_base<D>::stepper_begin(const S& shape) noexcept -> stepper
    {
        return stepper(m_e.stepper_begin(shape), &m_functor);
    }

    template <class D>
    template <class S>
    inline auto xfunctor_applier_base<D>::stepper_end(const S& shape, layout_type l) noexcept -> stepper
    {
        return stepper(m_e.stepper_end(shape, l), &m_functor);
    }

    template <class D>
    template <class S>
    inline auto xfunctor_applier_base<D>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        const xexpression_type& const_m_e = m_e;
        return const_stepper(const_m_e.stepper_begin(shape), &m_functor);
    }

    template <class D>
    template <class S>
    inline auto xfunctor_applier_base<D>::stepper_end(const S& shape, layout_type l) const noexcept
        -> const_stepper
    {
        const xexpression_type& const_m_e = m_e;
        return const_stepper(const_m_e.stepper_end(shape, l), &m_functor);
    }

    /********************************
     * xfunctor_view implementation *
     ********************************/

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class F, class CT>
    template <class E>
    inline auto xfunctor_view<F, CT>::operator=(const xexpression<E>& e) -> self_type&
    {
        bool cond = (e.derived_cast().shape().size() == this->dimension())
                    && std::equal(this->shape().begin(), this->shape().end(), e.derived_cast().shape().begin());
        if (!cond)
        {
            semantic_base::operator=(broadcast(e.derived_cast(), this->shape()));
        }
        else
        {
            semantic_base::operator=(e);
        }
        return *this;
    }

    //@}

    template <class F, class CT>
    template <class E>
    inline auto xfunctor_view<F, CT>::operator=(const E& e) -> disable_xexpression<E, self_type>&
    {
        std::fill(this->begin(), this->end(), e);
        return *this;
    }

    template <class F, class CT>
    inline void xfunctor_view<F, CT>::assign_temporary_impl(temporary_type&& tmp)
    {
        std::copy(tmp.cbegin(), tmp.cend(), this->begin());
    }

    template <class F, class CT>
    template <class E>
    inline auto xfunctor_view<F, CT>::build_functor_view(E&& e) const -> rebind_t<E>
    {
        return rebind_t<E>((this->m_functor), std::forward<E>(e));
    }

    /***********************************
     * xfunctor_adaptor implementation *
     ***********************************/

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class F, class CT>
    template <class E>
    inline auto xfunctor_adaptor<F, CT>::operator=(const xexpression<E>& e) -> self_type&
    {
        const auto& de = e.derived_cast();
        this->m_e.resize(de.shape());

        if (this->layout() == de.layout())
        {
            std::copy(de.linear_begin(), de.linear_end(), this->linear_begin());
        }
        else
        {
            // note: does this even select the current layout of *this* for iteration?
            std::copy(de.begin(), de.end(), this->begin());
        }

        return *this;
    }

    //@}

    template <class F, class CT>
    template <class S>
    auto xfunctor_adaptor<F, CT>::resize(S&& shape, bool force)
    {
        this->m_e.resize(std::forward<S>(shape), force);
    }

    template <class F, class CT>
    template <class S>
    auto xfunctor_adaptor<F, CT>::resize(S&& shape, layout_type l)
    {
        this->m_e.resize(std::forward<S>(shape), l);
    }

    template <class F, class CT>
    template <class S>
    auto xfunctor_adaptor<F, CT>::resize(S&& shape, const strides_type& strides)
    {
        this->m_e.resize(std::forward<S>(shape), strides);
    }

    template <class F, class CT>
    template <class S>
    auto& xfunctor_adaptor<F, CT>::reshape(S&& shape, layout_type layout) &
    {
        this->m_e.reshape(std::forward<S>(shape), layout);
        return *this;
    }

    /************************************
     * xfunctor_iterator implementation *
     ************************************/

    template <class F, class IT>
    xfunctor_iterator<F, IT>::xfunctor_iterator(const IT& it, functor_type* pf)
        : m_it(it)
        , p_functor(pf)
    {
    }

    template <class F, class IT>
    inline auto xfunctor_iterator<F, IT>::operator++() -> self_type&
    {
        ++m_it;
        return *this;
    }

    template <class F, class IT>
    inline auto xfunctor_iterator<F, IT>::operator--() -> self_type&
    {
        --m_it;
        return *this;
    }

    template <class F, class IT>
    inline auto xfunctor_iterator<F, IT>::operator+=(difference_type n) -> self_type&
    {
        m_it += n;
        return *this;
    }

    template <class F, class IT>
    inline auto xfunctor_iterator<F, IT>::operator-=(difference_type n) -> self_type&
    {
        m_it -= n;
        return *this;
    }

    template <class F, class IT>
    inline auto xfunctor_iterator<F, IT>::operator-(xfunctor_iterator rhs) const -> difference_type
    {
        return m_it - rhs.m_it;
    }

    template <class F, class IT>
    auto xfunctor_iterator<F, IT>::operator*() const -> reference
    {
        return (*p_functor)(*m_it);
    }

    template <class F, class IT>
    auto xfunctor_iterator<F, IT>::operator->() const -> pointer
    {
        return &(operator*());
    }

    template <class F, class IT>
    auto xfunctor_iterator<F, IT>::equal(const xfunctor_iterator& rhs) const -> bool
    {
        return m_it == rhs.m_it;
    }

    template <class F, class IT>
    auto xfunctor_iterator<F, IT>::less_than(const xfunctor_iterator& rhs) const -> bool
    {
        return m_it < rhs.m_it;
    }

    template <class F, class IT>
    bool operator==(const xfunctor_iterator<F, IT>& lhs, const xfunctor_iterator<F, IT>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class F, class IT>
    bool operator<(const xfunctor_iterator<F, IT>& lhs, const xfunctor_iterator<F, IT>& rhs)
    {
        return !lhs.less_than(rhs);
    }

    /***********************************
     * xfunctor_stepper implementation *
     ***********************************/

    template <class F, class ST>
    xfunctor_stepper<F, ST>::xfunctor_stepper(const ST& stepper, functor_type* pf)
        : m_stepper(stepper)
        , p_functor(pf)
    {
    }

    template <class F, class ST>
    auto xfunctor_stepper<F, ST>::operator*() const -> reference
    {
        return (*p_functor)(*m_stepper);
    }

    template <class F, class ST>
    void xfunctor_stepper<F, ST>::step(size_type dim)
    {
        m_stepper.step(dim);
    }

    template <class F, class ST>
    void xfunctor_stepper<F, ST>::step_back(size_type dim)
    {
        m_stepper.step_back(dim);
    }

    template <class F, class ST>
    void xfunctor_stepper<F, ST>::step(size_type dim, size_type n)
    {
        m_stepper.step(dim, n);
    }

    template <class F, class ST>
    void xfunctor_stepper<F, ST>::step_back(size_type dim, size_type n)
    {
        m_stepper.step_back(dim, n);
    }

    template <class F, class ST>
    void xfunctor_stepper<F, ST>::reset(size_type dim)
    {
        m_stepper.reset(dim);
    }

    template <class F, class ST>
    void xfunctor_stepper<F, ST>::reset_back(size_type dim)
    {
        m_stepper.reset_back(dim);
    }

    template <class F, class ST>
    void xfunctor_stepper<F, ST>::to_begin()
    {
        m_stepper.to_begin();
    }

    template <class F, class ST>
    void xfunctor_stepper<F, ST>::to_end(layout_type l)
    {
        m_stepper.to_end(l);
    }
}
#endif
