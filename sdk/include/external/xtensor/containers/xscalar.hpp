/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_SCALAR_HPP
#define XTENSOR_SCALAR_HPP

#include <array>
#include <cstddef>
#include <utility>

#include <xtl/xtype_traits.hpp>

#include "../core/xaccessible.hpp"
#include "../core/xexpression.hpp"
#include "../core/xiterable.hpp"
#include "../core/xlayout.hpp"
#include "../utils/xtensor_simd.hpp"

namespace xt
{

    /*********************
     * xscalar extension *
     *********************/

    namespace extension
    {
        template <class Tag, class CT>
        struct xscalar_base_impl;

        template <class CT>
        struct xscalar_base_impl<xtensor_expression_tag, CT>
        {
            using type = xtensor_empty_base;
        };

        template <class CT>
        struct xscalar_base : xscalar_base_impl<get_expression_tag_t<std::decay_t<CT>>, CT>
        {
        };

        template <class CT>
        using xscalar_base_t = typename xscalar_base<CT>::type;
    }

    /***********
     * xscalar *
     ***********/

    // xscalar is a cheap wrapper for a scalar value as an xexpression.
    template <class CT>
    class xscalar;

    template <bool is_const, class CT>
    class xscalar_stepper;

    template <bool is_const, class CT>
    class xdummy_iterator;

    template <class CT>
    struct xiterable_inner_types<xscalar<CT>>
    {
        using value_type = std::decay_t<CT>;
        using inner_shape_type = std::array<std::size_t, 0>;
        using shape_type = inner_shape_type;
        using const_stepper = xscalar_stepper<true, CT>;
        using stepper = xscalar_stepper<false, CT>;
    };

    template <class CT>
    struct xcontainer_inner_types<xscalar<CT>>
    {
        using value_type = std::decay_t<CT>;
        using reference = value_type&;
        using const_reference = const value_type&;
        using size_type = std::size_t;
    };

    template <class CT>
    class xscalar : public xsharable_expression<xscalar<CT>>,
                    private xiterable<xscalar<CT>>,
                    private xaccessible<xscalar<CT>>,
                    public extension::xscalar_base_t<CT>
    {
    public:

        using self_type = xscalar<CT>;
        using xexpression_type = std::decay_t<CT>;
        using extension_base = extension::xscalar_base_t<CT>;
        using accessible_base = xaccessible<self_type>;
        using expression_tag = typename extension_base::expression_tag;
        using inner_types = xcontainer_inner_types<self_type>;

        using value_type = typename inner_types::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = typename inner_types::size_type;
        using difference_type = std::ptrdiff_t;
        using simd_value_type = xt_simd::simd_type<value_type>;
        using bool_load_type = xt::bool_load_type<value_type>;

        using iterable_base = xiterable<self_type>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using shape_type = inner_shape_type;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        template <layout_type L>
        using layout_iterator = typename iterable_base::template layout_iterator<L>;
        template <layout_type L>
        using const_layout_iterator = typename iterable_base::template const_layout_iterator<L>;

        template <layout_type L>
        using reverse_layout_iterator = typename iterable_base::template reverse_layout_iterator<L>;
        template <layout_type L>
        using const_reverse_layout_iterator = typename iterable_base::template const_reverse_layout_iterator<L>;

        template <class S, layout_type L>
        using broadcast_iterator = typename iterable_base::template broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using const_broadcast_iterator = typename iterable_base::template const_broadcast_iterator<S, L>;

        template <class S, layout_type L>
        using reverse_broadcast_iterator = typename iterable_base::template reverse_broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using const_reverse_broadcast_iterator = typename iterable_base::template const_reverse_broadcast_iterator<S, L>;

        using iterator = value_type*;
        using const_iterator = const value_type*;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        using dummy_iterator = xdummy_iterator<false, CT>;
        using const_dummy_iterator = xdummy_iterator<true, CT>;

        static constexpr layout_type static_layout = layout_type::any;
        static constexpr bool contiguous_layout = true;

        xscalar() noexcept;
        xscalar(CT value) noexcept;

        operator value_type&() noexcept;
        operator const value_type&() const noexcept;

        size_type size() const noexcept;
        const shape_type& shape() const noexcept;
        size_type shape(size_type i) const noexcept;
        layout_type layout() const noexcept;
        bool is_contiguous() const noexcept;
        using accessible_base::dimension;
        using accessible_base::shape;

        template <class... Args>
        reference operator()(Args...) noexcept;
        template <class... Args>
        reference unchecked(Args...) noexcept;

        template <class... Args>
        const_reference operator()(Args...) const noexcept;
        template <class... Args>
        const_reference unchecked(Args...) const noexcept;

        using accessible_base::at;
        using accessible_base::operator[];
        using accessible_base::back;
        using accessible_base::front;
        using accessible_base::in_bounds;
        using accessible_base::periodic;

        template <class It>
        reference element(It, It) noexcept;

        template <class It>
        const_reference element(It, It) const noexcept;

        xexpression_type& expression() noexcept;
        const xexpression_type& expression() const noexcept;

        template <class S>
        bool broadcast_shape(S& shape, bool reuse_cache = false) const noexcept;

        template <class S>
        bool has_linear_assign(const S& strides) const noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        iterator begin() noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        iterator end() noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_iterator begin() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_iterator end() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_iterator cbegin() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_iterator cend() const noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        reverse_iterator rbegin() noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        reverse_iterator rend() noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_reverse_iterator rbegin() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_reverse_iterator rend() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_reverse_iterator crbegin() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_reverse_iterator crend() const noexcept;

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

        iterator linear_begin() noexcept;
        iterator linear_end() noexcept;

        const_iterator linear_begin() const noexcept;
        const_iterator linear_end() const noexcept;
        const_iterator linear_cbegin() const noexcept;
        const_iterator linear_cend() const noexcept;

        reverse_iterator linear_rbegin() noexcept;
        reverse_iterator linear_rend() noexcept;

        const_reverse_iterator linear_rbegin() const noexcept;
        const_reverse_iterator linear_rend() const noexcept;
        const_reverse_iterator linear_crbegin() const noexcept;
        const_reverse_iterator linear_crend() const noexcept;

        template <class S>
        stepper stepper_begin(const S& shape) noexcept;
        template <class S>
        stepper stepper_end(const S& shape, layout_type l) noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape, layout_type l) const noexcept;

        dummy_iterator dummy_begin() noexcept;
        dummy_iterator dummy_end() noexcept;

        const_dummy_iterator dummy_begin() const noexcept;
        const_dummy_iterator dummy_end() const noexcept;

        reference data_element(size_type i) noexcept;
        const_reference data_element(size_type i) const noexcept;

        reference flat(size_type i) noexcept;
        const_reference flat(size_type i) const noexcept;

        template <class align, class simd = simd_value_type>
        void store_simd(size_type i, const simd& e);
        template <class align, class requested_type = value_type, std::size_t N = xt_simd::simd_traits<requested_type>::size>
        xt_simd::simd_return_type<value_type, requested_type> load_simd(size_type i) const;

    private:

        CT m_value;

        friend class xconst_iterable<self_type>;
        friend class xiterable<self_type>;
        friend class xaccessible<self_type>;
        friend class xconst_accessible<self_type>;
    };

    namespace detail
    {
        template <class E>
        struct is_xscalar_impl : std::false_type
        {
        };

        template <class E>
        struct is_xscalar_impl<xscalar<E>> : std::true_type
        {
        };
    }

    template <class E>
    using is_xscalar = detail::is_xscalar_impl<E>;

    template <class E>
    concept xscalar_concept = is_xscalar<std::decay_t<E>>::value;

    namespace detail
    {
        template <class... E>
        struct all_xscalar
        {
            static constexpr bool value = std::conjunction<is_xscalar<std::decay_t<E>>...>::value;
        };
    }

    // Note: MSVC bug workaround. Cannot just define
    // template <class... E>
    // using all_xscalar = std::conjunction<is_xscalar<std::decay_t<E>>...>;

    template <class... E>
    using all_xscalar = detail::all_xscalar<E...>;

    /******************
     * xref and xcref *
     ******************/

    template <class T>
    xscalar<T&> xref(T& t);

    template <class T>
    xscalar<const T&> xcref(T& t);

    /*******************
     * xscalar_stepper *
     *******************/

    template <bool is_const, class CT>
    class xscalar_stepper
    {
    public:

        using self_type = xscalar_stepper<is_const, CT>;
        using storage_type = std::conditional_t<is_const, const xscalar<CT>, xscalar<CT>>;

        using value_type = typename storage_type::value_type;
        using reference = std::
            conditional_t<is_const, typename storage_type::const_reference, typename storage_type::reference>;
        using pointer = std::conditional_t<is_const, typename storage_type::const_pointer, typename storage_type::pointer>;
        using size_type = typename storage_type::size_type;
        using difference_type = typename storage_type::difference_type;
        using shape_type = typename storage_type::shape_type;

        template <class requested_type>
        using simd_return_type = xt_simd::simd_return_type<value_type, requested_type>;

        xscalar_stepper(storage_type* c) noexcept;

        reference operator*() const noexcept;

        void step(size_type dim, size_type n = 1) noexcept;
        void step_back(size_type dim, size_type n = 1) noexcept;
        void reset(size_type dim) noexcept;
        void reset_back(size_type dim) noexcept;

        void to_begin() noexcept;
        void to_end(layout_type l) noexcept;

        template <class T>
        simd_return_type<T> step_simd();

        void step_leading();

    private:

        storage_type* p_c;
    };

    /*******************
     * xdummy_iterator *
     *******************/

    namespace detail
    {
        template <bool is_const, class CT>
        using dummy_reference_t = std::
            conditional_t<is_const, typename xscalar<CT>::const_reference, typename xscalar<CT>::reference>;

        template <bool is_const, class CT>
        using dummy_pointer_t = std::
            conditional_t<is_const, typename xscalar<CT>::const_pointer, typename xscalar<CT>::pointer>;
    }

    template <bool is_const, class CT>
    class xdummy_iterator : public xtl::xrandom_access_iterator_base<
                                xdummy_iterator<is_const, CT>,
                                typename xscalar<CT>::value_type,
                                typename xscalar<CT>::difference_type,
                                detail::dummy_pointer_t<is_const, CT>,
                                detail::dummy_reference_t<is_const, CT>>
    {
    public:

        using self_type = xdummy_iterator<is_const, CT>;
        using storage_type = std::conditional_t<is_const, const xscalar<CT>, xscalar<CT>>;

        using value_type = typename storage_type::value_type;
        using reference = detail::dummy_reference_t<is_const, CT>;
        using pointer = detail::dummy_pointer_t<is_const, CT>;
        using difference_type = typename storage_type::difference_type;
        using iterator_category = std::random_access_iterator_tag;

        explicit xdummy_iterator(storage_type* c) noexcept;

        self_type& operator++() noexcept;
        self_type& operator--() noexcept;

        self_type& operator+=(difference_type n) noexcept;
        self_type& operator-=(difference_type n) noexcept;

        difference_type operator-(const self_type& rhs) const noexcept;

        reference operator*() const noexcept;

        bool equal(const self_type& rhs) const noexcept;
        bool less_than(const self_type& rhs) const noexcept;

    private:

        storage_type* p_c;
    };

    template <bool is_const, class CT>
    bool
    operator==(const xdummy_iterator<is_const, CT>& lhs, const xdummy_iterator<is_const, CT>& rhs) noexcept;

    template <bool is_const, class CT>
    bool operator<(const xdummy_iterator<is_const, CT>& lhs, const xdummy_iterator<is_const, CT>& rhs) noexcept;

    template <class T>
    struct is_not_xdummy_iterator : std::true_type
    {
    };

    template <bool is_const, class CT>
    struct is_not_xdummy_iterator<xdummy_iterator<is_const, CT>> : std::false_type
    {
    };

    /*****************************
     * linear_begin / linear_end *
     *****************************/

    template <class CT>
    XTENSOR_CONSTEXPR_RETURN auto linear_begin(xscalar<CT>& c) noexcept -> decltype(c.dummy_begin())
    {
        return c.dummy_begin();
    }

    template <class CT>
    XTENSOR_CONSTEXPR_RETURN auto linear_end(xscalar<CT>& c) noexcept -> decltype(c.dummy_end())
    {
        return c.dummy_end();
    }

    template <class CT>
    XTENSOR_CONSTEXPR_RETURN auto linear_begin(const xscalar<CT>& c) noexcept -> decltype(c.dummy_begin())
    {
        return c.dummy_begin();
    }

    template <class CT>
    XTENSOR_CONSTEXPR_RETURN auto linear_end(const xscalar<CT>& c) noexcept -> decltype(c.dummy_end())
    {
        return c.dummy_end();
    }

    /**************************
     * xscalar implementation *
     **************************/

    // This constructor will not compile when CT is a reference type.
    template <class CT>
    inline xscalar<CT>::xscalar() noexcept
        : m_value()
    {
    }

    template <class CT>
    inline xscalar<CT>::xscalar(CT value) noexcept
        : m_value(value)
    {
    }

    template <class CT>
    inline xscalar<CT>::operator value_type&() noexcept
    {
        return m_value;
    }

    template <class CT>
    inline xscalar<CT>::operator const value_type&() const noexcept
    {
        return m_value;
    }

    template <class CT>
    inline auto xscalar<CT>::size() const noexcept -> size_type
    {
        return 1;
    }

    template <class CT>
    inline auto xscalar<CT>::shape() const noexcept -> const shape_type&
    {
        static std::array<size_type, 0> zero_shape;
        return zero_shape;
    }

    template <class CT>
    inline auto xscalar<CT>::shape(size_type) const noexcept -> size_type
    {
        return 0;
    }

    template <class CT>
    inline layout_type xscalar<CT>::layout() const noexcept
    {
        return static_layout;
    }

    template <class CT>
    inline bool xscalar<CT>::is_contiguous() const noexcept
    {
        return true;
    }

    template <class CT>
    template <class... Args>
    inline auto xscalar<CT>::operator()(Args...) noexcept -> reference
    {
        XTENSOR_CHECK_DIMENSION((std::array<int, 0>()), Args()...);
        return m_value;
    }

    template <class CT>
    template <class... Args>
    inline auto xscalar<CT>::unchecked(Args...) noexcept -> reference
    {
        return m_value;
    }

    template <class CT>
    template <class... Args>
    inline auto xscalar<CT>::operator()(Args...) const noexcept -> const_reference
    {
        XTENSOR_CHECK_DIMENSION((std::array<int, 0>()), Args()...);
        return m_value;
    }

    template <class CT>
    template <class... Args>
    inline auto xscalar<CT>::unchecked(Args...) const noexcept -> const_reference
    {
        return m_value;
    }

    template <class CT>
    template <class It>
    inline auto xscalar<CT>::element(It, It) noexcept -> reference
    {
        return m_value;
    }

    template <class CT>
    template <class It>
    inline auto xscalar<CT>::element(It, It) const noexcept -> const_reference
    {
        return m_value;
    }

    template <class CT>
    inline auto xscalar<CT>::expression() noexcept -> xexpression_type&
    {
        return m_value;
    }

    template <class CT>
    inline auto xscalar<CT>::expression() const noexcept -> const xexpression_type&
    {
        return m_value;
    }

    template <class CT>
    template <class S>
    inline bool xscalar<CT>::broadcast_shape(S&, bool) const noexcept
    {
        return true;
    }

    template <class CT>
    template <class S>
    inline bool xscalar<CT>::has_linear_assign(const S&) const noexcept
    {
        return true;
    }

    template <class CT>
    template <layout_type L>
    inline auto xscalar<CT>::begin() noexcept -> iterator
    {
        return &m_value;
    }

    template <class CT>
    template <layout_type L>
    inline auto xscalar<CT>::end() noexcept -> iterator
    {
        return &m_value + 1;
    }

    template <class CT>
    template <layout_type L>
    inline auto xscalar<CT>::begin() const noexcept -> const_iterator
    {
        return &m_value;
    }

    template <class CT>
    template <layout_type L>
    inline auto xscalar<CT>::end() const noexcept -> const_iterator
    {
        return &m_value + 1;
    }

    template <class CT>
    template <layout_type L>
    inline auto xscalar<CT>::cbegin() const noexcept -> const_iterator
    {
        return &m_value;
    }

    template <class CT>
    template <layout_type L>
    inline auto xscalar<CT>::cend() const noexcept -> const_iterator
    {
        return &m_value + 1;
    }

    template <class CT>
    template <layout_type L>
    inline auto xscalar<CT>::rbegin() noexcept -> reverse_iterator
    {
        return reverse_iterator(end());
    }

    template <class CT>
    template <layout_type L>
    inline auto xscalar<CT>::rend() noexcept -> reverse_iterator
    {
        return reverse_iterator(begin());
    }

    template <class CT>
    template <layout_type L>
    inline auto xscalar<CT>::rbegin() const noexcept -> const_reverse_iterator
    {
        return crbegin();
    }

    template <class CT>
    template <layout_type L>
    inline auto xscalar<CT>::rend() const noexcept -> const_reverse_iterator
    {
        return crend();
    }

    template <class CT>
    template <layout_type L>
    inline auto xscalar<CT>::crbegin() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(cend());
    }

    template <class CT>
    template <layout_type L>
    inline auto xscalar<CT>::crend() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(cbegin());
    }

    /*****************************
     * Broadcasting iterator api *
     *****************************/

    template <class CT>
    template <class S, layout_type L>
    inline auto xscalar<CT>::begin(const S& shape) noexcept -> broadcast_iterator<S, L>
    {
        return iterable_base::template begin<S, L>(shape);
    }

    template <class CT>
    template <class S, layout_type L>
    inline auto xscalar<CT>::end(const S& shape) noexcept -> broadcast_iterator<S, L>
    {
        return iterable_base::template end<S, L>(shape);
    }

    template <class CT>
    template <class S, layout_type L>
    inline auto xscalar<CT>::begin(const S& shape) const noexcept -> const_broadcast_iterator<S, L>
    {
        return iterable_base::template begin<S, L>(shape);
    }

    template <class CT>
    template <class S, layout_type L>
    inline auto xscalar<CT>::end(const S& shape) const noexcept -> const_broadcast_iterator<S, L>
    {
        return iterable_base::template end<S, L>(shape);
    }

    template <class CT>
    template <class S, layout_type L>
    inline auto xscalar<CT>::cbegin(const S& shape) const noexcept -> const_broadcast_iterator<S, L>
    {
        return iterable_base::template cbegin<S, L>(shape);
    }

    template <class CT>
    template <class S, layout_type L>
    inline auto xscalar<CT>::cend(const S& shape) const noexcept -> const_broadcast_iterator<S, L>
    {
        return iterable_base::template cend<S, L>(shape);
    }

    template <class CT>
    template <class S, layout_type L>
    inline auto xscalar<CT>::rbegin(const S& shape) noexcept -> reverse_broadcast_iterator<S, L>
    {
        return iterable_base::template rbegin<S, L>(shape);
    }

    template <class CT>
    template <class S, layout_type L>
    inline auto xscalar<CT>::rend(const S& shape) noexcept -> reverse_broadcast_iterator<S, L>
    {
        return iterable_base::template rend<S, L>(shape);
    }

    template <class CT>
    template <class S, layout_type L>
    inline auto xscalar<CT>::rbegin(const S& shape) const noexcept -> const_reverse_broadcast_iterator<S, L>
    {
        return iterable_base::template rbegin<S, L>(shape);
    }

    template <class CT>
    template <class S, layout_type L>
    inline auto xscalar<CT>::rend(const S& shape) const noexcept -> const_reverse_broadcast_iterator<S, L>
    {
        return iterable_base::template rend<S, L>(shape);
    }

    template <class CT>
    template <class S, layout_type L>
    inline auto xscalar<CT>::crbegin(const S& shape) const noexcept -> const_reverse_broadcast_iterator<S, L>
    {
        return iterable_base::template crbegin<S, L>(shape);
    }

    template <class CT>
    template <class S, layout_type L>
    inline auto xscalar<CT>::crend(const S& shape) const noexcept -> const_reverse_broadcast_iterator<S, L>
    {
        return iterable_base::template crend<S, L>(shape);
    }

    template <class CT>
    inline auto xscalar<CT>::linear_begin() noexcept -> iterator
    {
        return this->template begin<XTENSOR_DEFAULT_LAYOUT>();
    }

    template <class CT>
    inline auto xscalar<CT>::linear_end() noexcept -> iterator
    {
        return this->template end<XTENSOR_DEFAULT_LAYOUT>();
    }

    template <class CT>
    inline auto xscalar<CT>::linear_begin() const noexcept -> const_iterator
    {
        return this->template begin<XTENSOR_DEFAULT_LAYOUT>();
    }

    template <class CT>
    inline auto xscalar<CT>::linear_end() const noexcept -> const_iterator
    {
        return this->template end<XTENSOR_DEFAULT_LAYOUT>();
    }

    template <class CT>
    inline auto xscalar<CT>::linear_cbegin() const noexcept -> const_iterator
    {
        return this->template cbegin<XTENSOR_DEFAULT_LAYOUT>();
    }

    template <class CT>
    inline auto xscalar<CT>::linear_cend() const noexcept -> const_iterator
    {
        return this->template cend<XTENSOR_DEFAULT_LAYOUT>();
    }

    template <class CT>
    inline auto xscalar<CT>::linear_rbegin() noexcept -> reverse_iterator
    {
        return this->template rbegin<XTENSOR_DEFAULT_LAYOUT>();
    }

    template <class CT>
    inline auto xscalar<CT>::linear_rend() noexcept -> reverse_iterator
    {
        return this->template rend<XTENSOR_DEFAULT_LAYOUT>();
    }

    template <class CT>
    inline auto xscalar<CT>::linear_rbegin() const noexcept -> const_reverse_iterator
    {
        return this->template rbegin<XTENSOR_DEFAULT_LAYOUT>();
    }

    template <class CT>
    inline auto xscalar<CT>::linear_rend() const noexcept -> const_reverse_iterator
    {
        return this->template rend<XTENSOR_DEFAULT_LAYOUT>();
    }

    template <class CT>
    inline auto xscalar<CT>::linear_crbegin() const noexcept -> const_reverse_iterator
    {
        return this->template crbegin<XTENSOR_DEFAULT_LAYOUT>();
    }

    template <class CT>
    inline auto xscalar<CT>::linear_crend() const noexcept -> const_reverse_iterator
    {
        return this->template crend<XTENSOR_DEFAULT_LAYOUT>();
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::stepper_begin(const S&) noexcept -> stepper
    {
        return stepper(this, false);
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::stepper_end(const S&, layout_type) noexcept -> stepper
    {
        return stepper(this);
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::stepper_begin(const S&) const noexcept -> const_stepper
    {
        return const_stepper(this);
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::stepper_end(const S&, layout_type) const noexcept -> const_stepper
    {
        return const_stepper(this);
    }

    template <class CT>
    inline auto xscalar<CT>::dummy_begin() noexcept -> dummy_iterator
    {
        return dummy_iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::dummy_end() noexcept -> dummy_iterator
    {
        return dummy_iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::dummy_begin() const noexcept -> const_dummy_iterator
    {
        return const_dummy_iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::dummy_end() const noexcept -> const_dummy_iterator
    {
        return const_dummy_iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::data_element(size_type) noexcept -> reference
    {
        return m_value;
    }

    template <class CT>
    inline auto xscalar<CT>::data_element(size_type) const noexcept -> const_reference
    {
        return m_value;
    }

    template <class CT>
    inline auto xscalar<CT>::flat(size_type) noexcept -> reference
    {
        return m_value;
    }

    template <class CT>
    inline auto xscalar<CT>::flat(size_type) const noexcept -> const_reference
    {
        return m_value;
    }

    template <class CT>
    template <class align, class simd>
    inline void xscalar<CT>::store_simd(size_type, const simd& e)
    {
        m_value = static_cast<value_type>(e[0]);
    }

    template <class CT>
    template <class align, class requested_type, std::size_t N>
    inline auto xscalar<CT>::load_simd(size_type) const
        -> xt_simd::simd_return_type<value_type, requested_type>
    {
        return xt_simd::broadcast_as<requested_type>(m_value);
    }

    template <class T>
    inline xscalar<T&> xref(T& t)
    {
        return xscalar<T&>(t);
    }

    template <class T>
    inline xscalar<const T&> xcref(T& t)
    {
        return xscalar<const T&>(t);
    }

    /**********************************
     * xscalar_stepper implementation *
     **********************************/

    template <bool is_const, class CT>
    inline xscalar_stepper<is_const, CT>::xscalar_stepper(storage_type* c) noexcept
        : p_c(c)
    {
    }

    template <bool is_const, class CT>
    inline auto xscalar_stepper<is_const, CT>::operator*() const noexcept -> reference
    {
        return p_c->operator()();
    }

    template <bool is_const, class CT>
    inline void xscalar_stepper<is_const, CT>::step(size_type /*dim*/, size_type /*n*/) noexcept
    {
    }

    template <bool is_const, class CT>
    inline void xscalar_stepper<is_const, CT>::step_back(size_type /*dim*/, size_type /*n*/) noexcept
    {
    }

    template <bool is_const, class CT>
    inline void xscalar_stepper<is_const, CT>::reset(size_type /*dim*/) noexcept
    {
    }

    template <bool is_const, class CT>
    inline void xscalar_stepper<is_const, CT>::reset_back(size_type /*dim*/) noexcept
    {
    }

    template <bool is_const, class CT>
    inline void xscalar_stepper<is_const, CT>::to_begin() noexcept
    {
    }

    template <bool is_const, class CT>
    inline void xscalar_stepper<is_const, CT>::to_end(layout_type /*l*/) noexcept
    {
    }

    template <bool is_const, class CT>
    template <class T>
    inline auto xscalar_stepper<is_const, CT>::step_simd() -> simd_return_type<T>
    {
        return simd_return_type<T>(p_c->operator()());
    }

    template <bool is_const, class CT>
    inline void xscalar_stepper<is_const, CT>::step_leading()
    {
    }

    /**********************************
     * xdummy_iterator implementation *
     **********************************/

    template <bool is_const, class CT>
    inline xdummy_iterator<is_const, CT>::xdummy_iterator(storage_type* c) noexcept
        : p_c(c)
    {
    }

    template <bool is_const, class CT>
    inline auto xdummy_iterator<is_const, CT>::operator++() noexcept -> self_type&
    {
        return *this;
    }

    template <bool is_const, class CT>
    inline auto xdummy_iterator<is_const, CT>::operator--() noexcept -> self_type&
    {
        return *this;
    }

    template <bool is_const, class CT>
    inline auto xdummy_iterator<is_const, CT>::operator+=(difference_type) noexcept -> self_type&
    {
        return *this;
    }

    template <bool is_const, class CT>
    inline auto xdummy_iterator<is_const, CT>::operator-=(difference_type) noexcept -> self_type&
    {
        return *this;
    }

    template <bool is_const, class CT>
    inline auto xdummy_iterator<is_const, CT>::operator-(const self_type&) const noexcept -> difference_type
    {
        return 0;
    }

    template <bool is_const, class CT>
    inline auto xdummy_iterator<is_const, CT>::operator*() const noexcept -> reference
    {
        return p_c->operator()();
    }

    template <bool is_const, class CT>
    inline bool xdummy_iterator<is_const, CT>::equal(const self_type& rhs) const noexcept
    {
        return p_c == rhs.p_c;
    }

    template <bool is_const, class CT>
    inline bool xdummy_iterator<is_const, CT>::less_than(const self_type& rhs) const noexcept
    {
        return p_c < rhs.p_c;
    }

    template <bool is_const, class CT>
    inline bool
    operator==(const xdummy_iterator<is_const, CT>& lhs, const xdummy_iterator<is_const, CT>& rhs) noexcept
    {
        return lhs.equal(rhs);
    }

    template <bool is_const, class CT>
    inline bool
    operator<(const xdummy_iterator<is_const, CT>& lhs, const xdummy_iterator<is_const, CT>& rhs) noexcept
    {
        return lhs.less_than(rhs);
    }
}

#endif
