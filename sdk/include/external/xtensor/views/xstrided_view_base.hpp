/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_STRIDED_VIEW_BASE_HPP
#define XTENSOR_STRIDED_VIEW_BASE_HPP

#include <type_traits>
#include <variant>

#include <xtl/xsequence.hpp>

#include "../core/xaccessible.hpp"
#include "../core/xstrides.hpp"
#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../utils/xutils.hpp"
#include "../views/xslice.hpp"

namespace xt
{
    namespace detail
    {
        template <class CT, layout_type L>
        class flat_expression_adaptor
        {
        public:

            using xexpression_type = std::decay_t<CT>;
            using shape_type = typename xexpression_type::shape_type;
            using inner_strides_type = get_strides_t<shape_type>;
            using index_type = inner_strides_type;
            using size_type = typename xexpression_type::size_type;
            using value_type = typename xexpression_type::value_type;
            using const_reference = typename xexpression_type::const_reference;
            using reference = std::conditional_t<
                std::is_const<std::remove_reference_t<CT>>::value,
                typename xexpression_type::const_reference,
                typename xexpression_type::reference>;

            using iterator = decltype(std::declval<std::remove_reference_t<CT>>().template begin<L>());
            using const_iterator = decltype(std::declval<std::decay_t<CT>>().template cbegin<L>());
            using reverse_iterator = decltype(std::declval<std::remove_reference_t<CT>>().template rbegin<L>());
            using const_reverse_iterator = decltype(std::declval<std::decay_t<CT>>().template crbegin<L>());

            explicit flat_expression_adaptor(CT* e);

            template <class FST>
            flat_expression_adaptor(CT* e, FST&& strides);

            void update_pointer(CT* ptr) const;

            size_type size() const;
            reference operator[](size_type idx);
            const_reference operator[](size_type idx) const;

            iterator begin();
            iterator end();
            const_iterator begin() const;
            const_iterator end() const;
            const_iterator cbegin() const;
            const_iterator cend() const;

        private:

            static index_type& get_index();

            mutable CT* m_e;
            inner_strides_type m_strides;
            size_type m_size;
        };

        template <class T>
        struct is_flat_expression_adaptor : std::false_type
        {
        };

        template <class CT, layout_type L>
        struct is_flat_expression_adaptor<flat_expression_adaptor<CT, L>> : std::true_type
        {
        };

        template <class E, class ST>
        struct provides_data_interface
            : std::conjunction<has_data_interface<std::decay_t<E>>, std::negation<is_flat_expression_adaptor<ST>>>
        {
        };
    }

    template <class D>
    class xstrided_view_base : public xaccessible<D>
    {
    public:

        using base_type = xaccessible<D>;
        using inner_types = xcontainer_inner_types<D>;
        using xexpression_type = typename inner_types::xexpression_type;
        using undecay_expression = typename inner_types::undecay_expression;
        static constexpr bool is_const = std::is_const<std::remove_reference_t<undecay_expression>>::value;

        using value_type = typename xexpression_type::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = std::
            conditional_t<is_const, typename xexpression_type::const_pointer, typename xexpression_type::pointer>;
        using const_pointer = typename xexpression_type::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using storage_getter = typename inner_types::storage_getter;
        using inner_storage_type = typename inner_types::inner_storage_type;
        using storage_type = std::remove_reference_t<inner_storage_type>;

        using shape_type = typename inner_types::shape_type;
        using strides_type = get_strides_t<shape_type>;
        using backstrides_type = strides_type;

        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;

        using undecay_shape = typename inner_types::undecay_shape;

        using simd_value_type = xt_simd::simd_type<value_type>;
        using bool_load_type = typename xexpression_type::bool_load_type;

        static constexpr layout_type static_layout = inner_types::layout;
        static constexpr bool contiguous_layout = static_layout != layout_type::dynamic
                                                  && xexpression_type::contiguous_layout;

        template <class CTA, class SA>
        xstrided_view_base(CTA&& e, SA&& shape, strides_type&& strides, size_type offset, layout_type layout) noexcept;

        xstrided_view_base(xstrided_view_base&& rhs);

        xstrided_view_base(const xstrided_view_base& rhs);

        const inner_shape_type& shape() const noexcept;
        const inner_strides_type& strides() const noexcept;
        const inner_backstrides_type& backstrides() const noexcept;
        layout_type layout() const noexcept;
        bool is_contiguous() const noexcept;
        using base_type::shape;

        reference operator()();
        const_reference operator()() const;

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        const_reference operator()(Args... args) const;

        template <class... Args>
        reference unchecked(Args... args);

        template <class... Args>
        const_reference unchecked(Args... args) const;

        template <class It>
        reference element(It first, It last);

        template <class It>
        const_reference element(It first, It last) const;

        storage_type& storage() noexcept;
        const storage_type& storage() const noexcept;

        template <class E = xexpression_type, class ST = storage_type>
        std::enable_if_t<detail::provides_data_interface<E, ST>::value, pointer> data() noexcept;
        template <class E = xexpression_type, class ST = storage_type>
        std::enable_if_t<detail::provides_data_interface<E, ST>::value, const_pointer> data() const noexcept;
        size_type data_offset() const noexcept;

        xexpression_type& expression() noexcept;
        const xexpression_type& expression() const noexcept;

        template <class O>
        bool broadcast_shape(O& shape, bool reuse_cache = false) const;

        template <class O>
        bool has_linear_assign(const O& strides) const noexcept;

    protected:

        using offset_type = typename strides_type::value_type;

        template <class... Args>
        offset_type compute_index(Args... args) const;

        template <class... Args>
        offset_type compute_unchecked_index(Args... args) const;

        template <class It>
        offset_type compute_element_index(It first, It last) const;

        void set_offset(size_type offset);

    private:

        undecay_expression m_e;
        inner_storage_type m_storage;
        inner_shape_type m_shape;
        inner_strides_type m_strides;
        inner_backstrides_type m_backstrides;
        size_type m_offset;
        layout_type m_layout;
    };

    /***************************
     * flat_expression_adaptor *
     ***************************/

    namespace detail
    {
        template <class CT>
        struct inner_storage_getter
        {
            using type = decltype(std::declval<CT>().storage());
            using reference = std::add_lvalue_reference_t<CT>;

            template <class E>
            using rebind_t = inner_storage_getter<E>;

            static decltype(auto) get_flat_storage(reference e)
            {
                return e.storage();
            }

            static auto get_offset(reference e)
            {
                return e.data_offset();
            }

            static decltype(auto) get_strides(reference e)
            {
                return e.strides();
            }
        };

        template <class CT, layout_type L>
        struct flat_adaptor_getter
        {
            using type = flat_expression_adaptor<std::remove_reference_t<CT>, L>;
            using reference = std::add_lvalue_reference_t<CT>;

            template <class E>
            using rebind_t = flat_adaptor_getter<E, L>;

            static type get_flat_storage(reference e)
            {
                // moved to addressof because ampersand on xview returns a closure pointer
                return type(std::addressof(e));
            }

            static auto get_offset(reference)
            {
                return typename std::decay_t<CT>::size_type(0);
            }

            static auto get_strides(reference e)
            {
                dynamic_shape<std::ptrdiff_t> strides;
                strides.resize(e.shape().size());
                compute_strides(e.shape(), L, strides);
                return strides;
            }
        };

        template <class CT, layout_type L>
        using flat_storage_getter = std::conditional_t<
            has_data_interface<std::decay_t<CT>>::value,
            inner_storage_getter<CT>,
            flat_adaptor_getter<CT, L>>;

        template <layout_type L, class E>
        inline auto get_offset(E& e)
        {
            return flat_storage_getter<E, L>::get_offset(e);
        }

        template <layout_type L, class E>
        inline decltype(auto) get_strides(E& e)
        {
            return flat_storage_getter<E, L>::get_strides(e);
        }
    }

    /*************************************
     * xstrided_view_base implementation *
     *************************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xstrided_view_base
     *
     * @param e the underlying xexpression for this view
     * @param shape the shape of the view
     * @param strides the strides of the view
     * @param offset the offset of the first element in the underlying container
     * @param layout the layout of the view
     */
    template <class D>
    template <class CTA, class SA>
    inline xstrided_view_base<D>::xstrided_view_base(
        CTA&& e,
        SA&& shape,
        strides_type&& strides,
        size_type offset,
        layout_type layout
    ) noexcept
        : m_e(std::forward<CTA>(e))
        ,
        // m_storage(detail::get_flat_storage<undecay_expression>(m_e)),
        m_storage(storage_getter::get_flat_storage(m_e))
        , m_shape(std::forward<SA>(shape))
        , m_strides(std::move(strides))
        , m_offset(offset)
        , m_layout(layout)
    {
        m_backstrides = xtl::make_sequence<backstrides_type>(m_shape.size(), 0);
        adapt_strides(m_shape, m_strides, m_backstrides);
    }

    namespace detail
    {
        template <class T, class S>
        auto& copy_move_storage(T& expr, const S& /*storage*/)
        {
            return expr.storage();
        }

        template <class T, class E, layout_type L>
        auto copy_move_storage(T& expr, const detail::flat_expression_adaptor<E, L>& storage)
        {
            detail::flat_expression_adaptor<E, L> new_storage = storage;  // copy storage
            new_storage.update_pointer(std::addressof(expr));
            return new_storage;
        }
    }

    template <class D>
    inline xstrided_view_base<D>::xstrided_view_base(xstrided_view_base&& rhs)
        : base_type(std::move(rhs))
        , m_e(std::forward<undecay_expression>(rhs.m_e))
        , m_storage(detail::copy_move_storage(m_e, rhs.m_storage))
        , m_shape(std::move(rhs.m_shape))
        , m_strides(std::move(rhs.m_strides))
        , m_backstrides(std::move(rhs.m_backstrides))
        , m_offset(std::move(rhs.m_offset))
        , m_layout(std::move(rhs.m_layout))
    {
    }

    template <class D>
    inline xstrided_view_base<D>::xstrided_view_base(const xstrided_view_base& rhs)
        : base_type(rhs)
        , m_e(rhs.m_e)
        , m_storage(detail::copy_move_storage(m_e, rhs.m_storage))
        , m_shape(rhs.m_shape)
        , m_strides(rhs.m_strides)
        , m_backstrides(rhs.m_backstrides)
        , m_offset(rhs.m_offset)
        , m_layout(rhs.m_layout)
    {
    }

    //@}

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the shape of the xtrided_view_base.
     */
    template <class D>
    inline auto xstrided_view_base<D>::shape() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    /**
     * Returns the strides of the xtrided_view_base.
     */
    template <class D>
    inline auto xstrided_view_base<D>::strides() const noexcept -> const inner_strides_type&
    {
        return m_strides;
    }

    /**
     * Returns the backstrides of the xtrided_view_base.
     */
    template <class D>
    inline auto xstrided_view_base<D>::backstrides() const noexcept -> const inner_backstrides_type&
    {
        return m_backstrides;
    }

    /**
     * Returns the layout of the xtrided_view_base.
     */
    template <class D>
    inline auto xstrided_view_base<D>::layout() const noexcept -> layout_type
    {
        return m_layout;
    }

    template <class D>
    inline bool xstrided_view_base<D>::is_contiguous() const noexcept
    {
        return m_layout != layout_type::dynamic && m_e.is_contiguous();
    }

    //@}

    /**
     * @name Data
     */
    //@{
    template <class D>
    inline auto xstrided_view_base<D>::operator()() -> reference
    {
        return m_storage[static_cast<size_type>(m_offset)];
    }

    template <class D>
    inline auto xstrided_view_base<D>::operator()() const -> const_reference
    {
        return m_storage[static_cast<size_type>(m_offset)];
    }

    /**
     * Returns a reference to the element at the specified position in the view.
     * @param args a list of indices specifying the position in the view. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the view.
     */
    template <class D>
    template <class... Args>
    inline auto xstrided_view_base<D>::operator()(Args... args) -> reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        offset_type index = compute_index(args...);
        return m_storage[static_cast<size_type>(index)];
    }

    /**
     * Returns a constant reference to the element at the specified position in the view.
     * @param args a list of indices specifying the position in the view. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the view.
     */
    template <class D>
    template <class... Args>
    inline auto xstrided_view_base<D>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        offset_type index = compute_index(args...);
        return m_storage[static_cast<size_type>(index)];
    }

    /**
     * Returns a reference to the element at the specified position in the view.
     * @param args a list of indices specifying the position in the view. Indices
     * must be unsigned integers, the number of indices must be equal to the number of
     * dimensions of the view, else the behavior is undefined.
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
    inline auto xstrided_view_base<D>::unchecked(Args... args) -> reference
    {
        offset_type index = compute_unchecked_index(args...);
        return m_storage[static_cast<size_type>(index)];
    }

    /**
     * Returns a constant reference to the element at the specified position in the view.
     * @param args a list of indices specifying the position in the view. Indices
     * must be unsigned integers, the number of indices must be equal to the number of
     * dimensions of the view, else the behavior is undefined.
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
    inline auto xstrided_view_base<D>::unchecked(Args... args) const -> const_reference
    {
        offset_type index = compute_unchecked_index(args...);
        return m_storage[static_cast<size_type>(index)];
    }

    /**
     * Returns a reference to the element at the specified position in the view.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater than the the number
     * of dimensions of the view..
     */
    template <class D>
    template <class It>
    inline auto xstrided_view_base<D>::element(It first, It last) -> reference
    {
        XTENSOR_TRY(check_element_index(shape(), first, last));
        return m_storage[static_cast<size_type>(compute_element_index(first, last))];
    }

    /**
     * Returns a constant reference to the element at the specified position in the view.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater than the the number
     * of dimensions of the view..
     */
    template <class D>
    template <class It>
    inline auto xstrided_view_base<D>::element(It first, It last) const -> const_reference
    {
        XTENSOR_TRY(check_element_index(shape(), first, last));
        return m_storage[static_cast<size_type>(compute_element_index(first, last))];
    }

    /**
     * Returns a reference to the buffer containing the elements of the view.
     */
    template <class D>
    inline auto xstrided_view_base<D>::storage() noexcept -> storage_type&
    {
        return m_storage;
    }

    /**
     * Returns a constant reference to the buffer containing the elements of the view.
     */
    template <class D>
    inline auto xstrided_view_base<D>::storage() const noexcept -> const storage_type&
    {
        return m_storage;
    }

    /**
     * Returns a pointer to the underlying array serving as element storage.
     * The first element of the view is at data() + data_offset().
     */
    template <class D>
    template <class E, class ST>
    inline auto xstrided_view_base<D>::data() noexcept
        -> std::enable_if_t<detail::provides_data_interface<E, ST>::value, pointer>
    {
        return m_e.data();
    }

    /**
     * Returns a constant pointer to the underlying array serving as element storage.
     * The first element of the view is at data() + data_offset().
     */
    template <class D>
    template <class E, class ST>
    inline auto xstrided_view_base<D>::data() const noexcept
        -> std::enable_if_t<detail::provides_data_interface<E, ST>::value, const_pointer>
    {
        return m_e.data();
    }

    /**
     * Returns the offset to the first element in the view.
     */
    template <class D>
    inline auto xstrided_view_base<D>::data_offset() const noexcept -> size_type
    {
        return m_offset;
    }

    /**
     * Returns a reference to the underlying expression of the view.
     */
    template <class D>
    inline auto xstrided_view_base<D>::expression() noexcept -> xexpression_type&
    {
        return m_e;
    }

    /**
     * Returns a constant reference to the underlying expression of the view.
     */
    template <class D>
    inline auto xstrided_view_base<D>::expression() const noexcept -> const xexpression_type&
    {
        return m_e;
    }

    //@}

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the view to the specified parameter.
     * @param shape the result shape
     * @param reuse_cache parameter for internal optimization
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class D>
    template <class O>
    inline bool xstrided_view_base<D>::broadcast_shape(O& shape, bool) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Checks whether the xstrided_view_base can be linearly assigned to an expression
     * with the specified strides.
     * @return a boolean indicating whether a linear assign is possible
     */
    template <class D>
    template <class O>
    inline bool xstrided_view_base<D>::has_linear_assign(const O& str) const noexcept
    {
        return has_data_interface<xexpression_type>::value && str.size() == strides().size()
               && std::equal(str.cbegin(), str.cend(), strides().begin());
    }

    //@}

    template <class D>
    template <class... Args>
    inline auto xstrided_view_base<D>::compute_index(Args... args) const -> offset_type
    {
        return static_cast<offset_type>(m_offset)
               + xt::data_offset<offset_type>(strides(), static_cast<offset_type>(args)...);
    }

    template <class D>
    template <class... Args>
    inline auto xstrided_view_base<D>::compute_unchecked_index(Args... args) const -> offset_type
    {
        return static_cast<offset_type>(m_offset)
               + xt::unchecked_data_offset<offset_type>(strides(), static_cast<offset_type>(args)...);
    }

    template <class D>
    template <class It>
    inline auto xstrided_view_base<D>::compute_element_index(It first, It last) const -> offset_type
    {
        return static_cast<offset_type>(m_offset) + xt::element_offset<offset_type>(strides(), first, last);
    }

    template <class D>
    void xstrided_view_base<D>::set_offset(size_type offset)
    {
        m_offset = offset;
    }

    /******************************************
     * flat_expression_adaptor implementation *
     ******************************************/

    namespace detail
    {
        template <class CT, layout_type L>
        inline flat_expression_adaptor<CT, L>::flat_expression_adaptor(CT* e)
            : m_e(e)
        {
            resize_container(get_index(), m_e->dimension());
            resize_container(m_strides, m_e->dimension());
            m_size = compute_size(m_e->shape());
            compute_strides(m_e->shape(), L, m_strides);
        }

        template <class CT, layout_type L>
        template <class FST>
        inline flat_expression_adaptor<CT, L>::flat_expression_adaptor(CT* e, FST&& strides)
            : m_e(e)
            , m_strides(xtl::forward_sequence<inner_strides_type, FST>(strides))
        {
            resize_container(get_index(), m_e->dimension());
            m_size = m_e->size();
        }

        template <class CT, layout_type L>
        inline void flat_expression_adaptor<CT, L>::update_pointer(CT* ptr) const
        {
            m_e = ptr;
        }

        template <class CT, layout_type L>
        inline auto flat_expression_adaptor<CT, L>::size() const -> size_type
        {
            return m_size;
        }

        template <class CT, layout_type L>
        inline auto flat_expression_adaptor<CT, L>::operator[](size_type idx) -> reference
        {
            auto i = static_cast<typename index_type::value_type>(idx);
            get_index() = detail::unravel_noexcept(i, m_strides, L);
            return m_e->element(get_index().cbegin(), get_index().cend());
        }

        template <class CT, layout_type L>
        inline auto flat_expression_adaptor<CT, L>::operator[](size_type idx) const -> const_reference
        {
            auto i = static_cast<typename index_type::value_type>(idx);
            get_index() = detail::unravel_noexcept(i, m_strides, L);
            return m_e->element(get_index().cbegin(), get_index().cend());
        }

        template <class CT, layout_type L>
        inline auto flat_expression_adaptor<CT, L>::begin() -> iterator
        {
            return m_e->template begin<L>();
        }

        template <class CT, layout_type L>
        inline auto flat_expression_adaptor<CT, L>::end() -> iterator
        {
            return m_e->template end<L>();
        }

        template <class CT, layout_type L>
        inline auto flat_expression_adaptor<CT, L>::begin() const -> const_iterator
        {
            return m_e->template cbegin<L>();
        }

        template <class CT, layout_type L>
        inline auto flat_expression_adaptor<CT, L>::end() const -> const_iterator
        {
            return m_e->template cend<L>();
        }

        template <class CT, layout_type L>
        inline auto flat_expression_adaptor<CT, L>::cbegin() const -> const_iterator
        {
            return m_e->template cbegin<L>();
        }

        template <class CT, layout_type L>
        inline auto flat_expression_adaptor<CT, L>::cend() const -> const_iterator
        {
            return m_e->template cend<L>();
        }

        template <class CT, layout_type L>
        inline auto flat_expression_adaptor<CT, L>::get_index() -> index_type&
        {
            thread_local static index_type index;
            return index;
        }
    }

    /**********************************
     * Builder helpers implementation *
     **********************************/

    namespace detail
    {
        template <class S>
        struct slice_getter_impl
        {
            const S& m_shape;
            mutable std::size_t idx;
            using array_type = std::array<std::ptrdiff_t, 3>;

            explicit slice_getter_impl(const S& shape)
                : m_shape(shape)
                , idx(0)
            {
            }

            template <class T>
            array_type operator()(const T& /*t*/) const
            {
                return array_type{{0, 0, 0}};
            }

            template <class A, class B, class C>
            array_type operator()(const xrange_adaptor<A, B, C>& range) const
            {
                auto sl = range.get(static_cast<std::size_t>(m_shape[idx]));
                return array_type({sl(0), sl.size(), sl.step_size()});
            }

            template <class T>
            array_type operator()(const xrange<T>& range) const
            {
                return array_type({range(T(0)), range.size(), T(1)});
            }

            template <class T>
            array_type operator()(const xstepped_range<T>& range) const
            {
                return array_type({range(T(0)), range.size(), range.step_size(T(0))});
            }
        };

        template <class adj_strides_policy>
        struct strided_view_args : adj_strides_policy
        {
            using base_type = adj_strides_policy;

            template <class S, class ST, class V>
            void
            fill_args(const S& shape, ST&& old_strides, std::size_t base_offset, layout_type layout, const V& slices)
            {
                // Compute dimension
                std::size_t dimension = shape.size(), n_newaxis = 0, n_add_all = 0;
                std::ptrdiff_t dimension_check = static_cast<std::ptrdiff_t>(shape.size());

                bool has_ellipsis = false;
                for (const auto& el : slices)
                {
                    if (std::get_if<xt::xnewaxis_tag>(&el) != nullptr)
                    {
                        ++dimension;
                        ++n_newaxis;
                    }
                    else if (std::get_if<std::ptrdiff_t>(&el) != nullptr)
                    {
                        --dimension;
                        --dimension_check;
                    }
                    else if (std::get_if<xt::xellipsis_tag>(&el) != nullptr)
                    {
                        if (has_ellipsis == true)
                        {
                            XTENSOR_THROW(std::runtime_error, "Ellipsis can only appear once.");
                        }
                        has_ellipsis = true;
                    }
                    else
                    {
                        --dimension_check;
                    }
                }

                if (dimension_check < 0)
                {
                    XTENSOR_THROW(std::runtime_error, "Too many slices for view.");
                }

                if (has_ellipsis)
                {
                    // replace ellipsis with N * xt::all
                    // remove -1 because of the ellipsis slize itself
                    n_add_all = shape.size() - (slices.size() - 1 - n_newaxis);
                }

                // Compute strided view
                new_offset = base_offset;
                new_shape.resize(dimension);
                new_strides.resize(dimension);
                base_type::resize(dimension);

                auto old_shape = shape;
                using old_strides_value_type = typename std::decay_t<ST>::value_type;

                std::ptrdiff_t axis_skip = 0;
                std::size_t idx = 0, i = 0, i_ax = 0;

                auto slice_getter = detail::slice_getter_impl<S>(shape);

                for (; i < slices.size(); ++i)
                {
                    i_ax = static_cast<std::size_t>(static_cast<std::ptrdiff_t>(i) - axis_skip);
                    auto ptr = std::get_if<std::ptrdiff_t>(&slices[i]);
                    if (ptr != nullptr)
                    {
                        auto slice0 = static_cast<old_strides_value_type>(*ptr);
                        new_offset += static_cast<std::size_t>(slice0 * old_strides[i_ax]);
                    }
                    else if (std::get_if<xt::xnewaxis_tag>(&slices[i]) != nullptr)
                    {
                        new_shape[idx] = 1;
                        base_type::set_fake_slice(idx);
                        ++axis_skip, ++idx;
                    }
                    else if (std::get_if<xt::xellipsis_tag>(&slices[i]) != nullptr)
                    {
                        for (std::size_t j = 0; j < n_add_all; ++j)
                        {
                            new_shape[idx] = old_shape[i_ax];
                            new_strides[idx] = old_strides[i_ax];
                            base_type::set_fake_slice(idx);
                            ++idx, ++i_ax;
                        }
                        axis_skip = axis_skip - static_cast<std::ptrdiff_t>(n_add_all) + 1;
                    }
                    else if (std::get_if<xt::xall_tag>(&slices[i]) != nullptr)
                    {
                        new_shape[idx] = old_shape[i_ax];
                        new_strides[idx] = old_strides[i_ax];
                        base_type::set_fake_slice(idx);
                        ++idx;
                    }
                    else if (base_type::fill_args(slices, i, idx, old_shape[i_ax], old_strides[i_ax], new_shape, new_strides))
                    {
                        ++idx;
                    }
                    else
                    {
                        slice_getter.idx = i_ax;
                        auto info = std::visit(slice_getter, slices[i]);
                        new_offset += static_cast<std::size_t>(info[0] * old_strides[i_ax]);
                        new_shape[idx] = static_cast<std::size_t>(info[1]);
                        new_strides[idx] = info[2] * old_strides[i_ax];
                        base_type::set_fake_slice(idx);
                        ++idx;
                    }
                }

                i_ax = static_cast<std::size_t>(static_cast<std::ptrdiff_t>(i) - axis_skip);
                for (; i_ax < old_shape.size(); ++i_ax, ++idx)
                {
                    new_shape[idx] = old_shape[i_ax];
                    new_strides[idx] = old_strides[i_ax];
                    base_type::set_fake_slice(idx);
                }

                new_layout = do_strides_match(new_shape, new_strides, layout, true) ? layout
                                                                                    : layout_type::dynamic;
            }

            using shape_type = dynamic_shape<std::size_t>;
            shape_type new_shape;
            using strides_type = dynamic_shape<std::ptrdiff_t>;
            strides_type new_strides;
            std::size_t new_offset;
            layout_type new_layout;
        };
    }
}

#endif
