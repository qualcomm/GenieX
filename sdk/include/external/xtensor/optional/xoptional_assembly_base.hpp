/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XOPTIONAL_ASSEMBLY_BASE_HPP
#define XOPTIONAL_ASSEMBLY_BASE_HPP

#include "../core/xiterable.hpp"
#include "../core/xtensor_forward.hpp"
#include "../optional/xoptional_assembly_storage.hpp"

namespace xt
{
    template <class D, bool is_const>
    class xoptional_assembly_stepper;

    /***************************
     * xoptional_assembly_base *
     ***************************/

    /**
     * @class xoptional_assembly_base
     * @brief Base class for dense multidimensional optional assemblies.
     *
     * The xoptional_assembly_base class defines the interface for dense multidimensional
     * optional assembly classes. Optional assembly classes hold optional values and are
     * optimized for tensor operations. xoptional_assembly_base does not embed any data
     * container, this responsibility is delegated to the inheriting classes.
     *
     * @tparam D The derived type, i.e. the inheriting class for which xoptional_assembly_base
     *           provides the interface.
     */
    template <class D>
    class xoptional_assembly_base : private xiterable<D>
    {
    public:

        using self_type = xoptional_assembly_base<D>;
        using derived_type = D;
        using inner_types = xcontainer_inner_types<D>;

        using raw_value_expression = typename inner_types::raw_value_expression;
        using base_value_type = typename raw_value_expression::value_type;
        using base_reference = typename raw_value_expression::reference;
        using base_const_reference = typename raw_value_expression::const_reference;

        using raw_flag_expression = typename inner_types::raw_flag_expression;
        using flag_type = typename raw_flag_expression::value_type;
        using flag_reference = typename raw_flag_expression::reference;
        using flag_const_reference = typename raw_flag_expression::const_reference;

        using storage_type = typename inner_types::storage_type;

        using value_type = typename storage_type::value_type;
        using reference = typename storage_type::reference;
        using const_reference = typename storage_type::const_reference;
        using pointer = typename storage_type::pointer;
        using const_pointer = typename storage_type::const_pointer;
        using size_type = typename raw_value_expression::size_type;
        using difference_type = typename raw_value_expression::difference_type;
        using simd_value_type = xt_simd::simd_type<value_type>;
        using bool_load_type = xt::bool_load_type<value_type>;

        using shape_type = typename raw_value_expression::shape_type;
        using strides_type = typename raw_value_expression::strides_type;
        using backstrides_type = typename raw_value_expression::backstrides_type;

        using inner_shape_type = typename raw_value_expression::inner_shape_type;
        using inner_strides_type = typename raw_value_expression::inner_strides_type;
        using inner_backstrides_type = typename raw_value_expression::inner_backstrides_type;

        using iterable_base = xiterable<D>;
        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        static constexpr layout_type static_layout = raw_value_expression::static_layout;
        static constexpr bool contiguous_layout = raw_value_expression::contiguous_layout;

        using expression_tag = xoptional_expression_tag;
        using value_expression = raw_value_expression&;
        using flag_expression = raw_flag_expression&;
        using const_value_expression = const raw_value_expression&;
        using const_flag_expression = const raw_flag_expression&;

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

        using linear_iterator = typename storage_type::iterator;
        using const_linear_iterator = typename storage_type::const_iterator;
        using reverse_linear_iterator = typename storage_type::reverse_iterator;
        using const_reverse_linear_iterator = typename storage_type::const_reverse_iterator;

        using iterator = typename iterable_base::iterator;
        using const_iterator = typename iterable_base::const_iterator;
        using reverse_iterator = typename iterable_base::reverse_iterator;
        using const_reverse_iterator = typename iterable_base::const_reverse_iterator;

        size_type size() const noexcept;
        constexpr size_type dimension() const noexcept;
        const inner_shape_type& shape() const noexcept;
        size_type shape(size_type index) const;
        const inner_strides_type& strides() const noexcept;
        const inner_backstrides_type& backstrides() const noexcept;

        template <class S = shape_type>
        void resize(const S& shape, bool force = false);
        template <class S = shape_type>
        void resize(const S& shape, layout_type l);
        template <class S = shape_type>
        void resize(const S& shape, const strides_type& strides);

        template <class S = shape_type>
        auto& reshape(const S& shape, layout_type layout = static_layout) &;

        template <class T>
        auto& reshape(std::initializer_list<T> shape, layout_type layout = static_layout) &;

        layout_type layout() const noexcept;
        bool is_contiguous() const noexcept;

        template <class T>
        void fill(const T& value);

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        const_reference operator()(Args... args) const;

        template <class... Args>
        reference at(Args... args);

        template <class... Args>
        const_reference at(Args... args) const;

        template <class... Args>
        reference unchecked(Args... args);

        template <class... Args>
        const_reference unchecked(Args... args) const;

        template <class S>
        disable_integral_t<S, reference> operator[](const S& index);
        template <class I>
        reference operator[](std::initializer_list<I> index);
        reference operator[](size_type i);

        template <class S>
        disable_integral_t<S, const_reference> operator[](const S& index) const;
        template <class I>
        const_reference operator[](std::initializer_list<I> index) const;
        const_reference operator[](size_type i) const;

        template <class... Args>
        reference periodic(Args... args);

        template <class... Args>
        const_reference periodic(Args... args) const;

        reference front();
        const_reference front() const;

        reference back();
        const_reference back() const;

        reference flat(size_type args);
        const_reference flat(size_type args) const;

        template <class It>
        reference element(It first, It last);
        template <class It>
        const_reference element(It first, It last) const;

        template <class... Args>
        bool in_bounds(Args... args) const;

        storage_type& storage() noexcept;
        const storage_type& storage() const noexcept;

        value_type* data() noexcept;
        const value_type* data() const noexcept;
        const size_type data_offset() const noexcept;

        template <class S>
        bool broadcast_shape(S& shape, bool reuse_cache = false) const;

        template <class S>
        bool has_linear_assign(const S& strides) const noexcept;

        using iterable_base::begin;
        using iterable_base::cbegin;
        using iterable_base::cend;
        using iterable_base::crbegin;
        using iterable_base::crend;
        using iterable_base::end;
        using iterable_base::rbegin;
        using iterable_base::rend;

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

        value_expression value() noexcept;
        const_value_expression value() const noexcept;

        flag_expression has_value() noexcept;
        const_flag_expression has_value() const noexcept;

    protected:

        xoptional_assembly_base() = default;
        ~xoptional_assembly_base() = default;

        xoptional_assembly_base(const xoptional_assembly_base&) = default;
        xoptional_assembly_base& operator=(const xoptional_assembly_base&) = default;

        xoptional_assembly_base(xoptional_assembly_base&&) = default;
        xoptional_assembly_base& operator=(xoptional_assembly_base&&) = default;

    private:

        derived_type& derived_cast() noexcept;
        const derived_type& derived_cast() const noexcept;

        friend class xiterable<D>;
        friend class xconst_iterable<D>;
    };

    /******************************
     * xoptional_assembly_stepper *
     ******************************/

    template <class D, bool is_const>
    class xoptional_assembly_stepper
    {
    public:

        using self_type = xoptional_assembly_stepper<D, is_const>;
        using assembly_type = typename D::assembly_type;
        using value_type = typename assembly_type::value_type;
        using reference = std::
            conditional_t<is_const, typename assembly_type::const_reference, typename assembly_type::reference>;
        using pointer = std::conditional_t<is_const, typename assembly_type::const_pointer, typename assembly_type::pointer>;
        using size_type = typename assembly_type::size_type;
        using difference_type = typename assembly_type::difference_type;
        using raw_value_expression = typename assembly_type::raw_value_expression;
        using raw_flag_expression = typename assembly_type::raw_flag_expression;
        using value_stepper = std::
            conditional_t<is_const, typename raw_value_expression::const_stepper, typename raw_value_expression::stepper>;
        using flag_stepper = std::
            conditional_t<is_const, typename raw_flag_expression::const_stepper, typename raw_flag_expression::stepper>;

        xoptional_assembly_stepper(value_stepper vs, flag_stepper fs) noexcept;


        void step(size_type dim);
        void step_back(size_type dim);
        void step(size_type dim, size_type n);
        void step_back(size_type dim, size_type n);
        void reset(size_type dim);
        void reset_back(size_type dim);

        void to_begin();
        void to_end(layout_type l);

        reference operator*() const;

    private:

        value_stepper m_vs;
        flag_stepper m_fs;
    };

    /******************************************
     * xoptional_assembly_base implementation *
     ******************************************/

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the number of element in the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::size() const noexcept -> size_type
    {
        return value().size();
    }

    /**
     * Returns the number of dimensions of the optional assembly.
     */
    template <class D>
    inline constexpr auto xoptional_assembly_base<D>::dimension() const noexcept -> size_type
    {
        return value().dimension();
    }

    /**
     * Returns the shape of the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::shape() const noexcept -> const inner_shape_type&
    {
        return value().shape();
    }

    /**
     * Returns the i-th dimension of the expression.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::shape(size_type i) const -> size_type
    {
        return value().shape(i);
    }

    /**
     * Returns the strides of the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::strides() const noexcept -> const inner_strides_type&
    {
        return value().strides();
    }

    /**
     * Returns the backstrides of the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::backstrides() const noexcept -> const inner_backstrides_type&
    {
        return value().backstrides();
    }

    //@}

    /**
     * Resizes the optional assembly.
     * @param shape the new shape
     * @param force force reshaping, even if the shape stays the same (default: false)
     */
    template <class D>
    template <class S>
    inline void xoptional_assembly_base<D>::resize(const S& shape, bool force)
    {
        value().resize(shape, force);
        has_value().resize(shape, force);
    }

    /**
     * Resizes the optional assembly.
     * @param shape the new shape
     * @param l the new layout_type
     */
    template <class D>
    template <class S>
    inline void xoptional_assembly_base<D>::resize(const S& shape, layout_type l)
    {
        value().resize(shape, l);
        has_value().resize(shape, l);
    }

    /**
     * Resizes the optional assembly.
     * @param shape the new shape
     * @param strides the new strides
     */
    template <class D>
    template <class S>
    inline void xoptional_assembly_base<D>::resize(const S& shape, const strides_type& strides)
    {
        value().resize(shape, strides);
        has_value().resize(shape, strides);
    }

    /**
     * Reshapes the optional assembly.
     * @param shape the new shape
     * @param layout the new layout
     */
    template <class D>
    template <class S>
    inline auto& xoptional_assembly_base<D>::reshape(const S& shape, layout_type layout) &
    {
        value().reshape(shape, layout);
        has_value().reshape(shape, layout);
        return *this;
    }

    template <class D>
    template <class T>
    inline auto& xoptional_assembly_base<D>::reshape(std::initializer_list<T> shape, layout_type layout) &
    {
        value().reshape(shape, layout);
        has_value().reshape(shape, layout);
        return *this;
    }

    /**
     * Return the layout_type of the container
     * @return layout_type of the container
     */
    template <class D>
    inline layout_type xoptional_assembly_base<D>::layout() const noexcept
    {
        return value().layout();
    }

    template <class D>
    inline bool xoptional_assembly_base<D>::is_contiguous() const noexcept
    {
        return value().is_contiguous();
    }

    /**
     * Fills the data with the given value.
     * @param value the value to fill the data with.
     */
    template <class D>
    template <class T>
    inline void xoptional_assembly_base<D>::fill(const T& value)
    {
        std::fill(this->linear_begin(), this->linear_end(), value);
    }

    /**
     * @name Data
     */
    //@{
    /**
     * Returns a reference to the element at the specified position in the optional assembly.
     * @param args a list of indices specifying the position in the optional assembly. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the optional assembly.
     */
    template <class D>
    template <class... Args>
    inline auto xoptional_assembly_base<D>::operator()(Args... args) -> reference
    {
        return reference(value()(args...), has_value()(args...));
    }

    /**
     * Returns a constant reference to the element at the specified position in the optional assembly.
     * @param args a list of indices specifying the position in the optional assembly. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the optional assembly.
     */
    template <class D>
    template <class... Args>
    inline auto xoptional_assembly_base<D>::operator()(Args... args) const -> const_reference
    {
        return const_reference(value()(args...), has_value()(args...));
    }

    /**
     * Returns a reference to the element at the specified position in the optional assembly,
     * after dimension and bounds checking.
     * @param args a list of indices specifying the position in the optional assembly. Indices
     * must be unsigned integers, the number of indices should be equal to the number of dimensions
     * of the optional assembly.
     * @exception std::out_of_range if the number of argument is greater than the number of dimensions
     * or if indices are out of bounds.
     */
    template <class D>
    template <class... Args>
    inline auto xoptional_assembly_base<D>::at(Args... args) -> reference
    {
        return reference(value().at(args...), has_value().at(args...));
    }

    /**
     * Returns a constant reference to the element at the specified position in the optional assembly,
     * after dimension and bounds checking.
     * @param args a list of indices specifying the position in the optional assembly. Indices
     * must be unsigned integers, the number of indices should be equal to the number of dimensions
     * of the optional assembly.
     * @exception std::out_of_range if the number of argument is greater than the number of dimensions
     * or if indices are out of bounds.
     */
    template <class D>
    template <class... Args>
    inline auto xoptional_assembly_base<D>::at(Args... args) const -> const_reference
    {
        return const_reference(value().at(args...), has_value().at(args...));
    }

    /**
     * Returns a reference to the element at the specified position in the  optional assembly.
     * @param args a list of indices specifying the position in the  optional assembly. Indices
     * must be unsigned integers, the number of indices must be equal to the number of
     * dimensions of the  optional assembly, else the behavior is undefined.
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
    inline auto xoptional_assembly_base<D>::unchecked(Args... args) -> reference
    {
        return reference(value().unchecked(args...), has_value().unchecked(args...));
    }

    /**
     * Returns a constant reference to the element at the specified position in the  optional assembly.
     * @param args a list of indices specifying the position in the  optional assembly. Indices
     * must be unsigned integers, the number of indices must be equal to the number of
     * dimensions of the  optional assembly, else the behavior is undefined.
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
    inline auto xoptional_assembly_base<D>::unchecked(Args... args) const -> const_reference
    {
        return const_reference(value().unchecked(args...), has_value().unchecked(args...));
    }

    /**
     * Returns a reference to the element at the specified position in the optional assembly.
     * @param index a sequence of indices specifying the position in the optional assembly. Indices
     * must be unsigned integers, the number of indices in the list should be equal or greater
     * than the number of dimensions of the optional assembly.
     */
    template <class D>
    template <class S>
    inline auto xoptional_assembly_base<D>::operator[](const S& index) -> disable_integral_t<S, reference>
    {
        return reference(value()[index], has_value()[index]);
    }

    template <class D>
    template <class I>
    inline auto xoptional_assembly_base<D>::operator[](std::initializer_list<I> index) -> reference
    {
        return reference(value()[index], has_value()[index]);
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::operator[](size_type i) -> reference
    {
        return reference(value()[i], has_value()[i]);
    }

    /**
     * Returns a constant reference to the element at the specified position in the optional assembly.
     * @param index a sequence of indices specifying the position in the optional assembly. Indices
     * must be unsigned integers, the number of indices in the list should be equal or greater
     * than the number of dimensions of the optional assembly.
     */
    template <class D>
    template <class S>
    inline auto xoptional_assembly_base<D>::operator[](const S& index) const
        -> disable_integral_t<S, const_reference>
    {
        return const_reference(value()[index], has_value()[index]);
    }

    template <class D>
    template <class I>
    inline auto xoptional_assembly_base<D>::operator[](std::initializer_list<I> index) const -> const_reference
    {
        return const_reference(value()[index], has_value()[index]);
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::operator[](size_type i) const -> const_reference
    {
        return const_reference(value()[i], has_value()[i]);
    }

    /**
     * Returns a reference to the element at the specified position in the optional assembly,
     * after applying periodicity to the indices (negative and 'overflowing' indices are changed).
     * @param args a list of indices specifying the position in the optional assembly. Indices
     * must be unsigned integers, the number of indices should be equal to the number of dimensions
     * of the optional assembly.
     */
    template <class D>
    template <class... Args>
    inline auto xoptional_assembly_base<D>::periodic(Args... args) -> reference
    {
        return reference(value().periodic(args...), has_value().periodic(args...));
    }

    /**
     * Returns a constant reference to the element at the specified position in the optional assembly,
     * after applying periodicity to the indices (negative and 'overflowing' indices are changed).
     * @param args a list of indices specifying the position in the optional assembly. Indices
     * must be unsigned integers, the number of indices should be equal to the number of dimensions
     * of the optional assembly.
     */
    template <class D>
    template <class... Args>
    inline auto xoptional_assembly_base<D>::periodic(Args... args) const -> const_reference
    {
        return const_reference(value().periodic(args...), has_value().periodic(args...));
    }

    /**
     * Returns a reference to the first element of the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::front() -> reference
    {
        return reference(value().front(), has_value().front());
    }

    /**
     * Returns a constant reference to the first element of the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::front() const -> const_reference
    {
        return const_reference(value().front(), has_value().front());
    }

    /**
     * Returns a reference to the last element of the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::back() -> reference
    {
        return reference(value().back(), has_value().back());
    }

    /**
     * Returns a constant reference to the last element of the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::back() const -> const_reference
    {
        return const_reference(value().back(), has_value().back());
    }

    /**
     * Returns a reference to the element at the specified position
     * of the underlying storage in the optional assembly.
     * @param index index to underlying flat storage.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::flat(size_type i) -> reference
    {
        return reference(value().flat(i), has_value().flat(i));
    }

    /**
     * Returns a constant reference to the element at the specified position
     * of the underlying storage in the optional assembly.
     * @param index index to underlying flat storage.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::flat(size_type i) const -> const_reference
    {
        return const_reference(value().flat(i), has_value().flat(i));
    }

    /**
     * Returns a reference to the element at the specified position in the optional assembly.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the optional assembly.
     */
    template <class D>
    template <class It>
    inline auto xoptional_assembly_base<D>::element(It first, It last) -> reference
    {
        return reference(value().element(first, last), has_value().element(first, last));
    }

    /**
     * Returns a constant reference to the element at the specified position in the optional assembly.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the optional assembly.
     */
    template <class D>
    template <class It>
    inline auto xoptional_assembly_base<D>::element(It first, It last) const -> const_reference
    {
        return const_reference(value().element(first, last), has_value().element(first, last));
    }

    /**
     * Returns ``true`` only if the the specified position is a valid entry in the expression.
     * @param args a list of indices specifying the position in the expression.
     * @return bool
     */
    template <class D>
    template <class... Args>
    inline bool xoptional_assembly_base<D>::in_bounds(Args... args) const
    {
        return value().in_bounds(args...) && has_value().in_bounds(args...);
    }

    //@}

    template <class D>
    inline auto xoptional_assembly_base<D>::storage() noexcept -> storage_type&
    {
        return derived_cast().storage_impl();
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::storage() const noexcept -> const storage_type&
    {
        return derived_cast().storage_impl();
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::data() noexcept -> value_type*
    {
        return storage().data();
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::data() const noexcept -> const value_type*
    {
        return storage().data();
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::data_offset() const noexcept -> const size_type
    {
        return size_type(0);
    }

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the optional assembly to the specified parameter.
     * @param shape the result shape
     * @param reuse_cache parameter for internal optimization
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class D>
    template <class S>
    inline bool xoptional_assembly_base<D>::broadcast_shape(S& shape, bool reuse_cache) const
    {
        bool res = value().broadcast_shape(shape, reuse_cache);
        return res && has_value().broadcast_shape(shape, reuse_cache);
    }

    /**
     * Checks whether the xoptional_assembly_base can be linearly assigned to an expression
     * with the specified strides.
     * @return a boolean indicating whether a linear assign is possible
     */
    template <class D>
    template <class S>
    inline bool xoptional_assembly_base<D>::has_linear_assign(const S& strides) const noexcept
    {
        return value().has_linear_assign(strides) && has_value().has_linear_assign(strides);
    }

    //@}

    template <class D>
    inline auto xoptional_assembly_base<D>::linear_begin() noexcept -> linear_iterator
    {
        return linear_iterator(value().linear_begin(), has_value().linear_begin());
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::linear_end() noexcept -> linear_iterator
    {
        return linear_iterator(value().linear_end(), has_value().linear_end());
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::linear_begin() const noexcept -> const_linear_iterator
    {
        return linear_cbegin();
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::linear_end() const noexcept -> const_linear_iterator
    {
        return linear_cend();
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::linear_cbegin() const noexcept -> const_linear_iterator
    {
        return const_linear_iterator(value().linear_cbegin(), has_value().linear_cbegin());
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::linear_cend() const noexcept -> const_linear_iterator
    {
        return const_linear_iterator(value().linear_cend(), has_value().linear_cend());
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::linear_rbegin() noexcept -> reverse_linear_iterator
    {
        return reverse_linear_iterator(linear_end());
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::linear_rend() noexcept -> reverse_linear_iterator
    {
        return reverse_linear_iterator(linear_begin());
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::linear_rbegin() const noexcept -> const_reverse_linear_iterator
    {
        return linear_crbegin();
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::linear_rend() const noexcept -> const_reverse_linear_iterator
    {
        return linear_crend();
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::linear_crbegin() const noexcept -> const_reverse_linear_iterator
    {
        return const_reverse_linear_iterator(linear_cend());
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::linear_crend() const noexcept -> const_reverse_linear_iterator
    {
        return const_reverse_linear_iterator(linear_cbegin());
    }

    template <class D>
    template <class S>
    inline auto xoptional_assembly_base<D>::stepper_begin(const S& shape) noexcept -> stepper
    {
        return stepper(value().stepper_begin(shape), has_value().stepper_begin(shape));
    }

    template <class D>
    template <class S>
    inline auto xoptional_assembly_base<D>::stepper_end(const S& shape, layout_type l) noexcept -> stepper
    {
        return stepper(value().stepper_end(shape, l), has_value().stepper_end(shape, l));
    }

    template <class D>
    template <class S>
    inline auto xoptional_assembly_base<D>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        return const_stepper(value().stepper_begin(shape), has_value().stepper_begin(shape));
    }

    template <class D>
    template <class S>
    inline auto xoptional_assembly_base<D>::stepper_end(const S& shape, layout_type l) const noexcept
        -> const_stepper
    {
        return const_stepper(value().stepper_end(shape, l), has_value().stepper_end(shape, l));
    }

    /**
     * Return an expression for the values of the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::value() noexcept -> value_expression
    {
        return derived_cast().value_impl();
    }

    /**
     * Return a constant expression for the values of the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::value() const noexcept -> const_value_expression
    {
        return derived_cast().value_impl();
    }

    /**
     * Return an expression for the missing mask of the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::has_value() noexcept -> flag_expression
    {
        return derived_cast().has_value_impl();
    }

    /**
     * Return a constant expression for the missing mask of the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::has_value() const noexcept -> const_flag_expression
    {
        return derived_cast().has_value_impl();
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::derived_cast() noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::derived_cast() const noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    /*********************************************
     * xoptional_assembly_stepper implementation *
     *********************************************/

    template <class D, bool C>
    inline xoptional_assembly_stepper<D, C>::xoptional_assembly_stepper(value_stepper vs, flag_stepper fs) noexcept
        : m_vs(vs)
        , m_fs(fs)
    {
    }

    template <class D, bool C>
    inline void xoptional_assembly_stepper<D, C>::step(size_type dim)
    {
        m_vs.step(dim);
        m_fs.step(dim);
    }

    template <class D, bool C>
    inline void xoptional_assembly_stepper<D, C>::step_back(size_type dim)
    {
        m_vs.step_back(dim);
        m_fs.step_back(dim);
    }

    template <class D, bool C>
    inline void xoptional_assembly_stepper<D, C>::step(size_type dim, size_type n)
    {
        m_vs.step(dim, n);
        m_fs.step(dim, n);
    }

    template <class D, bool C>
    inline void xoptional_assembly_stepper<D, C>::step_back(size_type dim, size_type n)
    {
        m_vs.step_back(dim, n);
        m_fs.step_back(dim, n);
    }

    template <class D, bool C>
    inline void xoptional_assembly_stepper<D, C>::reset(size_type dim)
    {
        m_vs.reset(dim);
        m_fs.reset(dim);
    }

    template <class D, bool C>
    inline void xoptional_assembly_stepper<D, C>::reset_back(size_type dim)
    {
        m_vs.reset_back(dim);
        m_fs.reset_back(dim);
    }

    template <class D, bool C>
    inline void xoptional_assembly_stepper<D, C>::to_begin()
    {
        m_vs.to_begin();
        m_fs.to_begin();
    }

    template <class D, bool C>
    inline void xoptional_assembly_stepper<D, C>::to_end(layout_type l)
    {
        m_vs.to_end(l);
        m_fs.to_end(l);
    }

    template <class D, bool C>
    inline auto xoptional_assembly_stepper<D, C>::operator*() const -> reference
    {
        return reference(*m_vs, *m_fs);
    }
}

#endif
