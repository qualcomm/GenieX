/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_CONTAINER_HPP
#define XTENSOR_CONTAINER_HPP

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>

#include <xtl/xmeta_utils.hpp>
#include <xtl/xsequence.hpp>

#include "../core/xaccessible.hpp"
#include "../core/xiterable.hpp"
#include "../core/xiterator.hpp"
#include "../core/xmath.hpp"
#include "../core/xoperation.hpp"
#include "../core/xstrides.hpp"
#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"

namespace xt
{
    template <class D>
    struct xcontainer_iterable_types
    {
        using inner_shape_type = typename xcontainer_inner_types<D>::inner_shape_type;
        using stepper = xstepper<D>;
        using const_stepper = xstepper<const D>;
    };

    namespace detail
    {
        template <class T>
        struct allocator_type_impl
        {
            using type = typename T::allocator_type;
        };

        template <class T, std::size_t N>
        struct allocator_type_impl<std::array<T, N>>
        {
            using type = std::allocator<T>;  // fake allocator for testing
        };
    }

    template <class T>
    using allocator_type_t = typename detail::allocator_type_impl<T>::type;

    /**
     * @class xcontainer
     * @brief Base class for dense multidimensional containers.
     *
     * The xcontainer class defines the interface for dense multidimensional
     * container classes. It does not embed any data container, this responsibility
     * is delegated to the inheriting classes.
     *
     * @tparam D The derived type, i.e. the inheriting class for which xcontainer
     *           provides the interface.
     */
    template <class D>
    class xcontainer : public xcontiguous_iterable<D>,
                       private xaccessible<D>
    {
    public:

        using derived_type = D;

        using inner_types = xcontainer_inner_types<D>;
        using storage_type = typename inner_types::storage_type;
        using allocator_type = allocator_type_t<std::decay_t<storage_type>>;
        using value_type = typename storage_type::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = typename storage_type::pointer;
        using const_pointer = typename storage_type::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename storage_type::difference_type;
        using simd_value_type = xt_simd::simd_type<value_type>;
        using bool_load_type = xt::bool_load_type<value_type>;

        using shape_type = typename inner_types::shape_type;
        using strides_type = typename inner_types::strides_type;
        using backstrides_type = typename inner_types::backstrides_type;

        using inner_shape_type = typename inner_types::inner_shape_type;
        using inner_strides_type = typename inner_types::inner_strides_type;
        using inner_backstrides_type = typename inner_types::inner_backstrides_type;

        using iterable_base = xcontiguous_iterable<D>;
        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        using accessible_base = xaccessible<D>;

        static constexpr layout_type static_layout = inner_types::layout;
        static constexpr bool contiguous_layout = static_layout != layout_type::dynamic;
        using data_alignment = xt_simd::container_alignment_t<storage_type>;
        using simd_type = xt_simd::simd_type<value_type>;

        using linear_iterator = typename iterable_base::linear_iterator;
        using const_linear_iterator = typename iterable_base::const_linear_iterator;
        using reverse_linear_iterator = typename iterable_base::reverse_linear_iterator;
        using const_reverse_linear_iterator = typename iterable_base::const_reverse_linear_iterator;

        static_assert(static_layout != layout_type::any, "Container layout can never be layout_type::any!");

        size_type size() const noexcept;

        XTENSOR_CONSTEXPR_RETURN size_type dimension() const noexcept;

        XTENSOR_CONSTEXPR_RETURN const inner_shape_type& shape() const noexcept;
        XTENSOR_CONSTEXPR_RETURN const inner_strides_type& strides() const noexcept;
        XTENSOR_CONSTEXPR_RETURN const inner_backstrides_type& backstrides() const noexcept;

        template <class T>
        void fill(const T& value);

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        const_reference operator()(Args... args) const;

        template <class... Args>
        reference unchecked(Args... args);

        template <class... Args>
        const_reference unchecked(Args... args) const;

        using accessible_base::at;
        using accessible_base::shape;
        using accessible_base::operator[];
        using accessible_base::back;
        using accessible_base::front;
        using accessible_base::in_bounds;
        using accessible_base::periodic;

        template <class It>
        reference element(It first, It last);
        template <class It>
        const_reference element(It first, It last) const;

        storage_type& storage() noexcept;
        const storage_type& storage() const noexcept;

        pointer data() noexcept;
        const_pointer data() const noexcept;
        const size_type data_offset() const noexcept;

        template <class S>
        bool broadcast_shape(S& shape, bool reuse_cache = false) const;

        template <class S>
        bool has_linear_assign(const S& strides) const noexcept;
        template <class S>
        stepper stepper_begin(const S& shape) noexcept;
        template <class S>
        stepper stepper_end(const S& shape, layout_type l) noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape, layout_type l) const noexcept;

        reference data_element(size_type i);
        const_reference data_element(size_type i) const;

        reference flat(size_type i);
        const_reference flat(size_type i) const;

        template <class requested_type>
        using simd_return_type = xt_simd::simd_return_type<value_type, requested_type>;

        template <class align, class simd>
        void store_simd(size_type i, const simd& e);
        template <class align, class requested_type = value_type, std::size_t N = xt_simd::simd_traits<requested_type>::size>
        container_simd_return_type_t<storage_type, value_type, requested_type>
        /*simd_return_type<requested_type>*/ load_simd(size_type i) const;

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

        using container_iterator = linear_iterator;
        using const_container_iterator = const_linear_iterator;

    protected:

        xcontainer() = default;
        ~xcontainer() = default;

        xcontainer(const xcontainer&) = default;
        xcontainer& operator=(const xcontainer&) = default;

        xcontainer(xcontainer&&) = default;
        xcontainer& operator=(xcontainer&&) = default;

        container_iterator data_xbegin() noexcept;
        const_container_iterator data_xbegin() const noexcept;
        container_iterator data_xend(layout_type l, size_type offset) noexcept;
        const_container_iterator data_xend(layout_type l, size_type offset) const noexcept;

    protected:

        derived_type& derived_cast() & noexcept;
        const derived_type& derived_cast() const& noexcept;
        derived_type derived_cast() && noexcept;

    private:

        template <class It>
        It data_xend_impl(It begin, layout_type l, size_type offset) const noexcept;

        inner_shape_type& mutable_shape();
        inner_strides_type& mutable_strides();
        inner_backstrides_type& mutable_backstrides();

        template <class C>
        friend class xstepper;

        friend class xaccessible<D>;
        friend class xconst_accessible<D>;
    };

    /**
     * @class xstrided_container
     * @brief Partial implementation of xcontainer that embeds the strides and the shape
     *
     * The xstrided_container class is a partial implementation of the xcontainer interface
     * that embed the strides and the shape of the multidimensional container. It does
     * not embed the data container, this responsibility is delegated to the inheriting
     * classes.
     *
     * @tparam D The derived type, i.e. the inheriting class for which xstrided_container
     *           provides the partial imlpementation of xcontainer.
     */
    template <class D>
    class xstrided_container : public xcontainer<D>
    {
    public:

        using base_type = xcontainer<D>;
        using storage_type = typename base_type::storage_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using size_type = typename base_type::size_type;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using inner_shape_type = typename base_type::inner_shape_type;
        using inner_strides_type = typename base_type::inner_strides_type;
        using inner_backstrides_type = typename base_type::inner_backstrides_type;

        template <class S = shape_type>
        void resize(S&& shape, bool force = false);
        template <class S = shape_type>
        void resize(S&& shape, layout_type l);
        template <class S = shape_type>
        void resize(S&& shape, const strides_type& strides);

        template <class S = shape_type>
        auto& reshape(S&& shape, layout_type layout = base_type::static_layout) &;

        template <class T>
        auto& reshape(std::initializer_list<T> shape, layout_type layout = base_type::static_layout) &;

        layout_type layout() const noexcept;
        bool is_contiguous() const noexcept;

    protected:

        xstrided_container() noexcept;
        ~xstrided_container() = default;

        xstrided_container(const xstrided_container&) = default;
        xstrided_container& operator=(const xstrided_container&) = default;

        xstrided_container(xstrided_container&&) = default;
        xstrided_container& operator=(xstrided_container&&) = default;

        explicit xstrided_container(inner_shape_type&&, inner_strides_type&&) noexcept;
        explicit xstrided_container(inner_shape_type&&, inner_strides_type&&, inner_backstrides_type&&, layout_type&&) noexcept;

        inner_shape_type& shape_impl() noexcept;
        const inner_shape_type& shape_impl() const noexcept;

        inner_strides_type& strides_impl() noexcept;
        const inner_strides_type& strides_impl() const noexcept;

        inner_backstrides_type& backstrides_impl() noexcept;
        const inner_backstrides_type& backstrides_impl() const noexcept;

        template <class S = shape_type>
        void reshape_impl(S&& shape, std::true_type, layout_type layout = base_type::static_layout);
        template <class S = shape_type>
        void reshape_impl(S&& shape, std::false_type, layout_type layout = base_type::static_layout);

        layout_type& mutable_layout() noexcept;

    private:

        inner_shape_type m_shape;
        inner_strides_type m_strides;
        inner_backstrides_type m_backstrides;
        layout_type m_layout = base_type::static_layout;
    };

    /******************************
     * xcontainer implementation *
     ******************************/

    template <class D>
    template <class It>
    inline It xcontainer<D>::data_xend_impl(It begin, layout_type l, size_type offset) const noexcept
    {
        return strided_data_end(*this, begin, l, offset);
    }

    template <class D>
    inline auto xcontainer<D>::mutable_shape() -> inner_shape_type&
    {
        return derived_cast().shape_impl();
    }

    template <class D>
    inline auto xcontainer<D>::mutable_strides() -> inner_strides_type&
    {
        return derived_cast().strides_impl();
    }

    template <class D>
    inline auto xcontainer<D>::mutable_backstrides() -> inner_backstrides_type&
    {
        return derived_cast().backstrides_impl();
    }

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the number of element in the container.
     */
    template <class D>
    inline auto xcontainer<D>::size() const noexcept -> size_type
    {
        return contiguous_layout ? storage().size() : compute_size(shape());
    }

    /**
     * Returns the number of dimensions of the container.
     */
    template <class D>
    XTENSOR_CONSTEXPR_RETURN auto xcontainer<D>::dimension() const noexcept -> size_type
    {
        return shape().size();
    }

    /**
     * Returns the shape of the container.
     */
    template <class D>
    XTENSOR_CONSTEXPR_RETURN auto xcontainer<D>::shape() const noexcept -> const inner_shape_type&
    {
        return derived_cast().shape_impl();
    }

    /**
     * Returns the strides of the container.
     */
    template <class D>
    XTENSOR_CONSTEXPR_RETURN auto xcontainer<D>::strides() const noexcept -> const inner_strides_type&
    {
        return derived_cast().strides_impl();
    }

    /**
     * Returns the backstrides of the container.
     */
    template <class D>
    XTENSOR_CONSTEXPR_RETURN auto xcontainer<D>::backstrides() const noexcept -> const inner_backstrides_type&
    {
        return derived_cast().backstrides_impl();
    }

    //@}

    /**
     * @name Data
     */
    //@{

    /**
     * Fills the container with the given value.
     * @param value the value to fill the container with.
     */
    template <class D>
    template <class T>
    inline void xcontainer<D>::fill(const T& value)
    {
        if (contiguous_layout)
        {
            std::fill(this->linear_begin(), this->linear_end(), value);
        }
        else
        {
            std::fill(this->begin(), this->end(), value);
        }
    }

    /**
     * Returns a reference to the element at the specified position in the container.
     * @param args a list of indices specifying the position in the container. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the container.
     */
    template <class D>
    template <class... Args>
    inline auto xcontainer<D>::operator()(Args... args) -> reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        size_type index = xt::data_offset<size_type>(strides(), args...);
        return storage()[index];
    }

    /**
     * Returns a constant reference to the element at the specified position in the container.
     * @param args a list of indices specifying the position in the container. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the container.
     */
    template <class D>
    template <class... Args>
    inline auto xcontainer<D>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        size_type index = xt::data_offset<size_type>(strides(), args...);
        return storage()[index];
    }

    /**
     * Returns a reference to the element at the specified position in the container.
     * @param args a list of indices specifying the position in the container. Indices
     * must be unsigned integers, the number of indices must be equal to the number of
     * dimensions of the container, else the behavior is undefined.
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
    inline auto xcontainer<D>::unchecked(Args... args) -> reference
    {
        size_type index = xt::unchecked_data_offset<size_type, static_layout>(
            strides(),
            static_cast<std::ptrdiff_t>(args)...
        );
        return storage()[index];
    }

    /**
     * Returns a constant reference to the element at the specified position in the container.
     * @param args a list of indices specifying the position in the container. Indices
     * must be unsigned integers, the number of indices must be equal to the number of
     * dimensions of the container, else the behavior is undefined.
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
    inline auto xcontainer<D>::unchecked(Args... args) const -> const_reference
    {
        size_type index = xt::unchecked_data_offset<size_type, static_layout>(
            strides(),
            static_cast<std::ptrdiff_t>(args)...
        );
        return storage()[index];
    }

    /**
     * Returns a reference to the element at the specified position in the container.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the container.
     */
    template <class D>
    template <class It>
    inline auto xcontainer<D>::element(It first, It last) -> reference
    {
        XTENSOR_TRY(check_element_index(shape(), first, last));
        return storage()[element_offset<size_type>(strides(), first, last)];
    }

    /**
     * Returns a reference to the element at the specified position in the container.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the container.
     */
    template <class D>
    template <class It>
    inline auto xcontainer<D>::element(It first, It last) const -> const_reference
    {
        XTENSOR_TRY(check_element_index(shape(), first, last));
        return storage()[element_offset<size_type>(strides(), first, last)];
    }

    /**
     * Returns a reference to the buffer containing the elements of the container.
     */
    template <class D>
    inline auto xcontainer<D>::storage() noexcept -> storage_type&
    {
        return derived_cast().storage_impl();
    }

    /**
     * Returns a constant reference to the buffer containing the elements of the
     * container.
     */
    template <class D>
    inline auto xcontainer<D>::storage() const noexcept -> const storage_type&
    {
        return derived_cast().storage_impl();
    }

    /**
     * Returns a pointer to the underlying array serving as element storage. The pointer
     * is such that range [data(); data() + size()] is always a valid range, even if the
     * container is empty (data() is not is not dereferenceable in that case)
     */
    template <class D>
    inline auto xcontainer<D>::data() noexcept -> pointer
    {
        return storage().data();
    }

    /**
     * Returns a constant pointer to the underlying array serving as element storage. The pointer
     * is such that range [data(); data() + size()] is always a valid range, even if the
     * container is empty (data() is not is not dereferenceable in that case)
     */
    template <class D>
    inline auto xcontainer<D>::data() const noexcept -> const_pointer
    {
        return storage().data();
    }

    /**
     * Returns the offset to the first element in the container.
     */
    template <class D>
    inline auto xcontainer<D>::data_offset() const noexcept -> const size_type
    {
        return size_type(0);
    }

    //@}

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the container to the specified parameter.
     * @param shape the result shape
     * @param reuse_cache parameter for internal optimization
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class D>
    template <class S>
    inline bool xcontainer<D>::broadcast_shape(S& shape, bool) const
    {
        return xt::broadcast_shape(this->shape(), shape);
    }

    /**
     * Checks whether the xcontainer can be linearly assigned to an expression
     * with the specified strides.
     * @return a boolean indicating whether a linear assign is possible
     */
    template <class D>
    template <class S>
    inline bool xcontainer<D>::has_linear_assign(const S& str) const noexcept
    {
        return str.size() == strides().size() && std::equal(str.cbegin(), str.cend(), strides().begin());
    }

    //@}

    template <class D>
    inline auto xcontainer<D>::derived_cast() const& noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    template <class D>
    inline auto xcontainer<D>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D>
    inline auto xcontainer<D>::data_element(size_type i) -> reference
    {
        return storage()[i];
    }

    template <class D>
    inline auto xcontainer<D>::data_element(size_type i) const -> const_reference
    {
        return storage()[i];
    }

    /**
     * Returns a reference to the element at the specified position in the container
     * storage (as if it was one dimensional).
     * @param i index specifying the position in the storage.
     * Must be smaller than the number of elements in the container.
     */
    template <class D>
    inline auto xcontainer<D>::flat(size_type i) -> reference
    {
        XTENSOR_ASSERT(i < size());
        return storage()[i];
    }

    /**
     * Returns a constant reference to the element at the specified position in the container
     * storage (as if it was one dimensional).
     * @param i index specifying the position in the storage.
     * Must be smaller than the number of elements in the container.
     */
    template <class D>
    inline auto xcontainer<D>::flat(size_type i) const -> const_reference
    {
        XTENSOR_ASSERT(i < size());
        return storage()[i];
    }

    /***************
     * stepper api *
     ***************/

    template <class D>
    template <class S>
    inline auto xcontainer<D>::stepper_begin(const S& shape) noexcept -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(static_cast<derived_type*>(this), data_xbegin(), offset);
    }

    template <class D>
    template <class S>
    inline auto xcontainer<D>::stepper_end(const S& shape, layout_type l) noexcept -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(static_cast<derived_type*>(this), data_xend(l, offset), offset);
    }

    template <class D>
    template <class S>
    inline auto xcontainer<D>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(static_cast<const derived_type*>(this), data_xbegin(), offset);
    }

    template <class D>
    template <class S>
    inline auto xcontainer<D>::stepper_end(const S& shape, layout_type l) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(static_cast<const derived_type*>(this), data_xend(l, offset), offset);
    }

    template <class D>
    inline auto xcontainer<D>::data_xbegin() noexcept -> container_iterator
    {
        return storage().begin();
    }

    template <class D>
    inline auto xcontainer<D>::data_xbegin() const noexcept -> const_container_iterator
    {
        return storage().cbegin();
    }

    template <class D>
    inline auto xcontainer<D>::data_xend(layout_type l, size_type offset) noexcept -> container_iterator
    {
        return data_xend_impl(storage().begin(), l, offset);
    }

    template <class D>
    inline auto xcontainer<D>::data_xend(layout_type l, size_type offset) const noexcept
        -> const_container_iterator
    {
        return data_xend_impl(storage().cbegin(), l, offset);
    }

    template <class D>
    template <class alignment, class simd>
    inline void xcontainer<D>::store_simd(size_type i, const simd& e)
    {
        using align_mode = driven_align_mode_t<alignment, data_alignment>;
        xt_simd::store_as(std::addressof(storage()[i]), e, align_mode());
    }

    template <class D>
    template <class alignment, class requested_type, std::size_t N>
    inline auto xcontainer<D>::load_simd(size_type i) const
        -> container_simd_return_type_t<storage_type, value_type, requested_type>
    {
        using align_mode = driven_align_mode_t<alignment, data_alignment>;
        return xt_simd::load_as<requested_type>(std::addressof(storage()[i]), align_mode());
    }

    template <class D>
    inline auto xcontainer<D>::linear_begin() noexcept -> linear_iterator
    {
        return storage().begin();
    }

    template <class D>
    inline auto xcontainer<D>::linear_end() noexcept -> linear_iterator
    {
        return storage().end();
    }

    template <class D>
    inline auto xcontainer<D>::linear_begin() const noexcept -> const_linear_iterator
    {
        return storage().begin();
    }

    template <class D>
    inline auto xcontainer<D>::linear_end() const noexcept -> const_linear_iterator
    {
        return storage().cend();
    }

    template <class D>
    inline auto xcontainer<D>::linear_cbegin() const noexcept -> const_linear_iterator
    {
        return storage().cbegin();
    }

    template <class D>
    inline auto xcontainer<D>::linear_cend() const noexcept -> const_linear_iterator
    {
        return storage().cend();
    }

    template <class D>
    inline auto xcontainer<D>::linear_rbegin() noexcept -> reverse_linear_iterator
    {
        return storage().rbegin();
    }

    template <class D>
    inline auto xcontainer<D>::linear_rend() noexcept -> reverse_linear_iterator
    {
        return storage().rend();
    }

    template <class D>
    inline auto xcontainer<D>::linear_rbegin() const noexcept -> const_reverse_linear_iterator
    {
        return storage().rbegin();
    }

    template <class D>
    inline auto xcontainer<D>::linear_rend() const noexcept -> const_reverse_linear_iterator
    {
        return storage().rend();
    }

    template <class D>
    inline auto xcontainer<D>::linear_crbegin() const noexcept -> const_reverse_linear_iterator
    {
        return storage().crbegin();
    }

    template <class D>
    inline auto xcontainer<D>::linear_crend() const noexcept -> const_reverse_linear_iterator
    {
        return storage().crend();
    }

    template <class D>
    inline auto xcontainer<D>::derived_cast() & noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    /*************************************
     * xstrided_container implementation *
     *************************************/

    template <class D>
    inline xstrided_container<D>::xstrided_container() noexcept
        : base_type()
    {
        m_shape = xtl::make_sequence<inner_shape_type>(base_type::dimension(), 0);
        m_strides = xtl::make_sequence<inner_strides_type>(base_type::dimension(), 0);
        m_backstrides = xtl::make_sequence<inner_backstrides_type>(base_type::dimension(), 0);
    }

    template <class D>
    inline xstrided_container<D>::xstrided_container(inner_shape_type&& shape, inner_strides_type&& strides) noexcept
        : base_type()
        , m_shape(std::move(shape))
        , m_strides(std::move(strides))
    {
        m_backstrides = xtl::make_sequence<inner_backstrides_type>(m_shape.size(), 0);
        adapt_strides(m_shape, m_strides, m_backstrides);
    }

    template <class D>
    inline xstrided_container<D>::xstrided_container(
        inner_shape_type&& shape,
        inner_strides_type&& strides,
        inner_backstrides_type&& backstrides,
        layout_type&& layout
    ) noexcept
        : base_type()
        , m_shape(std::move(shape))
        , m_strides(std::move(strides))
        , m_backstrides(std::move(backstrides))
        , m_layout(std::move(layout))
    {
    }

    template <class D>
    inline auto xstrided_container<D>::shape_impl() noexcept -> inner_shape_type&
    {
        return m_shape;
    }

    template <class D>
    inline auto xstrided_container<D>::shape_impl() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    template <class D>
    inline auto xstrided_container<D>::strides_impl() noexcept -> inner_strides_type&
    {
        return m_strides;
    }

    template <class D>
    inline auto xstrided_container<D>::strides_impl() const noexcept -> const inner_strides_type&
    {
        return m_strides;
    }

    template <class D>
    inline auto xstrided_container<D>::backstrides_impl() noexcept -> inner_backstrides_type&
    {
        return m_backstrides;
    }

    template <class D>
    inline auto xstrided_container<D>::backstrides_impl() const noexcept -> const inner_backstrides_type&
    {
        return m_backstrides;
    }

    /**
     * Return the layout_type of the container
     * @return layout_type of the container
     */
    template <class D>
    inline layout_type xstrided_container<D>::layout() const noexcept
    {
        return m_layout;
    }

    template <class D>
    inline bool xstrided_container<D>::is_contiguous() const noexcept
    {
        using str_type = typename inner_strides_type::value_type;
        auto is_zero = [](auto i)
        {
            return i == 0;
        };
        if (!is_contiguous_container<storage_type>::value)
        {
            return false;
        }
        // We need to make sure the inner-most non-zero stride is one.
        // Trailing zero strides are ignored because they indicate bradcasted dimensions.
        if (m_layout == layout_type::row_major)
        {
            auto it = std::find_if_not(m_strides.rbegin(), m_strides.rend(), is_zero);
            // If the array has strides of zero, it is a constant, and therefore contiguous.
            return it == m_strides.rend() || *it == str_type(1);
        }
        else if (m_layout == layout_type::column_major)
        {
            auto it = std::find_if_not(m_strides.begin(), m_strides.end(), is_zero);
            // If the array has strides of zero, it is a constant, and therefore contiguous.
            return it == m_strides.end() || *it == str_type(1);
        }
        else
        {
            return m_strides.empty();
        }
    }

    namespace detail
    {
        template <class C, class S>
        inline void resize_data_container(C& c, S size)
        {
            xt::resize_container(c, size);
        }

        template <class C, class S>
        inline void resize_data_container(const C& c, S size)
        {
            (void) c;  // remove unused parameter warning
            (void) size;
            XTENSOR_ASSERT_MSG(c.size() == size, "Trying to resize const data container with wrong size.");
        }

        template <class S, class T>
        constexpr bool check_resize_dimension(const S&, const T&)
        {
            return true;
        }

        template <class T, size_t N, class S>
        constexpr bool check_resize_dimension(const std::array<T, N>&, const S& s)
        {
            return N == s.size();
        }
    }

    /**
     * Resizes the container.
     * @warning Contrary to STL containers like std::vector, resize
     * does NOT preserve the container elements.
     * @param shape the new shape
     * @param force force reshaping, even if the shape stays the same (default: false)
     */
    template <class D>
    template <class S>
    inline void xstrided_container<D>::resize(S&& shape, bool force)
    {
        XTENSOR_ASSERT_MSG(
            detail::check_resize_dimension(m_shape, shape),
            "cannot change the number of dimensions of xtensor"
        )
        std::size_t dim = shape.size();
        if (m_shape.size() != dim || !std::equal(std::begin(shape), std::end(shape), std::begin(m_shape))
            || force)
        {
            if (D::static_layout == layout_type::dynamic && m_layout == layout_type::dynamic)
            {
                m_layout = XTENSOR_DEFAULT_LAYOUT;  // fall back to default layout
            }
            m_shape = xtl::forward_sequence<shape_type, S>(shape);

            resize_container(m_strides, dim);
            resize_container(m_backstrides, dim);
            size_type data_size = compute_strides<D::static_layout>(m_shape, m_layout, m_strides, m_backstrides);
            detail::resize_data_container(this->storage(), data_size);
        }
    }

    /**
     * Resizes the container.
     * @warning Contrary to STL containers like std::vector, resize
     * does NOT preserve the container elements.
     * @param shape the new shape
     * @param l the new layout_type
     */
    template <class D>
    template <class S>
    inline void xstrided_container<D>::resize(S&& shape, layout_type l)
    {
        XTENSOR_ASSERT_MSG(
            detail::check_resize_dimension(m_shape, shape),
            "cannot change the number of dimensions of xtensor"
        )
        if (base_type::static_layout != layout_type::dynamic && l != base_type::static_layout)
        {
            XTENSOR_THROW(
                std::runtime_error,
                "Cannot change layout_type if template parameter not layout_type::dynamic."
            );
        }
        m_layout = l;
        resize(std::forward<S>(shape), true);
    }

    /**
     * Resizes the container.
     * @warning Contrary to STL containers like std::vector, resize
     * does NOT preserve the container elements.
     * @param shape the new shape
     * @param strides the new strides
     */
    template <class D>
    template <class S>
    inline void xstrided_container<D>::resize(S&& shape, const strides_type& strides)
    {
        XTENSOR_ASSERT_MSG(
            detail::check_resize_dimension(m_shape, shape),
            "cannot change the number of dimensions of xtensor"
        )
        if (base_type::static_layout != layout_type::dynamic)
        {
            XTENSOR_THROW(
                std::runtime_error,
                "Cannot resize with custom strides when layout() is != layout_type::dynamic."
            );
        }
        m_shape = xtl::forward_sequence<shape_type, S>(shape);
        m_strides = strides;
        resize_container(m_backstrides, m_strides.size());
        adapt_strides(m_shape, m_strides, m_backstrides);
        m_layout = layout_type::dynamic;
        detail::resize_data_container(this->storage(), compute_size(m_shape));
    }

    /**
     * Reshapes the container and keeps old elements. The `shape` argument can have one of its value
     * equal to `-1`, in this case the value is inferred from the number of elements in the container
     * and the remaining values in the `shape`.
     * @code{.cpp}
     * xt::xarray<int> a = { 1, 2, 3, 4, 5, 6, 7, 8 };
     * a.reshape({-1, 4});
     * //a.shape() is {2, 4}
     * @endcode
     * @param shape the new shape (has to have same number of elements as the original container)
     * @param layout the layout to compute the strides (defaults to static layout of the container,
     *               or for a container with dynamic layout to XTENSOR_DEFAULT_LAYOUT)
     */
    template <class D>
    template <class S>
    inline auto& xstrided_container<D>::reshape(S&& shape, layout_type layout) &
    {
        reshape_impl(
            std::forward<S>(shape),
            xtl::is_signed<std::decay_t<typename std::decay_t<S>::value_type>>(),
            std::forward<layout_type>(layout)
        );
        return this->derived_cast();
    }

    template <class D>
    template <class T>
    inline auto& xstrided_container<D>::reshape(std::initializer_list<T> shape, layout_type layout) &
    {
        using sh_type = rebind_container_t<T, shape_type>;
        sh_type sh = xtl::make_sequence<sh_type>(shape.size());
        std::copy(shape.begin(), shape.end(), sh.begin());
        reshape_impl(std::move(sh), xtl::is_signed<T>(), std::forward<layout_type>(layout));
        return this->derived_cast();
    }

    template <class D>
    template <class S>
    inline void
    xstrided_container<D>::reshape_impl(S&& shape, std::false_type /* is unsigned */, layout_type layout)
    {
        if (compute_size(shape) != this->size())
        {
            XTENSOR_THROW(
                std::runtime_error,
                "Cannot reshape with incorrect number of elements. Do you mean to resize?"
            );
        }
        if (D::static_layout == layout_type::dynamic && layout == layout_type::dynamic)
        {
            layout = XTENSOR_DEFAULT_LAYOUT;  // fall back to default layout
        }
        if (D::static_layout != layout_type::dynamic && layout != D::static_layout)
        {
            XTENSOR_THROW(std::runtime_error, "Cannot reshape with different layout if static layout != dynamic.");
        }
        m_layout = layout;
        m_shape = xtl::forward_sequence<shape_type, S>(shape);
        resize_container(m_strides, m_shape.size());
        resize_container(m_backstrides, m_shape.size());
        compute_strides<D::static_layout>(m_shape, m_layout, m_strides, m_backstrides);
    }

    template <class D>
    template <class S>
    inline void
    xstrided_container<D>::reshape_impl(S&& _shape, std::true_type /* is signed */, layout_type layout)
    {
        using tmp_value_type = typename std::decay_t<S>::value_type;
        auto new_size = compute_size(_shape);
        if (this->size() % new_size)
        {
            XTENSOR_THROW(std::runtime_error, "Negative axis size cannot be inferred. Shape mismatch.");
        }
        std::decay_t<S> shape = _shape;
        tmp_value_type accumulator = 1;
        std::size_t neg_idx = 0;
        std::size_t i = 0;
        for (auto it = shape.begin(); it != shape.end(); ++it, i++)
        {
            auto&& dim = *it;
            if (dim < 0)
            {
                XTENSOR_ASSERT(dim == -1 && !neg_idx);
                neg_idx = i;
            }
            accumulator *= dim;
        }
        if (accumulator < 0)
        {
            shape[neg_idx] = static_cast<tmp_value_type>(this->size()) / std::abs(accumulator);
        }
        else if (this->size() != new_size)
        {
            XTENSOR_THROW(
                std::runtime_error,
                "Cannot reshape with incorrect number of elements. Do you mean to resize?"
            );
        }
        m_layout = layout;
        m_shape = xtl::forward_sequence<shape_type, S>(shape);
        resize_container(m_strides, m_shape.size());
        resize_container(m_backstrides, m_shape.size());
        compute_strides<D::static_layout>(m_shape, m_layout, m_strides, m_backstrides);
    }

    template <class D>
    inline auto xstrided_container<D>::mutable_layout() noexcept -> layout_type&
    {
        return m_layout;
    }
}

#endif
