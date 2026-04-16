/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_TENSOR_HPP
#define XTENSOR_TENSOR_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "../containers/xbuffer_adaptor.hpp"
#include "../containers/xcontainer.hpp"
#include "../core/xsemantic.hpp"

namespace xt
{

    /***********************
     * xtensor declaration *
     ***********************/

    namespace extension
    {
        template <class EC, std::size_t N, layout_type L, class Tag>
        struct xtensor_container_base;

        template <class EC, std::size_t N, layout_type L>
        struct xtensor_container_base<EC, N, L, xtensor_expression_tag>
        {
            using type = xtensor_empty_base;
        };

        template <class EC, std::size_t N, layout_type L, class Tag>
        using xtensor_container_base_t = typename xtensor_container_base<EC, N, L, Tag>::type;
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    struct xcontainer_inner_types<xtensor_container<EC, N, L, Tag>>
    {
        using storage_type = EC;
        using reference = inner_reference_t<storage_type>;
        using const_reference = typename storage_type::const_reference;
        using size_type = typename storage_type::size_type;
        using shape_type = std::array<typename storage_type::size_type, N>;
        using strides_type = get_strides_t<shape_type>;
        using backstrides_type = get_strides_t<shape_type>;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xtensor_container<EC, N, L, Tag>;
        static constexpr layout_type layout = L;
    };

    template <class EC, std::size_t N, layout_type L, class Tag>
    struct xiterable_inner_types<xtensor_container<EC, N, L, Tag>>
        : xcontainer_iterable_types<xtensor_container<EC, N, L, Tag>>
    {
    };

    /**
     * @class xtensor_container
     * @brief Dense multidimensional container with tensor semantic and fixed
     * dimension.
     *
     * The xtensor_container class implements a dense multidimensional container
     * with tensor semantics and fixed dimension
     *
     * @tparam EC The type of the container holding the elements.
     * @tparam N The dimension of the container.
     * @tparam L The layout_type of the tensor.
     * @tparam Tag The expression tag.
     * @sa xtensor, xstrided_container, xcontainer
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    class xtensor_container : public xstrided_container<xtensor_container<EC, N, L, Tag>>,
                              public xcontainer_semantic<xtensor_container<EC, N, L, Tag>>,
                              public extension::xtensor_container_base_t<EC, N, L, Tag>
    {
    public:

        using self_type = xtensor_container<EC, N, L, Tag>;
        using base_type = xstrided_container<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using extension_base = extension::xtensor_container_base_t<EC, N, L, Tag>;
        using storage_type = typename base_type::storage_type;
        using allocator_type = typename base_type::allocator_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using inner_shape_type = typename base_type::inner_shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;
        using inner_backstrides_type = typename base_type::inner_backstrides_type;
        using inner_strides_type = typename base_type::inner_strides_type;
        using temporary_type = typename semantic_base::temporary_type;
        using expression_tag = Tag;
        static constexpr std::size_t rank = N;

        xtensor_container();
        xtensor_container(nested_initializer_list_t<value_type, N> t);
        explicit xtensor_container(const shape_type& shape, layout_type l = L);
        explicit xtensor_container(const shape_type& shape, const_reference value, layout_type l = L);
        explicit xtensor_container(const shape_type& shape, const strides_type& strides);
        explicit xtensor_container(const shape_type& shape, const strides_type& strides, const_reference value);
        explicit xtensor_container(storage_type&& storage, inner_shape_type&& shape, inner_strides_type&& strides);

        template <class S = shape_type>
        static xtensor_container from_shape(S&& s);

        ~xtensor_container() = default;

        xtensor_container(const xtensor_container&) = default;
        xtensor_container& operator=(const xtensor_container&) = default;

        xtensor_container(xtensor_container&&) = default;
        xtensor_container& operator=(xtensor_container&&) = default;

        template <class SC>
        explicit xtensor_container(xarray_container<EC, L, SC, Tag>&&);
        template <class SC>
        xtensor_container& operator=(xarray_container<EC, L, SC, Tag>&&);

        template <class E>
        xtensor_container(const xexpression<E>& e);

        template <class E>
        xtensor_container& operator=(const xexpression<E>& e);

    private:

        storage_type m_storage;

        storage_type& storage_impl() noexcept;
        const storage_type& storage_impl() const noexcept;

        friend class xcontainer<xtensor_container<EC, N, L, Tag>>;
    };

    /*****************************************
     * xtensor_container_adaptor declaration *
     *****************************************/

    namespace extension
    {
        template <class EC, std::size_t N, layout_type L, class Tag>
        struct xtensor_adaptor_base;

        template <class EC, std::size_t N, layout_type L>
        struct xtensor_adaptor_base<EC, N, L, xtensor_expression_tag>
        {
            using type = xtensor_empty_base;
        };

        template <class EC, std::size_t N, layout_type L, class Tag>
        using xtensor_adaptor_base_t = typename xtensor_adaptor_base<EC, N, L, Tag>::type;
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    struct xcontainer_inner_types<xtensor_adaptor<EC, N, L, Tag>>
    {
        using storage_type = std::remove_reference_t<EC>;
        using reference = inner_reference_t<storage_type>;
        using const_reference = typename storage_type::const_reference;
        using size_type = typename storage_type::size_type;
        using shape_type = std::array<typename storage_type::size_type, N>;
        using strides_type = get_strides_t<shape_type>;
        using backstrides_type = get_strides_t<shape_type>;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xtensor_container<temporary_container_t<storage_type>, N, L, Tag>;
        static constexpr layout_type layout = L;
    };

    template <class EC, std::size_t N, layout_type L, class Tag>
    struct xiterable_inner_types<xtensor_adaptor<EC, N, L, Tag>>
        : xcontainer_iterable_types<xtensor_adaptor<EC, N, L, Tag>>
    {
    };

    /**
     * @class xtensor_adaptor
     * @brief Dense multidimensional container adaptor with tensor
     * semantics and fixed dimension.
     *
     * The xtensor_adaptor class implements a dense multidimensional
     * container adaptor with tensor semantics and fixed dimension. It
     * is used to provide a multidimensional container semantic and a
     * tensor semantic to stl-like containers.
     *
     * @tparam EC The closure for the container type to adapt.
     * @tparam N The dimension of the adaptor.
     * @tparam L The layout_type of the adaptor.
     * @tparam Tag The expression tag.
     * @sa xstrided_container, xcontainer
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    class xtensor_adaptor : public xstrided_container<xtensor_adaptor<EC, N, L, Tag>>,
                            public xcontainer_semantic<xtensor_adaptor<EC, N, L, Tag>>,
                            public extension::xtensor_adaptor_base_t<EC, N, L, Tag>
    {
    public:

        using container_closure_type = EC;

        using self_type = xtensor_adaptor<EC, N, L, Tag>;
        using base_type = xstrided_container<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using extension_base = extension::xtensor_adaptor_base_t<EC, N, L, Tag>;
        using storage_type = typename base_type::storage_type;
        using allocator_type = typename base_type::allocator_type;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;
        using temporary_type = typename semantic_base::temporary_type;
        using expression_tag = Tag;
        static constexpr std::size_t rank = N;

        xtensor_adaptor(storage_type&& storage);
        xtensor_adaptor(const storage_type& storage);

        template <class D>
        xtensor_adaptor(D&& storage, const shape_type& shape, layout_type l = L);

        template <class D>
        xtensor_adaptor(D&& storage, const shape_type& shape, const strides_type& strides);

        ~xtensor_adaptor() = default;

        xtensor_adaptor(const xtensor_adaptor&) = default;
        xtensor_adaptor& operator=(const xtensor_adaptor&);

        xtensor_adaptor(xtensor_adaptor&&) = default;
        xtensor_adaptor& operator=(xtensor_adaptor&&);
        xtensor_adaptor& operator=(temporary_type&&);

        template <class E>
        xtensor_adaptor& operator=(const xexpression<E>& e);

        template <class P, class S>
        void reset_buffer(P&& pointer, S&& size);

    private:

        container_closure_type m_storage;

        storage_type& storage_impl() noexcept;
        const storage_type& storage_impl() const noexcept;

        friend class xcontainer<xtensor_adaptor<EC, N, L, Tag>>;
    };

    /****************************
     * xtensor_view declaration *
     ****************************/

    template <class EC, std::size_t N, layout_type L, class Tag>
    class xtensor_view;

    namespace extension
    {
        template <class EC, std::size_t N, layout_type L, class Tag>
        struct xtensor_view_base;

        template <class EC, std::size_t N, layout_type L>
        struct xtensor_view_base<EC, N, L, xtensor_expression_tag>
        {
            using type = xtensor_empty_base;
        };

        template <class EC, std::size_t N, layout_type L, class Tag>
        using xtensor_view_base_t = typename xtensor_view_base<EC, N, L, Tag>::type;
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    struct xcontainer_inner_types<xtensor_view<EC, N, L, Tag>>
    {
        using storage_type = std::remove_reference_t<EC>;
        using reference = inner_reference_t<storage_type>;
        using const_reference = typename storage_type::const_reference;
        using size_type = typename storage_type::size_type;
        using shape_type = std::array<typename storage_type::size_type, N>;
        using strides_type = get_strides_t<shape_type>;
        using backstrides_type = get_strides_t<shape_type>;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xtensor_container<temporary_container_t<storage_type>, N, L, Tag>;
        static constexpr layout_type layout = L;
    };

    template <class EC, std::size_t N, layout_type L, class Tag>
    struct xiterable_inner_types<xtensor_view<EC, N, L, Tag>>
        : xcontainer_iterable_types<xtensor_view<EC, N, L, Tag>>
    {
    };

    /**
     * @class xtensor_view
     * @brief Dense multidimensional container adaptor with view
     * semantics and fixed dimension.
     *
     * The xtensor_view class implements a dense multidimensional
     * container adaptor with viewsemantics and fixed dimension. It
     * is used to provide a multidimensional container semantic and a
     * view semantic to stl-like containers.
     *
     * @tparam EC The closure for the container type to adapt.
     * @tparam N The dimension of the view.
     * @tparam L The layout_type of the view.
     * @tparam Tag The expression tag.
     * @sa xstrided_container, xcontainer
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    class xtensor_view : public xstrided_container<xtensor_view<EC, N, L, Tag>>,
                         public xview_semantic<xtensor_view<EC, N, L, Tag>>,
                         public extension::xtensor_view_base_t<EC, N, L, Tag>
    {
    public:

        using container_closure_type = EC;

        using self_type = xtensor_view<EC, N, L, Tag>;
        using base_type = xstrided_container<self_type>;
        using semantic_base = xview_semantic<self_type>;
        using extension_base = extension::xtensor_adaptor_base_t<EC, N, L, Tag>;
        using storage_type = typename base_type::storage_type;
        using allocator_type = typename base_type::allocator_type;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;
        using temporary_type = typename semantic_base::temporary_type;
        using expression_tag = Tag;

        xtensor_view(storage_type&& storage);
        xtensor_view(const storage_type& storage);

        template <class D>
        xtensor_view(D&& storage, const shape_type& shape, layout_type l = L);

        template <class D>
        xtensor_view(D&& storage, const shape_type& shape, const strides_type& strides);

        ~xtensor_view() = default;

        xtensor_view(const xtensor_view&) = default;
        xtensor_view& operator=(const xtensor_view&);

        xtensor_view(xtensor_view&&) = default;
        xtensor_view& operator=(xtensor_view&&);

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

    private:

        container_closure_type m_storage;

        storage_type& storage_impl() noexcept;
        const storage_type& storage_impl() const noexcept;

        void assign_temporary_impl(temporary_type&& tmp);

        friend class xcontainer<xtensor_view<EC, N, L, Tag>>;
        friend class xview_semantic<xtensor_view<EC, N, L, Tag>>;
    };

    namespace detail
    {
        template <class V>
        struct tensor_view_simd_helper
        {
            using valid_return_type = detail::has_simd_interface_impl<V, typename V::value_type>;
            using valid_reference = std::is_lvalue_reference<typename V::reference>;
            static constexpr bool value = valid_return_type::value && valid_reference::value;
            using type = std::integral_constant<bool, value>;
        };
    }

    // xtensor_view can be used on pseudo containers, i.e. containers
    // whose access operator does not return a reference. Since it
    // is not possible to take the address f a temporary, the load_simd
    // method implementation leads to a compilation error.
    template <class EC, std::size_t N, layout_type L, class Tag>
    struct has_simd_interface<xtensor_view<EC, N, L, Tag>>
        : detail::tensor_view_simd_helper<xtensor_view<EC, N, L, Tag>>::type
    {
    };

    /************************************
     * xtensor_container implementation *
     ************************************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Allocates an uninitialized xtensor_container that holds 0 elements.
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_container<EC, N, L, Tag>::xtensor_container()
        : base_type()
        , m_storage(N == 0 ? 1 : 0, value_type())
    {
    }

    /**
     * Allocates an xtensor_container with nested initializer lists.
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_container<EC, N, L, Tag>::xtensor_container(nested_initializer_list_t<value_type, N> t)
        : base_type()
    {
        base_type::resize(xt::shape<shape_type>(t), true);
        constexpr auto tmp = layout_type::row_major;
        L == tmp ? nested_copy(m_storage.begin(), t) : nested_copy(this->template begin<tmp>(), t);
    }

    /**
     * Allocates an uninitialized xtensor_container with the specified shape and
     * layout_type.
     * @param shape the shape of the xtensor_container
     * @param l the layout_type of the xtensor_container
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_container<EC, N, L, Tag>::xtensor_container(const shape_type& shape, layout_type l)
        : base_type()
    {
        base_type::resize(shape, l);
    }

    /**
     * Allocates an xtensor_container with the specified shape and layout_type. Elements
     * are initialized to the specified value.
     * @param shape the shape of the xtensor_container
     * @param value the value of the elements
     * @param l the layout_type of the xtensor_container
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_container<EC, N, L, Tag>::xtensor_container(
        const shape_type& shape,
        const_reference value,
        layout_type l
    )
        : base_type()
    {
        base_type::resize(shape, l);
        std::fill(m_storage.begin(), m_storage.end(), value);
    }

    /**
     * Allocates an uninitialized xtensor_container with the specified shape and strides.
     * @param shape the shape of the xtensor_container
     * @param strides the strides of the xtensor_container
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_container<EC, N, L, Tag>::xtensor_container(const shape_type& shape, const strides_type& strides)
        : base_type()
    {
        base_type::resize(shape, strides);
    }

    /**
     * Allocates an uninitialized xtensor_container with the specified shape and strides.
     * Elements are initialized to the specified value.
     * @param shape the shape of the xtensor_container
     * @param strides the strides of the xtensor_container
     * @param value the value of the elements
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_container<EC, N, L, Tag>::xtensor_container(
        const shape_type& shape,
        const strides_type& strides,
        const_reference value
    )
        : base_type()
    {
        base_type::resize(shape, strides);
        std::fill(m_storage.begin(), m_storage.end(), value);
    }

    /**
     * Allocates an xtensor_container by moving specified data, shape and strides
     *
     * @param storage the data for the xtensor_container
     * @param shape the shape of the xtensor_container
     * @param strides the strides of the xtensor_container
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_container<EC, N, L, Tag>::xtensor_container(
        storage_type&& storage,
        inner_shape_type&& shape,
        inner_strides_type&& strides
    )
        : base_type(std::move(shape), std::move(strides))
        , m_storage(std::move(storage))
    {
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    template <class SC>
    inline xtensor_container<EC, N, L, Tag>::xtensor_container(xarray_container<EC, L, SC, Tag>&& rhs)
        : base_type(
            xtl::forward_sequence<inner_shape_type, decltype(rhs.shape())>(rhs.shape()),
            xtl::forward_sequence<inner_strides_type, decltype(rhs.strides())>(rhs.strides()),
            xtl::forward_sequence<inner_backstrides_type, decltype(rhs.backstrides())>(rhs.backstrides()),
            std::move(rhs.layout())
        )
        , m_storage(std::move(rhs.storage()))
    {
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    template <class SC>
    inline xtensor_container<EC, N, L, Tag>&
    xtensor_container<EC, N, L, Tag>::operator=(xarray_container<EC, L, SC, Tag>&& rhs)
    {
        XTENSOR_ASSERT_MSG(N == rhs.dimension(), "Cannot change dimension of xtensor.");
        std::copy(rhs.shape().begin(), rhs.shape().end(), this->shape_impl().begin());
        std::copy(rhs.strides().cbegin(), rhs.strides().cend(), this->strides_impl().begin());
        std::copy(rhs.backstrides().cbegin(), rhs.backstrides().cend(), this->backstrides_impl().begin());
        this->mutable_layout() = std::move(rhs.layout());
        m_storage = std::move(std::move(rhs.storage()));
        return *this;
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    template <class S>
    inline xtensor_container<EC, N, L, Tag> xtensor_container<EC, N, L, Tag>::from_shape(S&& s)
    {
        XTENSOR_ASSERT_MSG(s.size() == N, "Cannot change dimension of xtensor.");
        shape_type shape = xtl::forward_sequence<shape_type, S>(s);
        return self_type(shape);
    }

    //@}

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended copy constructor.
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    template <class E>
    inline xtensor_container<EC, N, L, Tag>::xtensor_container(const xexpression<E>& e)
        : base_type()
    {
        XTENSOR_ASSERT_MSG(N == e.derived_cast().dimension(), "Cannot change dimension of xtensor.");
        // Avoids uninitialized data because of (m_shape == shape) condition
        // in resize (called by assign), which is always true when dimension() == 0.
        if (e.derived_cast().dimension() == 0)
        {
            detail::resize_data_container(m_storage, std::size_t(1));
        }
        semantic_base::assign(e);
    }

    /**
     * The extended assignment operator.
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    template <class E>
    inline auto xtensor_container<EC, N, L, Tag>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }

    //@}

    template <class EC, std::size_t N, layout_type L, class Tag>
    inline auto xtensor_container<EC, N, L, Tag>::storage_impl() noexcept -> storage_type&
    {
        return m_storage;
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    inline auto xtensor_container<EC, N, L, Tag>::storage_impl() const noexcept -> const storage_type&
    {
        return m_storage;
    }

    /**********************************
     * xtensor_adaptor implementation *
     **********************************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Constructs an xtensor_adaptor of the given stl-like container.
     * @param storage the container to adapt
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_adaptor<EC, N, L, Tag>::xtensor_adaptor(storage_type&& storage)
        : base_type()
        , m_storage(std::move(storage))
    {
    }

    /**
     * Constructs an xtensor_adaptor of the given stl-like container.
     * @param storage the container to adapt
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_adaptor<EC, N, L, Tag>::xtensor_adaptor(const storage_type& storage)
        : base_type()
        , m_storage(storage)
    {
    }

    /**
     * Constructs an xtensor_adaptor of the given stl-like container,
     * with the specified shape and layout_type.
     * @param storage the container to adapt
     * @param shape the shape of the xtensor_adaptor
     * @param l the layout_type of the xtensor_adaptor
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    template <class D>
    inline xtensor_adaptor<EC, N, L, Tag>::xtensor_adaptor(D&& storage, const shape_type& shape, layout_type l)
        : base_type()
        , m_storage(std::forward<D>(storage))
    {
        base_type::resize(shape, l);
    }

    /**
     * Constructs an xtensor_adaptor of the given stl-like container,
     * with the specified shape and strides.
     * @param storage the container to adapt
     * @param shape the shape of the xtensor_adaptor
     * @param strides the strides of the xtensor_adaptor
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    template <class D>
    inline xtensor_adaptor<EC, N, L, Tag>::xtensor_adaptor(
        D&& storage,
        const shape_type& shape,
        const strides_type& strides
    )
        : base_type()
        , m_storage(std::forward<D>(storage))
    {
        base_type::resize(shape, strides);
    }

    //@}

    template <class EC, std::size_t N, layout_type L, class Tag>
    inline auto xtensor_adaptor<EC, N, L, Tag>::operator=(const xtensor_adaptor& rhs) -> self_type&
    {
        base_type::operator=(rhs);
        m_storage = rhs.m_storage;
        return *this;
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    inline auto xtensor_adaptor<EC, N, L, Tag>::operator=(xtensor_adaptor&& rhs) -> self_type&
    {
        base_type::operator=(std::move(rhs));
        m_storage = rhs.m_storage;
        return *this;
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    inline auto xtensor_adaptor<EC, N, L, Tag>::operator=(temporary_type&& rhs) -> self_type&
    {
        base_type::shape_impl() = std::move(const_cast<shape_type&>(rhs.shape()));
        base_type::strides_impl() = std::move(const_cast<strides_type&>(rhs.strides()));
        base_type::backstrides_impl() = std::move(const_cast<backstrides_type&>(rhs.backstrides()));
        m_storage = std::move(rhs.storage());
        return *this;
    }

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    template <class E>
    inline auto xtensor_adaptor<EC, N, L, Tag>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }

    //@}

    template <class EC, std::size_t N, layout_type L, class Tag>
    inline auto xtensor_adaptor<EC, N, L, Tag>::storage_impl() noexcept -> storage_type&
    {
        return m_storage;
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    inline auto xtensor_adaptor<EC, N, L, Tag>::storage_impl() const noexcept -> const storage_type&
    {
        return m_storage;
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    template <class P, class S>
    inline void xtensor_adaptor<EC, N, L, Tag>::reset_buffer(P&& pointer, S&& size)
    {
        return m_storage.reset_data(std::forward<P>(pointer), std::forward<S>(size));
    }

    /*******************************
     * xtensor_view implementation *
     *******************************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Constructs an xtensor_view of the given stl-like container.
     * @param storage the container to adapt
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_view<EC, N, L, Tag>::xtensor_view(storage_type&& storage)
        : base_type()
        , m_storage(std::move(storage))
    {
    }

    /**
     * Constructs an xtensor_view of the given stl-like container.
     * @param storage the container to adapt
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_view<EC, N, L, Tag>::xtensor_view(const storage_type& storage)
        : base_type()
        , m_storage(storage)
    {
    }

    /**
     * Constructs an xtensor_view of the given stl-like container,
     * with the specified shape and layout_type.
     * @param storage the container to adapt
     * @param shape the shape of the xtensor_view
     * @param l the layout_type of the xtensor_view
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    template <class D>
    inline xtensor_view<EC, N, L, Tag>::xtensor_view(D&& storage, const shape_type& shape, layout_type l)
        : base_type()
        , m_storage(std::forward<D>(storage))
    {
        base_type::resize(shape, l);
    }

    /**
     * Constructs an xtensor_view of the given stl-like container,
     * with the specified shape and strides.
     * @param storage the container to adapt
     * @param shape the shape of the xtensor_view
     * @param strides the strides of the xtensor_view
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    template <class D>
    inline xtensor_view<EC, N, L, Tag>::xtensor_view(D&& storage, const shape_type& shape, const strides_type& strides)
        : base_type()
        , m_storage(std::forward<D>(storage))
    {
        base_type::resize(shape, strides);
    }

    //@}

    template <class EC, std::size_t N, layout_type L, class Tag>
    inline auto xtensor_view<EC, N, L, Tag>::operator=(const xtensor_view& rhs) -> self_type&
    {
        base_type::operator=(rhs);
        m_storage = rhs.m_storage;
        return *this;
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    inline auto xtensor_view<EC, N, L, Tag>::operator=(xtensor_view&& rhs) -> self_type&
    {
        base_type::operator=(std::move(rhs));
        m_storage = rhs.m_storage;
        return *this;
    }

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    template <class E>
    inline auto xtensor_view<EC, N, L, Tag>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }

    //@}

    template <class EC, std::size_t N, layout_type L, class Tag>
    template <class E>
    inline auto xtensor_view<EC, N, L, Tag>::operator=(const E& e) -> disable_xexpression<E, self_type>&
    {
        std::fill(m_storage.begin(), m_storage.end(), e);
        return *this;
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    inline auto xtensor_view<EC, N, L, Tag>::storage_impl() noexcept -> storage_type&
    {
        return m_storage;
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    inline auto xtensor_view<EC, N, L, Tag>::storage_impl() const noexcept -> const storage_type&
    {
        return m_storage;
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    inline void xtensor_view<EC, N, L, Tag>::assign_temporary_impl(temporary_type&& tmp)
    {
        std::copy(tmp.cbegin(), tmp.cend(), m_storage.begin());
    }

    /**
     * Converts ``std::vector<index_type>`` (returned e.g. from ``xt::argwhere``) to ``xtensor``.
     *
     * @param idx vector of indices
     *
     * @return ``xt::xtensor<typename index_type::value_type, 2>`` (e.g. ``xt::xtensor<size_t, 2>``)
     */
    template <class T>
    inline auto from_indices(const std::vector<T>& idx)
    {
        using return_type = xtensor<typename T::value_type, 2>;
        using size_type = typename return_type::size_type;

        if (idx.size() == 0)
        {
            return return_type::from_shape({size_type(0), size_type(0)});
        }

        return_type out = return_type::from_shape({idx.size(), idx[0].size()});

        for (size_type i = 0; i < out.shape()[0]; ++i)
        {
            for (size_type j = 0; j < out.shape()[1]; ++j)
            {
                out(i, j) = idx[i][j];
            }
        }

        return out;
    }

    /**
     * Converts ``std::vector<index_type>`` (returned e.g. from ``xt::argwhere``) to a flattened
     * ``xtensor``.
     *
     * @param idx a vector of indices
     *
     * @return ``xt::xtensor<typename index_type::value_type, 1>`` (e.g. ``xt::xtensor<size_t, 1>``)
     */
    template <class T>
    inline auto flatten_indices(const std::vector<T>& idx)
    {
        auto n = idx.size();
        if (n != 0)
        {
            n *= idx[0].size();
        }

        using return_type = xtensor<typename T::value_type, 1>;
        return_type out = return_type::from_shape({n});
        auto iter = out.begin();
        for_each(
            idx.begin(),
            idx.end(),
            [&iter](const auto& t)
            {
                iter = std::copy(t.cbegin(), t.cend(), iter);
            }
        );

        return out;
    }

    struct ravel_vector_tag;
    struct ravel_tensor_tag;

    namespace detail
    {
        template <class C, class Tag>
        struct ravel_return_type;

        template <class C>
        struct ravel_return_type<C, ravel_vector_tag>
        {
            using index_type = typename C::value_type;
            using value_type = typename index_type::value_type;
            using type = std::vector<value_type>;

            template <class T>
            static std::vector<value_type> init(T n)
            {
                return std::vector<value_type>(n);
            }
        };

        template <class C>
        struct ravel_return_type<C, ravel_tensor_tag>
        {
            using index_type = typename C::value_type;
            using value_type = typename index_type::value_type;
            using type = xt::xtensor<value_type, 1>;

            template <class T>
            static xt::xtensor<value_type, 1> init(T n)
            {
                return xtensor<value_type, 1>::from_shape({n});
            }
        };
    }

    template <class C, class Tag>
    using ravel_return_type_t = typename detail::ravel_return_type<C, Tag>::type;

    /**
     * Converts ``std::vector<index_type>`` (returned e.g. from ``xt::argwhere``) to ``xtensor``
     * whereby the indices are ravelled. For 1-d input there is no conversion.
     *
     * @param idx vector of indices
     * @param shape the shape of the original array
     * @param l the layout type (row-major or column-major)
     *
     * @return ``xt::xtensor<typename index_type::value_type, 1>`` (e.g. ``xt::xtensor<size_t, 1>``)
     */
    template <class Tag = ravel_tensor_tag, class C, class S>
    ravel_return_type_t<C, Tag>
    ravel_indices(const C& idx, const S& shape, layout_type l = layout_type::row_major)
    {
        using return_type = typename detail::ravel_return_type<C, Tag>::type;
        using value_type = typename detail::ravel_return_type<C, Tag>::value_type;
        using strides_type = get_strides_t<S>;
        strides_type strides = xtl::make_sequence<strides_type>(shape.size(), 0);
        compute_strides(shape, l, strides);
        return_type out = detail::ravel_return_type<C, Tag>::init(idx.size());
        auto out_iter = out.begin();
        auto idx_iter = idx.begin();
        for (; out_iter != out.end(); ++out_iter, ++idx_iter)
        {
            *out_iter = element_offset<value_type>(strides, (*idx_iter).cbegin(), (*idx_iter).cend());
        }
        return out;
    }
}

#endif
