/***************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_DYNAMIC_VIEW_HPP
#define XTENSOR_DYNAMIC_VIEW_HPP

#include <variant>

#include <xtl/xsequence.hpp>

#include "../core/xexpression.hpp"
#include "../core/xiterable.hpp"
#include "../core/xlayout.hpp"
#include "../core/xsemantic.hpp"
#include "../views/xstrided_view_base.hpp"

namespace xt
{

    template <class CT, class S, layout_type L, class FST>
    class xdynamic_view;

    template <class CT, class S, layout_type L, class FST>
    struct xcontainer_inner_types<xdynamic_view<CT, S, L, FST>>
    {
        using xexpression_type = std::decay_t<CT>;
        using undecay_expression = CT;
        using reference = inner_reference_t<undecay_expression>;
        using const_reference = typename xexpression_type::const_reference;
        using size_type = typename xexpression_type::size_type;
        using shape_type = std::decay_t<S>;
        using undecay_shape = S;
        using storage_getter = FST;
        using inner_storage_type = typename storage_getter::type;
        using temporary_type = xarray<std::decay_t<typename xexpression_type::value_type>, xexpression_type::static_layout>;
        static constexpr layout_type layout = L;
    };

    template <class CT, class S, layout_type L, class FST>
    struct xiterable_inner_types<xdynamic_view<CT, S, L, FST>>
    {
        using inner_shape_type = S;
        using inner_strides_type = inner_shape_type;
        using inner_backstrides_type = inner_shape_type;

#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ == 8
        static constexpr auto
            random_instantiation_var_for_gcc8_data_iface = has_data_interface<xdynamic_view<CT, S, L, FST>>::value;
        static constexpr auto
            random_instantiation_var_for_gcc8_has_strides = has_strides<xdynamic_view<CT, S, L, FST>>::value;
#endif

        // TODO: implement efficient stepper specific to the dynamic_view
        using const_stepper = xindexed_stepper<const xdynamic_view<CT, S, L, FST>, true>;
        using stepper = xindexed_stepper<xdynamic_view<CT, S, L, FST>, false>;
    };

    /****************************
     * xdynamic_view extensions *
     ****************************/

    namespace extension
    {
        template <class Tag, class CT, class S, layout_type L, class FST>
        struct xdynamic_view_base_impl;

        template <class CT, class S, layout_type L, class FST>
        struct xdynamic_view_base_impl<xtensor_expression_tag, CT, S, L, FST>
        {
            using type = xtensor_empty_base;
        };

        template <class CT, class S, layout_type L, class FST>
        struct xdynamic_view_base : xdynamic_view_base_impl<xexpression_tag_t<CT>, CT, S, L, FST>
        {
        };

        template <class CT, class S, layout_type L, class FST>
        using xdynamic_view_base_t = typename xdynamic_view_base<CT, S, L, FST>::type;
    }

    /*****************
     * xdynamic_view *
     *****************/

    namespace detail
    {
        template <class T>
        class xfake_slice;
    }

    template <class CT, class S, layout_type L = layout_type::dynamic, class FST = detail::flat_storage_getter<CT, XTENSOR_DEFAULT_TRAVERSAL>>
    class xdynamic_view : public xview_semantic<xdynamic_view<CT, S, L, FST>>,
                          public xiterable<xdynamic_view<CT, S, L, FST>>,
                          public extension::xdynamic_view_base_t<CT, S, L, FST>,
                          private xstrided_view_base<xdynamic_view<CT, S, L, FST>>
    {
    public:

        using self_type = xdynamic_view<CT, S, L, FST>;
        using base_type = xstrided_view_base<self_type>;
        using semantic_base = xview_semantic<self_type>;
        using extension_base = extension::xdynamic_view_base_t<CT, S, L, FST>;
        using expression_tag = typename extension_base::expression_tag;

        using xexpression_type = typename base_type::xexpression_type;
        using base_type::is_const;

        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using size_type = typename base_type::size_type;
        using difference_type = typename base_type::difference_type;

        using inner_storage_type = typename base_type::inner_storage_type;
        using storage_type = typename base_type::storage_type;

        using iterable_base = xiterable<self_type>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using inner_strides_type = typename base_type::inner_strides_type;
        using inner_backstrides_type = typename base_type::inner_backstrides_type;

        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        using base_type::contiguous_layout;
        using base_type::static_layout;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        using base_index_type = xindex_type_t<shape_type>;

        using simd_value_type = typename base_type::simd_value_type;
        using bool_load_type = typename base_type::bool_load_type;

        using strides_vt = typename strides_type::value_type;
        using slice_type = std::variant<detail::xfake_slice<strides_vt>, xkeep_slice<strides_vt>, xdrop_slice<strides_vt>>;
        using slice_vector_type = std::vector<slice_type>;

        template <class CTA, class SA>
        xdynamic_view(
            CTA&& e,
            SA&& shape,
            get_strides_t<S>&& strides,
            std::size_t offset,
            layout_type layout,
            slice_vector_type&& slices,
            get_strides_t<S>&& adj_strides
        ) noexcept;

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

        using base_type::dimension;
        using base_type::is_contiguous;
        using base_type::layout;
        using base_type::shape;
        using base_type::size;

        // Explicitly deleting strides method to avoid compilers complaining
        // about not being able to call the strides method from xstrided_view_base
        // private base
        const inner_strides_type& strides() const noexcept = delete;

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

        reference flat(size_type index);
        const_reference flat(size_type index) const;

        using base_type::operator[];
        using base_type::at;
        using base_type::back;
        using base_type::front;
        using base_type::in_bounds;
        using base_type::periodic;

        template <class It>
        reference element(It first, It last);

        template <class It>
        const_reference element(It first, It last) const;

        size_type data_offset() const noexcept;

        // Explicitly deleting data methods so has_data_interface results
        // to false instead of having compilers complaining about not being
        // able to call the methods from the private base
        value_type* data() noexcept = delete;
        const value_type* data() const noexcept = delete;

        using base_type::broadcast_shape;
        using base_type::expression;
        using base_type::storage;

        template <class O>
        bool has_linear_assign(const O& str) const noexcept;

        template <class T>
        void fill(const T& value);

        template <class ST>
        stepper stepper_begin(const ST& shape);
        template <class ST>
        stepper stepper_end(const ST& shape, layout_type l);

        template <class ST>
        const_stepper stepper_begin(const ST& shape) const;
        template <class ST>
        const_stepper stepper_end(const ST& shape, layout_type l) const;

        using container_iterator = std::
            conditional_t<is_const, typename storage_type::const_iterator, typename storage_type::iterator>;
        using const_container_iterator = typename storage_type::const_iterator;

        template <class E>
        using rebind_t = xdynamic_view<E, S, L, typename FST::template rebind_t<E>>;

        template <class E>
        rebind_t<E> build_view(E&& e) const;

    private:

        using offset_type = typename base_type::offset_type;

        slice_vector_type m_slices;
        inner_strides_type m_adj_strides;

        container_iterator data_xbegin() noexcept;
        const_container_iterator data_xbegin() const noexcept;
        container_iterator data_xend(layout_type l, size_type offset) noexcept;
        const_container_iterator data_xend(layout_type l, size_type offset) const noexcept;

        template <class It>
        It data_xbegin_impl(It begin) const noexcept;

        template <class It>
        It data_xend_impl(It end, layout_type l, size_type offset) const noexcept;

        void assign_temporary_impl(temporary_type&& tmp);

        template <class T, class... Args>
        offset_type adjust_offset(offset_type offset, T idx, Args... args) const noexcept;
        offset_type adjust_offset(offset_type offset) const noexcept;

        template <class T, class... Args>
        offset_type
        adjust_offset_impl(offset_type offset, size_type idx_offset, T idx, Args... args) const noexcept;
        offset_type adjust_offset_impl(offset_type offset, size_type idx_offset) const noexcept;

        template <class It>
        offset_type adjust_element_offset(offset_type offset, It first, It last) const noexcept;

        template <class C>
        friend class xstepper;
        friend class xview_semantic<self_type>;
        friend class xaccessible<self_type>;
        friend class xconst_accessible<self_type>;
    };

    /**************************
     * xdynamic_view builders *
     **************************/

    template <class T>
    using xdynamic_slice = std::variant<
        T,

        xrange_adaptor<placeholders::xtuph, T, T>,
        xrange_adaptor<T, placeholders::xtuph, T>,
        xrange_adaptor<T, T, placeholders::xtuph>,

        xrange_adaptor<T, placeholders::xtuph, placeholders::xtuph>,
        xrange_adaptor<placeholders::xtuph, T, placeholders::xtuph>,
        xrange_adaptor<placeholders::xtuph, placeholders::xtuph, T>,

        xrange_adaptor<T, T, T>,
        xrange_adaptor<placeholders::xtuph, placeholders::xtuph, placeholders::xtuph>,

        xrange<T>,
        xstepped_range<T>,

        xkeep_slice<T>,
        xdrop_slice<T>,

        xall_tag,
        xellipsis_tag,
        xnewaxis_tag>;

    using xdynamic_slice_vector = std::vector<xdynamic_slice<std::ptrdiff_t>>;

    template <class E>
    auto dynamic_view(E&& e, const xdynamic_slice_vector& slices);

    /******************************
     * xfake_slice implementation *
     ******************************/

    namespace detail
    {
        template <class T>
        class xfake_slice : public xslice<xfake_slice<T>>
        {
        public:

            using size_type = T;
            using self_type = xfake_slice<T>;

            xfake_slice() = default;

            size_type operator()(size_type /*i*/) const noexcept
            {
                return size_type(0);
            }

            size_type size() const noexcept
            {
                return size_type(1);
            }

            size_type step_size() const noexcept
            {
                return size_type(0);
            }

            size_type step_size(std::size_t /*i*/, std::size_t /*n*/ = 1) const noexcept
            {
                return size_type(0);
            }

            size_type revert_index(std::size_t i) const noexcept
            {
                return i;
            }

            bool contains(size_type /*i*/) const noexcept
            {
                return true;
            }

            bool operator==(const self_type& /*rhs*/) const noexcept
            {
                return true;
            }

            bool operator!=(const self_type& /*rhs*/) const noexcept
            {
                return false;
            }
        };
    }

    /********************************
     * xdynamic_view implementation *
     ********************************/

    template <class CT, class S, layout_type L, class FST>
    template <class CTA, class SA>
    inline xdynamic_view<CT, S, L, FST>::xdynamic_view(
        CTA&& e,
        SA&& shape,
        get_strides_t<S>&& strides,
        std::size_t offset,
        layout_type layout,
        slice_vector_type&& slices,
        get_strides_t<S>&& adj_strides
    ) noexcept
        : base_type(std::forward<CTA>(e), std::forward<SA>(shape), std::move(strides), offset, layout)
        , m_slices(std::move(slices))
        , m_adj_strides(std::move(adj_strides))
    {
    }

    template <class CT, class S, layout_type L, class FST>
    template <class E>
    inline auto xdynamic_view<CT, S, L, FST>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class E>
    inline auto xdynamic_view<CT, S, L, FST>::operator=(const E& e) -> disable_xexpression<E, self_type>&
    {
        std::fill(this->begin(), this->end(), e);
        return *this;
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::operator()() -> reference
    {
        return base_type::storage()[data_offset()];
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::operator()() const -> const_reference
    {
        return base_type::storage()[data_offset()];
    }

    template <class CT, class S, layout_type L, class FST>
    template <class... Args>
    inline auto xdynamic_view<CT, S, L, FST>::operator()(Args... args) -> reference
    {
        XTENSOR_TRY(check_index(base_type::shape(), args...));
        XTENSOR_CHECK_DIMENSION(base_type::shape(), args...);
        offset_type offset = base_type::compute_index(args...);
        offset = adjust_offset(offset, args...);
        return base_type::storage()[static_cast<size_type>(offset)];
    }

    template <class CT, class S, layout_type L, class FST>
    template <class... Args>
    inline auto xdynamic_view<CT, S, L, FST>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_TRY(check_index(base_type::shape(), args...));
        XTENSOR_CHECK_DIMENSION(base_type::shape(), args...);
        offset_type offset = base_type::compute_index(args...);
        offset = adjust_offset(offset, args...);
        return base_type::storage()[static_cast<size_type>(offset)];
    }

    template <class CT, class S, layout_type L, class FST>
    template <class O>
    inline bool xdynamic_view<CT, S, L, FST>::has_linear_assign(const O&) const noexcept
    {
        return false;
    }

    template <class CT, class S, layout_type L, class FST>
    template <class... Args>
    inline auto xdynamic_view<CT, S, L, FST>::unchecked(Args... args) -> reference
    {
        offset_type offset = base_type::compute_unchecked_index(args...);
        offset = adjust_offset(args...);
        return base_type::storage()[static_cast<size_type>(offset)];
    }

    template <class CT, class S, layout_type L, class FST>
    template <class... Args>
    inline auto xdynamic_view<CT, S, L, FST>::unchecked(Args... args) const -> const_reference
    {
        offset_type offset = base_type::compute_unchecked_index(args...);
        offset = adjust_offset(args...);
        return base_type::storage()[static_cast<size_type>(offset)];
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::flat(size_type i) -> reference
    {
        return base_type::storage()[data_offset() + i];
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::flat(size_type i) const -> const_reference
    {
        return base_type::storage()[data_offset() + i];
    }

    template <class CT, class S, layout_type L, class FST>
    template <class It>
    inline auto xdynamic_view<CT, S, L, FST>::element(It first, It last) -> reference
    {
        XTENSOR_TRY(check_element_index(base_type::shape(), first, last));
        offset_type offset = base_type::compute_element_index(first, last);
        offset = adjust_element_offset(offset, first, last);
        return base_type::storage()[static_cast<size_type>(offset)];
    }

    template <class CT, class S, layout_type L, class FST>
    template <class It>
    inline auto xdynamic_view<CT, S, L, FST>::element(It first, It last) const -> const_reference
    {
        XTENSOR_TRY(check_element_index(base_type::shape(), first, last));
        offset_type offset = base_type::compute_element_index(first, last);
        offset = adjust_element_offset(offset, first, last);
        return base_type::storage()[static_cast<size_type>(offset)];
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::data_offset() const noexcept -> size_type
    {
        size_type offset = base_type::data_offset();
        size_type sl_offset = std::visit(
            [](const auto& sl)
            {
                return sl(size_type(0));
            },
            m_slices[0]
        );
        return offset + sl_offset * m_adj_strides[0];
    }

    template <class CT, class S, layout_type L, class FST>
    template <class T>
    inline void xdynamic_view<CT, S, L, FST>::fill(const T& value)
    {
        return std::fill(this->linear_begin(), this->linear_end(), value);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class ST>
    inline auto xdynamic_view<CT, S, L, FST>::stepper_begin(const ST& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, offset);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class ST>
    inline auto xdynamic_view<CT, S, L, FST>::stepper_end(const ST& shape, layout_type /*l*/) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, offset, true);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class ST>
    inline auto xdynamic_view<CT, S, L, FST>::stepper_begin(const ST& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class ST>
    inline auto xdynamic_view<CT, S, L, FST>::stepper_end(const ST& shape, layout_type /*l*/) const
        -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset, true);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class E>
    inline auto xdynamic_view<CT, S, L, FST>::build_view(E&& e) const -> rebind_t<E>
    {
        inner_shape_type sh(this->shape());
        inner_strides_type str(base_type::strides());
        slice_vector_type svt(m_slices);
        inner_strides_type adj_str(m_adj_strides);
        return rebind_t<E>(
            std::forward<E>(e),
            std::move(sh),
            std::move(str),
            base_type::data_offset(),
            this->layout(),
            std::move(svt),
            std::move(adj_str)
        );
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::data_xbegin() noexcept -> container_iterator
    {
        return data_xbegin_impl(this->storage().begin());
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::data_xbegin() const noexcept -> const_container_iterator
    {
        return data_xbegin_impl(this->storage().cbegin());
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::data_xend(layout_type l, size_type offset) noexcept
        -> container_iterator
    {
        return data_xend_impl(this->storage().begin(), l, offset);
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::data_xend(layout_type l, size_type offset) const noexcept
        -> const_container_iterator
    {
        return data_xend_impl(this->storage().cbegin(), l, offset);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class It>
    inline It xdynamic_view<CT, S, L, FST>::data_xbegin_impl(It begin) const noexcept
    {
        return begin + static_cast<std::ptrdiff_t>(data_offset());
    }

    // TODO: fix the data_xend implementation and assign_temporary_impl

    template <class CT, class S, layout_type L, class FST>
    template <class It>
    inline It
    xdynamic_view<CT, S, L, FST>::data_xend_impl(It begin, layout_type l, size_type offset) const noexcept
    {
        return strided_data_end(*this, begin + std::ptrdiff_t(data_offset()), l, offset);
    }

    template <class CT, class S, layout_type L, class FST>
    inline void xdynamic_view<CT, S, L, FST>::assign_temporary_impl(temporary_type&& tmp)
    {
        std::copy(tmp.cbegin(), tmp.cend(), this->begin());
    }

    template <class CT, class S, layout_type L, class FST>
    template <class T, class... Args>
    inline auto
    xdynamic_view<CT, S, L, FST>::adjust_offset(offset_type offset, T idx, Args... args) const noexcept
        -> offset_type
    {
        constexpr size_type nb_args = sizeof...(Args) + 1;
        size_type dim = base_type::dimension();
        offset_type res = nb_args > dim ? adjust_offset(offset, args...)
                                        : adjust_offset_impl(offset, dim - nb_args, idx, args...);
        return res;
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::adjust_offset(offset_type offset) const noexcept -> offset_type
    {
        return offset;
    }

    template <class CT, class S, layout_type L, class FST>
    template <class T, class... Args>
    inline auto
    xdynamic_view<CT, S, L, FST>::adjust_offset_impl(offset_type offset, size_type idx_offset, T idx, Args... args)
        const noexcept -> offset_type
    {
        offset_type sl_offset = std::visit(
            [idx](const auto& sl)
            {
                using type = typename std::decay_t<decltype(sl)>::size_type;
                return sl(type(idx));
            },
            m_slices[idx_offset]
        );
        offset_type res = offset + sl_offset * m_adj_strides[idx_offset];
        return adjust_offset_impl(res, idx_offset + 1, args...);
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::adjust_offset_impl(offset_type offset, size_type) const noexcept
        -> offset_type
    {
        return offset;
    }

    template <class CT, class S, layout_type L, class FST>
    template <class It>
    inline auto
    xdynamic_view<CT, S, L, FST>::adjust_element_offset(offset_type offset, It first, It last) const noexcept
        -> offset_type
    {
        auto dst = std::distance(first, last);
        offset_type dim = static_cast<offset_type>(dimension());
        offset_type loop_offset = dst < dim ? dim - dst : offset_type(0);
        offset_type idx_offset = dim < dst ? dst - dim : offset_type(0);
        offset_type res = offset;
        for (offset_type i = loop_offset; i < dim; ++i, ++first)
        {
            offset_type j = static_cast<offset_type>(first[idx_offset]);
            offset_type sl_offset = std::visit(
                [j](const auto& sl)
                {
                    return static_cast<offset_type>(sl(j));
                },
                m_slices[static_cast<std::size_t>(i)]
            );
            res += sl_offset * m_adj_strides[static_cast<std::size_t>(i)];
        }
        return res;
    }

    /*****************************************
     * xdynamic_view builders implementation *
     *****************************************/

    namespace detail
    {
        template <class V>
        struct adj_strides_policy
        {
            using slice_vector = V;
            using strides_type = dynamic_shape<std::ptrdiff_t>;

            slice_vector new_slices;
            strides_type new_adj_strides;

        protected:

            inline void resize(std::size_t size)
            {
                new_slices.resize(size);
                new_adj_strides.resize(size);
            }

            inline void set_fake_slice(std::size_t idx)
            {
                new_slices[idx] = xfake_slice<std::ptrdiff_t>();
                new_adj_strides[idx] = std::ptrdiff_t(0);
            }

            template <class ST, class S>
            bool fill_args(
                const xdynamic_slice_vector& slices,
                std::size_t sl_idx,
                std::size_t i,
                std::size_t old_shape,
                const ST& old_stride,
                S& shape,
                get_strides_t<S>& strides
            )
            {
                return fill_args_impl<xkeep_slice<std::ptrdiff_t>>(
                           slices,
                           sl_idx,
                           i,
                           old_shape,
                           old_stride,
                           shape,
                           strides
                       )
                       || fill_args_impl<xdrop_slice<std::ptrdiff_t>>(
                           slices,
                           sl_idx,
                           i,
                           old_shape,
                           old_stride,
                           shape,
                           strides
                       );
            }

            template <class SL, class ST, class S>
            bool fill_args_impl(
                const xdynamic_slice_vector& slices,
                std::size_t sl_idx,
                std::size_t i,
                std::size_t old_shape,
                const ST& old_stride,
                S& shape,
                get_strides_t<S>& strides
            )
            {
                auto* sl = std::get_if<SL>(&slices[sl_idx]);
                if (sl != nullptr)
                {
                    new_slices[i] = *sl;
                    auto& ns = std::get<SL>(new_slices[i]);
                    ns.normalize(old_shape);
                    shape[i] = static_cast<std::size_t>(ns.size());
                    strides[i] = std::ptrdiff_t(0);
                    new_adj_strides[i] = static_cast<std::ptrdiff_t>(old_stride);
                }
                return sl != nullptr;
            }
        };
    }

    template <class E>
    inline auto dynamic_view(E&& e, const xdynamic_slice_vector& slices)
    {
        using view_type = xdynamic_view<xclosure_t<E>, dynamic_shape<std::size_t>>;
        using slice_vector = typename view_type::slice_vector_type;
        using policy = detail::adj_strides_policy<slice_vector>;
        detail::strided_view_args<policy> args;
        args.fill_args(
            e.shape(),
            detail::get_strides<XTENSOR_DEFAULT_TRAVERSAL>(e),
            detail::get_offset<XTENSOR_DEFAULT_TRAVERSAL>(e),
            e.layout(),
            slices
        );
        return view_type(
            std::forward<E>(e),
            std::move(args.new_shape),
            std::move(args.new_strides),
            args.new_offset,
            args.new_layout,
            std::move(args.new_slices),
            std::move(args.new_adj_strides)
        );
    }
}

#endif
