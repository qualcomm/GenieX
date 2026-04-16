/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_VIEW_HPP
#define XTENSOR_VIEW_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <xtl/xclosure.hpp>
#include <xtl/xmeta_utils.hpp>
#include <xtl/xsequence.hpp>
#include <xtl/xtype_traits.hpp>

#include "../containers/xarray.hpp"
#include "../containers/xcontainer.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xaccessible.hpp"
#include "../core/xiterable.hpp"
#include "../core/xsemantic.hpp"
#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../views/xbroadcast.hpp"
#include "../views/xslice.hpp"
#include "../views/xview_utils.hpp"

namespace xt
{

    /*******************
     * xview extension *
     *******************/

    namespace extension
    {
        template <class Tag, class CT, class... S>
        struct xview_base_impl;

        template <class CT, class... S>
        struct xview_base_impl<xtensor_expression_tag, CT, S...>
        {
            using type = xtensor_empty_base;
        };

        template <class CT, class... S>
        struct xview_base : xview_base_impl<xexpression_tag_t<CT>, CT, S...>
        {
        };

        template <class CT, class... S>
        using xview_base_t = typename xview_base<CT, S...>::type;
    }

    /*********************
     * xview declaration *
     *********************/

    template <bool is_const, class CT, class... S>
    class xview_stepper;

    template <class ST, class... S>
    struct xview_shape_type;

    namespace detail
    {

        template <class T>
        struct is_xrange : std::false_type
        {
        };

        template <class T>
        struct is_xrange<xrange<T>> : std::true_type
        {
        };

        template <class S>
        struct is_xall_slice : std::false_type
        {
        };

        template <class T>
        struct is_xall_slice<xall<T>> : std::true_type
        {
        };

        template <layout_type L, bool valid, bool all_seen, bool range_seen, class V>
        struct is_contiguous_view_impl
        {
            static constexpr bool value = false;
        };

        template <class T>
        struct static_dimension
        {
            static constexpr std::ptrdiff_t value = -1;
        };

        template <class T, std::size_t N>
        struct static_dimension<std::array<T, N>>
        {
            static constexpr std::ptrdiff_t value = static_cast<std::ptrdiff_t>(N);
        };

        template <class T, std::size_t N>
        struct static_dimension<xt::const_array<T, N>>
        {
            static constexpr std::ptrdiff_t value = static_cast<std::ptrdiff_t>(N);
        };

        template <std::size_t... I>
        struct static_dimension<xt::fixed_shape<I...>>
        {
            static constexpr std::ptrdiff_t value = sizeof...(I);
        };

        // if we have the same number of integers as we have static dimensions
        // this can be interpreted like a xscalar
        template <class CT, class... S>
        struct is_xscalar_impl<xview<CT, S...>>
        {
            static constexpr bool value = static_cast<std::ptrdiff_t>(integral_count<S...>()
                                          ) == static_dimension<typename std::decay_t<CT>::shape_type>::value
                                              ? true
                                              : false;
        };

        template <class S>
        struct is_strided_slice_impl : std::true_type
        {
        };

        template <class T>
        struct is_strided_slice_impl<xkeep_slice<T>> : std::false_type
        {
        };

        template <class T>
        struct is_strided_slice_impl<xdrop_slice<T>> : std::false_type
        {
        };

        // If we have no discontiguous slices, we can calculate strides for this view.
        template <class E, class... S>
        struct is_strided_view
            : std::integral_constant<
                  bool,
                  std::conjunction<has_data_interface<E>, is_strided_slice_impl<std::decay_t<S>>...>::value>
        {
        };

        // if row major the view can only be (statically) computed as contiguous if:
        // any number of integers is followed by either one or no range which
        // are followed by explicit (or implicit) all's
        //
        // e.g.
        //      (i, j, all(), all()) == contiguous
        //      (i, range(0, 2), all()) == contiguous
        //      (i) == contiguous (implicit all slices)
        //      (i, all(), j) == *not* contiguous
        //      (i, range(0, 2), range(0, 2)) == *not* contiguous etc.
        template <bool valid, bool all_seen, bool range_seen, class V>
        struct is_contiguous_view_impl<layout_type::row_major, valid, all_seen, range_seen, V>
        {
            using slice = xtl::mpl::front_t<V>;
            static constexpr bool is_range_slice = is_xrange<slice>::value;
            static constexpr bool is_int_slice = xtl::is_integral<slice>::value;
            static constexpr bool is_all_slice = is_xall_slice<slice>::value;
            static constexpr bool have_all_seen = all_seen || is_all_slice;
            static constexpr bool have_range_seen = is_range_slice;

            static constexpr bool is_valid = valid
                                             && (have_all_seen
                                                     ? is_all_slice
                                                     : (!range_seen && (is_int_slice || is_range_slice)));

            static constexpr bool value = is_contiguous_view_impl < layout_type::row_major, is_valid,
                                  have_all_seen, range_seen || is_range_slice,
                                  xtl::mpl::pop_front_t < V >> ::value;
        };

        template <bool valid, bool all_seen, bool range_seen>
        struct is_contiguous_view_impl<layout_type::row_major, valid, all_seen, range_seen, xtl::mpl::vector<>>
        {
            static constexpr bool value = valid;
        };

        // For column major the *same* but reverse is true -- with the additional
        // constraint that we have to know the dimension at compile time otherwise
        // we cannot make the decision as there might be implicit all's following.
        template <bool valid, bool int_seen, bool range_seen, class V>
        struct is_contiguous_view_impl<layout_type::column_major, valid, int_seen, range_seen, V>
        {
            using slice = xtl::mpl::front_t<V>;
            static constexpr bool is_range_slice = is_xrange<slice>::value;
            static constexpr bool is_int_slice = xtl::is_integral<slice>::value;
            static constexpr bool is_all_slice = is_xall_slice<slice>::value;

            static constexpr bool have_int_seen = int_seen || is_int_slice;

            static constexpr bool is_valid = valid
                                             && (have_int_seen
                                                     ? is_int_slice
                                                     : (!range_seen && (is_all_slice || is_range_slice)));
            static constexpr bool value = is_contiguous_view_impl < layout_type::column_major, is_valid,
                                  have_int_seen, is_range_slice || range_seen,
                                  xtl::mpl::pop_front_t < V >> ::value;
        };

        template <bool valid, bool int_seen, bool range_seen>
        struct is_contiguous_view_impl<layout_type::column_major, valid, int_seen, range_seen, xtl::mpl::vector<>>
        {
            static constexpr bool value = valid;
        };

        // TODO relax has_data_interface constraint here!
        template <class E, class... S>
        struct is_contiguous_view
            : std::integral_constant<
                  bool,
                  has_data_interface<E>::value
                      && !(
                          E::static_layout == layout_type::column_major
                          && static_cast<std::size_t>(static_dimension<typename E::shape_type>::value) != sizeof...(S)
                      )
                      && is_contiguous_view_impl<E::static_layout, true, false, false, xtl::mpl::vector<S...>>::value>
        {
        };

        template <layout_type L, class T, std::ptrdiff_t offset>
        struct unwrap_offset_container
        {
            using type = void;
        };

        template <class T, std::ptrdiff_t offset>
        struct unwrap_offset_container<layout_type::row_major, T, offset>
        {
            using type = sequence_view<T, offset, static_dimension<T>::value>;
        };

        template <class T, std::ptrdiff_t start, std::ptrdiff_t end, std::ptrdiff_t offset>
        struct unwrap_offset_container<layout_type::row_major, sequence_view<T, start, end>, offset>
        {
            using type = sequence_view<T, start + offset, end>;
        };

        template <class T, std::ptrdiff_t offset>
        struct unwrap_offset_container<layout_type::column_major, T, offset>
        {
            using type = sequence_view<T, 0, static_dimension<T>::value - offset>;
        };

        template <class T, std::ptrdiff_t start, std::ptrdiff_t end, std::ptrdiff_t offset>
        struct unwrap_offset_container<layout_type::column_major, sequence_view<T, start, end>, offset>
        {
            using type = sequence_view<T, start, end - offset>;
        };

        template <class E, class... S>
        struct get_contigous_shape_type
        {
            // if we have no `range` in the slices we can re-use the shape with an offset
            using type = std::conditional_t<
                std::disjunction<is_xrange<S>...>::value,
                typename xview_shape_type<typename E::shape_type, S...>::type,
                // In the false branch we know that we have only integers at the front OR end, and NO range
                typename unwrap_offset_container<E::static_layout, typename E::inner_shape_type, integral_count<S...>()>::type>;
        };

        template <class T>
        struct is_sequence_view : std::integral_constant<bool, false>
        {
        };

        template <class T, std::ptrdiff_t S, std::ptrdiff_t E>
        struct is_sequence_view<sequence_view<T, S, E>> : std::integral_constant<bool, true>
        {
        };
    }

    template <class E, class... S>
    concept contiguous_view_concept = detail::is_contiguous_view<E, S...>::value;
    template <class E, class... S>
    concept strided_view_concept = detail::is_strided_view<std::decay_t<E>, S...>::value;

    template <class CT, class... S>
    struct xcontainer_inner_types<xview<CT, S...>>
    {
        using xexpression_type = std::decay_t<CT>;
        using reference = inner_reference_t<CT>;
        using const_reference = typename xexpression_type::const_reference;
        using size_type = typename xexpression_type::size_type;
        using temporary_type = view_temporary_type_t<xexpression_type, S...>;

        static constexpr layout_type layout = detail::is_contiguous_view<xexpression_type, S...>::value
                                                  ? xexpression_type::static_layout
                                                  : layout_type::dynamic;

        static constexpr bool is_const = std::is_const<std::remove_reference_t<CT>>::value;

        using extract_storage_type = xtl::mpl::eval_if_t<
            has_data_interface<xexpression_type>,
            detail::expr_storage_type<xexpression_type>,
            make_invalid_type<>>;
        using storage_type = std::conditional_t<is_const, const extract_storage_type, extract_storage_type>;
    };

    template <class CT, class... S>
    struct xiterable_inner_types<xview<CT, S...>>
    {
        using xexpression_type = std::decay_t<CT>;

        static constexpr bool is_strided_view = detail::is_strided_view<xexpression_type, S...>::value;
        static constexpr bool is_contiguous_view = detail::is_contiguous_view<xexpression_type, S...>::value;

        using inner_shape_type = std::conditional_t<
            is_contiguous_view,
            typename detail::get_contigous_shape_type<xexpression_type, S...>::type,
            typename xview_shape_type<typename xexpression_type::shape_type, S...>::type>;

        using stepper = std::conditional_t<
            is_strided_view,
            xstepper<xview<CT, S...>>,
            xview_stepper<std::is_const<std::remove_reference_t<CT>>::value, CT, S...>>;

        using const_stepper = std::conditional_t<
            is_strided_view,
            xstepper<const xview<CT, S...>>,
            xview_stepper<true, std::remove_cv_t<CT>, S...>>;
    };

    /**
     * @class xview
     * @brief Multidimensional view with tensor semantic.
     *
     * The xview class implements a multidimensional view with tensor
     * semantic. It is used to adapt the shape of an xexpression without
     * changing it. xview is not meant to be used directly, but
     * only with the \ref view helper functions.
     *
     * @tparam CT the closure type of the \ref xexpression to adapt
     * @tparam S the slices type describing the shape adaptation
     *
     * @sa view, range, all, newaxis, keep, drop
     */
    template <class CT, class... S>
    class xview : public xview_semantic<xview<CT, S...>>,
                  public std::conditional_t<
                      detail::is_contiguous_view<std::decay_t<CT>, S...>::value,
                      xcontiguous_iterable<xview<CT, S...>>,
                      xiterable<xview<CT, S...>>>,
                  public xaccessible<xview<CT, S...>>,
                  public extension::xview_base_t<CT, S...>
    {
    public:

        using self_type = xview<CT, S...>;
        using inner_types = xcontainer_inner_types<self_type>;
        using xexpression_type = std::decay_t<CT>;
        using semantic_base = xview_semantic<self_type>;
        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;

        using accessible_base = xaccessible<self_type>;
        using extension_base = extension::xview_base_t<CT, S...>;
        using expression_tag = typename extension_base::expression_tag;

        static constexpr bool is_const = std::is_const<std::remove_reference_t<CT>>::value;
        using value_type = typename xexpression_type::value_type;
        using simd_value_type = xt_simd::simd_type<value_type>;
        using bool_load_type = typename xexpression_type::bool_load_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = std::
            conditional_t<is_const, typename xexpression_type::const_pointer, typename xexpression_type::pointer>;
        using const_pointer = typename xexpression_type::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename xexpression_type::difference_type;

        static constexpr layout_type static_layout = inner_types::layout;
        static constexpr bool contiguous_layout = static_layout != layout_type::dynamic;

        static constexpr bool is_strided_view = detail::is_strided_view<xexpression_type, S...>::value;
        static constexpr bool is_contiguous_view = contiguous_layout;

        using iterable_base = xiterable<self_type>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using shape_type = typename xview_shape_type<typename xexpression_type::shape_type, S...>::type;

        using xexpression_inner_strides_type = xtl::mpl::eval_if_t<
            has_strides<xexpression_type>,
            detail::expr_inner_strides_type<xexpression_type>,
            get_strides_type<shape_type>>;

        using xexpression_inner_backstrides_type = xtl::mpl::eval_if_t<
            has_strides<xexpression_type>,
            detail::expr_inner_backstrides_type<xexpression_type>,
            get_strides_type<shape_type>>;

        using storage_type = typename inner_types::storage_type;

        static constexpr bool has_trivial_strides = is_contiguous_view
                                                    && !std::disjunction<detail::is_xrange<S>...>::value;
        using inner_strides_type = std::conditional_t<
            has_trivial_strides,
            typename detail::unwrap_offset_container<
                xexpression_type::static_layout,
                xexpression_inner_strides_type,
                integral_count<S...>()>::type,
            get_strides_t<shape_type>>;

        using inner_backstrides_type = std::conditional_t<
            has_trivial_strides,
            typename detail::unwrap_offset_container<
                xexpression_type::static_layout,
                xexpression_inner_backstrides_type,
                integral_count<S...>()>::type,
            get_strides_t<shape_type>>;

        using strides_type = get_strides_t<shape_type>;
        using backstrides_type = strides_type;


        using slice_type = std::tuple<S...>;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        using linear_iterator = std::conditional_t<
            has_data_interface<xexpression_type>::value && is_strided_view,
            std::conditional_t<is_const, typename xexpression_type::const_linear_iterator, typename xexpression_type::linear_iterator>,
            typename iterable_base::linear_iterator>;
        using const_linear_iterator = std::conditional_t<
            has_data_interface<xexpression_type>::value && is_strided_view,
            typename xexpression_type::const_linear_iterator,
            typename iterable_base::const_linear_iterator>;

        using reverse_linear_iterator = std::reverse_iterator<linear_iterator>;
        using const_reverse_linear_iterator = std::reverse_iterator<const_linear_iterator>;

        using container_iterator = pointer;
        using const_container_iterator = const_pointer;
        static constexpr std::size_t rank = SIZE_MAX;

        // The FSL argument prevents the compiler from calling this constructor
        // instead of the copy constructor when sizeof...(SL) == 0.
        template <class CTA, class FSL, class... SL>
        explicit xview(CTA&& e, FSL&& first_slice, SL&&... slices) noexcept;

        xview(const xview&) = default;
        self_type& operator=(const xview& rhs);

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

        const inner_shape_type& shape() const noexcept;
        const slice_type& slices() const noexcept;
        layout_type layout() const noexcept;
        bool is_contiguous() const noexcept;
        using accessible_base::shape;

        template <class T>
        void fill(const T& value);

        template <class... Args>
        reference operator()(Args... args);
        template <class... Args>
        reference unchecked(Args... args);
        template <class It>
        reference element(It first, It last);

        template <class... Args>
        const_reference operator()(Args... args) const;
        template <class... Args>
        const_reference unchecked(Args... args) const;
        template <class It>
        const_reference element(It first, It last) const;

        xexpression_type& expression() noexcept;
        const xexpression_type& expression() const noexcept;

        template <class ST>
        bool broadcast_shape(ST& shape, bool reuse_cache = false) const;

        template <class ST>
        bool has_linear_assign(const ST& strides) const;

        template <class ST>
        stepper stepper_begin(const ST& shape);
        template <class ST>
        stepper stepper_end(const ST& shape, layout_type l);

        template <class ST>
        const_stepper stepper_begin(const ST& shape) const;
        template <class ST>
        const_stepper stepper_end(const ST& shape, layout_type l) const;

        template <class T = xexpression_type>
        storage_type& storage()
            requires(has_data_interface_concept<T>);

        template <class T = xexpression_type>
        const storage_type& storage() const
            requires(has_data_interface_concept<T>);

        template <class T = xexpression_type>
        linear_iterator linear_begin()
            requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        linear_iterator linear_end()
            requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        const_linear_iterator linear_begin() const
            requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        const_linear_iterator linear_end() const
            requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        const_linear_iterator linear_cbegin() const
            requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        const_linear_iterator linear_cend() const
            requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        reverse_linear_iterator linear_rbegin()
            requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        reverse_linear_iterator linear_rend()
            requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        const_reverse_linear_iterator linear_rbegin() const
            requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        const_reverse_linear_iterator linear_rend() const
            requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        const_reverse_linear_iterator linear_crbegin() const
            requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        const_reverse_linear_iterator linear_crend() const
            requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        const inner_strides_type& strides() const
            requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        const inner_strides_type& backstrides() const
            requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        const_pointer data() const
            requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        pointer data()
            requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        std::size_t data_offset() const noexcept
            requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class It>
        inline It data_xbegin_impl(It begin) const noexcept;

        template <class It>
        inline It data_xend_impl(It begin, layout_type l, size_type offset) const noexcept;
        inline container_iterator data_xbegin() noexcept;
        inline const_container_iterator data_xbegin() const noexcept;
        inline container_iterator data_xend(layout_type l, size_type offset) noexcept;

        inline const_container_iterator data_xend(layout_type l, size_type offset) const noexcept;

        // Conversion operator enabled for statically "scalar" views
        template <xscalar_concept ST = self_type>
        operator reference()
        {
            return (*this)();
        }

        template <xscalar_concept ST = self_type>
        operator const_reference() const
        {
            return (*this)();
        }

        size_type underlying_size(size_type dim) const;

        xtl::xclosure_pointer<self_type&> operator&() &;
        xtl::xclosure_pointer<const self_type&> operator&() const&;
        xtl::xclosure_pointer<self_type> operator&() &&;

        template <class E, class T = xexpression_type>
        void assign_to(xexpression<E>& e, bool force_resize) const
            requires(has_data_interface_concept<T> and contiguous_view_concept<E, S...>);

        template <class E>
        using rebind_t = xview<E, S...>;

        template <class E>
        rebind_t<E> build_view(E&& e) const;

        //
        // SIMD interface
        //

        template <class requested_type>
        using simd_return_type = xt_simd::simd_return_type<value_type, requested_type>;

        template <class align, class simd, class T = xexpression_type>
        void store_simd(size_type i, const simd& e)
            requires(has_simd_interface_concept<T> and strided_view_concept<CT, S...>);

        template <
            class align,
            class requested_type = value_type,
            std::size_t N = xt_simd::simd_traits<requested_type>::size,
            class T = xexpression_type>
        simd_return_type<requested_type> load_simd(size_type i) const
            requires(has_simd_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        reference data_element(size_type i)
            requires(has_simd_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        const_reference data_element(size_type i) const
            requires(has_simd_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        reference flat(size_type i)
            requires(has_simd_interface_concept<T> and strided_view_concept<CT, S...>);

        template <class T = xexpression_type>
        const_reference flat(size_type i) const
            requires(has_simd_interface_concept<T> and strided_view_concept<CT, S...>);

    private:

        // VS 2015 workaround (yes, really)
        template <std::size_t I>
        struct lesser_condition
        {
            static constexpr bool value = (I + newaxis_count_before<S...>(I + 1) < sizeof...(S));
        };

        CT m_e;
        slice_type m_slices;
        inner_shape_type m_shape;
        mutable inner_strides_type m_strides;
        mutable inner_backstrides_type m_backstrides;
        mutable std::size_t m_data_offset;
        mutable bool m_strides_computed;

        template <class CTA, class FSL, class... SL>
        explicit xview(std::true_type, CTA&& e, FSL&& first_slice, SL&&... slices) noexcept;

        template <class CTA, class FSL, class... SL>
        explicit xview(std::false_type, CTA&& e, FSL&& first_slice, SL&&... slices) noexcept;

        template <class... Args>
        auto make_index_sequence(Args... args) const noexcept;

        void compute_strides(std::true_type) const;
        void compute_strides(std::false_type) const;

        reference access();

        template <class Arg, class... Args>
        reference access(Arg arg, Args... args);

        const_reference access() const;

        template <class Arg, class... Args>
        const_reference access(Arg arg, Args... args) const;

        template <typename std::decay_t<CT>::size_type... I, class... Args>
        reference unchecked_impl(std::index_sequence<I...>, Args... args);

        template <typename std::decay_t<CT>::size_type... I, class... Args>
        const_reference unchecked_impl(std::index_sequence<I...>, Args... args) const;

        template <typename std::decay_t<CT>::size_type... I, class... Args>
        reference access_impl(std::index_sequence<I...>, Args... args);

        template <typename std::decay_t<CT>::size_type... I, class... Args>
        const_reference access_impl(std::index_sequence<I...>, Args... args) const;

        template <typename std::decay_t<CT>::size_type I, class... Args>
        size_type index(Args... args) const;

        template <typename std::decay_t<CT>::size_type, class T>
        size_type sliced_access(const xslice<T>& slice) const;

        template <typename std::decay_t<CT>::size_type I, class T, class Arg, class... Args>
        size_type sliced_access(const xslice<T>& slice, Arg arg, Args... args) const;

        template <typename std::decay_t<CT>::size_type I, class T, class... Args>
        disable_xslice<T, size_type> sliced_access(const T& squeeze, Args...) const;

        using base_index_type = xindex_type_t<typename xexpression_type::shape_type>;

        template <class It>
        base_index_type make_index(It first, It last) const;

        void assign_temporary_impl(temporary_type&& tmp);

        template <std::size_t... I>
        std::size_t data_offset_impl(std::index_sequence<I...>) const noexcept;

        template <std::size_t... I>
        auto compute_strides_impl(std::index_sequence<I...>) const noexcept;

        inner_shape_type compute_shape(std::true_type) const;
        inner_shape_type compute_shape(std::false_type) const;

        template <class E, std::size_t... I>
        rebind_t<E> build_view_impl(E&& e, std::index_sequence<I...>) const;

        friend class xview_semantic<xview<CT, S...>>;
    };

    template <class E, class... S>
    auto view(E&& e, S&&... slices);

    template <class E>
    auto row(E&& e, std::ptrdiff_t index);

    template <class E>
    auto col(E&& e, std::ptrdiff_t index);

    /*****************************
     * xview_stepper declaration *
     *****************************/

    namespace detail
    {
        template <class V>
        struct get_stepper_impl
        {
            using xexpression_type = typename V::xexpression_type;
            using type = typename xexpression_type::stepper;
        };

        template <class V>
        struct get_stepper_impl<const V>
        {
            using xexpression_type = typename V::xexpression_type;
            using type = typename xexpression_type::const_stepper;
        };
    }

    template <class V>
    using get_stepper = typename detail::get_stepper_impl<V>::type;

    template <bool is_const, class CT, class... S>
    class xview_stepper
    {
    public:

        using view_type = std::conditional_t<is_const, const xview<CT, S...>, xview<CT, S...>>;
        using substepper_type = get_stepper<view_type>;

        using value_type = typename substepper_type::value_type;
        using reference = typename substepper_type::reference;
        using pointer = typename substepper_type::pointer;
        using difference_type = typename substepper_type::difference_type;
        using size_type = typename view_type::size_type;

        using shape_type = typename substepper_type::shape_type;

        xview_stepper() = default;
        xview_stepper(
            view_type* view,
            substepper_type it,
            size_type offset,
            bool end = false,
            layout_type l = XTENSOR_DEFAULT_TRAVERSAL
        );

        reference operator*() const;

        void step(size_type dim);
        void step_back(size_type dim);
        void step(size_type dim, size_type n);
        void step_back(size_type dim, size_type n);
        void reset(size_type dim);
        void reset_back(size_type dim);

        void to_begin();
        void to_end(layout_type l);

    private:

        bool is_newaxis_slice(size_type index) const noexcept;
        void to_end_impl(layout_type l);

        template <class F>
        void common_step_forward(size_type dim, F f);
        template <class F>
        void common_step_backward(size_type dim, F f);

        template <class F>
        void common_step_forward(size_type dim, size_type n, F f);
        template <class F>
        void common_step_backward(size_type dim, size_type n, F f);

        template <class F>
        void common_reset(size_type dim, F f, bool backwards);

        view_type* p_view;
        substepper_type m_it;
        size_type m_offset;
        std::array<std::size_t, sizeof...(S)> m_index_keeper;
    };

    // meta-function returning the shape type for an xview
    template <class ST, class... S>
    struct xview_shape_type
    {
        using type = ST;
    };

    template <class I, std::size_t L, class... S>
    struct xview_shape_type<std::array<I, L>, S...>
    {
        using type = std::array<I, L - integral_count<S...>() + newaxis_count<S...>()>;
    };

    template <std::size_t... I, class... S>
    struct xview_shape_type<fixed_shape<I...>, S...>
    {
        using type = typename xview_shape_type<std::array<std::size_t, sizeof...(I)>, S...>::type;
    };

    /************************
     * xview implementation *
     ************************/

    /**
     * @name Constructor
     */

    //@{
    /**
     * Constructs a view on the specified xexpression.
     * Users should not call directly this constructor but
     * use the view function instead.
     * @param e the xexpression to adapt
     * @param first_slice the first slice describing the view
     * @param slices the slices list describing the view
     * @sa view
     */
    template <class CT, class... S>
    template <class CTA, class FSL, class... SL>
    xview<CT, S...>::xview(CTA&& e, FSL&& first_slice, SL&&... slices) noexcept
        : xview(
            std::integral_constant<bool, has_trivial_strides>{},
            std::forward<CTA>(e),
            std::forward<FSL>(first_slice),
            std::forward<SL>(slices)...
        )
    {
    }

    // trivial strides initializer
    template <class CT, class... S>
    template <class CTA, class FSL, class... SL>
    xview<CT, S...>::xview(std::true_type, CTA&& e, FSL&& first_slice, SL&&... slices) noexcept
        : m_e(std::forward<CTA>(e))
        , m_slices(std::forward<FSL>(first_slice), std::forward<SL>(slices)...)
        , m_shape(compute_shape(detail::is_sequence_view<inner_shape_type>{}))
        , m_strides(m_e.strides())
        , m_backstrides(m_e.backstrides())
        , m_data_offset(data_offset_impl(std::make_index_sequence<sizeof...(S)>()))
        , m_strides_computed(true)
    {
    }

    template <class CT, class... S>
    template <class CTA, class FSL, class... SL>
    xview<CT, S...>::xview(std::false_type, CTA&& e, FSL&& first_slice, SL&&... slices) noexcept
        : m_e(std::forward<CTA>(e))
        , m_slices(std::forward<FSL>(first_slice), std::forward<SL>(slices)...)
        , m_shape(compute_shape(std::false_type{}))
        , m_strides_computed(false)
    {
    }

    //@}

    template <class CT, class... S>
    inline auto xview<CT, S...>::operator=(const xview& rhs) -> self_type&
    {
        temporary_type tmp(rhs);
        return this->assign_temporary(std::move(tmp));
    }

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class CT, class... S>
    template <class E>
    inline auto xview<CT, S...>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }

    //@}

    template <class CT, class... S>
    template <class E>
    inline auto xview<CT, S...>::operator=(const E& e) -> disable_xexpression<E, self_type>&
    {
        this->fill(e);
        return *this;
    }

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the shape of the view.
     */
    template <class CT, class... S>
    inline auto xview<CT, S...>::shape() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    /**
     * Returns the slices of the view.
     */
    template <class CT, class... S>
    inline auto xview<CT, S...>::slices() const noexcept -> const slice_type&
    {
        return m_slices;
    }

    /**
     * Returns the slices of the view.
     */
    template <class CT, class... S>
    inline layout_type xview<CT, S...>::layout() const noexcept
    {
        if constexpr (is_strided_view)
        {
            if (static_layout != layout_type::dynamic)
            {
                return static_layout;
            }
            else
            {
                bool strides_match = do_strides_match(shape(), strides(), m_e.layout(), true);
                return strides_match ? m_e.layout() : layout_type::dynamic;
            }
        }
        else
        {
            return layout_type::dynamic;
        }
    }

    template <class CT, class... S>
    inline bool xview<CT, S...>::is_contiguous() const noexcept
    {
        return layout() != layout_type::dynamic;
    }

    //@}

    /**
     * @name Data
     */
    //@{

    /**
     * Fills the view with the given value.
     * @param value the value to fill the view with.
     */
    template <class CT, class... S>
    template <class T>
    inline void xview<CT, S...>::fill(const T& value)
    {
        if constexpr (static_layout != layout_type::dynamic)
        {
            std::fill(linear_begin(), linear_end(), value);
        }
        else
        {
            std::fill(this->begin(), this->end(), value);
        }
    }

    /**
     * Returns a reference to the element at the specified position in the view.
     * @param args a list of indices specifying the position in the view. Indices
     * must be unsigned integers, the number of indices should be equal or greater
     * than the number of dimensions of the view.
     */
    template <class CT, class... S>
    template <class... Args>
    inline auto xview<CT, S...>::operator()(Args... args) -> reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        // The static cast prevents the compiler from instantiating the template methods with signed integers,
        // leading to warning about signed/unsigned conversions in the deeper layers of the access methods
        return access(static_cast<size_type>(args)...);
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
     * double res = fd.unchecked(0, 1);
     * @endcode
     */
    template <class CT, class... S>
    template <class... Args>
    inline auto xview<CT, S...>::unchecked(Args... args) -> reference
    {
        return unchecked_impl(make_index_sequence(args...), static_cast<size_type>(args)...);
    }

    template <class CT, class... S>
    template <class It>
    inline auto xview<CT, S...>::element(It first, It last) -> reference
    {
        XTENSOR_TRY(check_element_index(shape(), first, last));
        // TODO: avoid memory allocation
        auto index = make_index(first, last);
        return m_e.element(index.cbegin(), index.cend());
    }

    /**
     * Returns a constant reference to the element at the specified position in the view.
     * @param args a list of indices specifying the position in the view. Indices must be
     * unsigned integers, the number of indices should be equal or greater than the number
     * of dimensions of the view.
     */
    template <class CT, class... S>
    template <class... Args>
    inline auto xview<CT, S...>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        // The static cast prevents the compiler from instantiating the template methods with signed integers,
        // leading to warning about signed/unsigned conversions in the deeper layers of the access methods
        return access(static_cast<size_type>(args)...);
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
     * double res = fd.unchecked(0, 1);
     * @endcode
     */
    template <class CT, class... S>
    template <class... Args>
    inline auto xview<CT, S...>::unchecked(Args... args) const -> const_reference
    {
        return unchecked_impl(make_index_sequence(args...), static_cast<size_type>(args)...);
    }

    template <class CT, class... S>
    template <class It>
    inline auto xview<CT, S...>::element(It first, It last) const -> const_reference
    {
        // TODO: avoid memory allocation
        auto index = make_index(first, last);
        return m_e.element(index.cbegin(), index.cend());
    }

    /**
     * Returns a reference to the underlying expression of the view.
     */
    template <class CT, class... S>
    inline auto xview<CT, S...>::expression() noexcept -> xexpression_type&
    {
        return m_e;
    }

    /**
     * Returns a const reference to the underlying expression of the view.
     */
    template <class CT, class... S>
    inline auto xview<CT, S...>::expression() const noexcept -> const xexpression_type&
    {
        return m_e;
    }

    /**
     * Returns the data holder of the underlying container (only if the view is on a realized
     * container). ``xt::eval`` will make sure that the underlying xexpression is
     * on a realized container.
     */
    template <class CT, class... S>
    template <class T>
    inline auto xview<CT, S...>::storage() -> storage_type&
        requires(has_data_interface_concept<T>)
    {
        return m_e.storage();
    }

    template <class CT, class... S>
    template <class T>
    inline auto xview<CT, S...>::storage() const -> const storage_type&
        requires(has_data_interface_concept<T>)
    {
        return m_e.storage();
    }

    template <class CT, class... S>
    template <class T>
    auto xview<CT, S...>::linear_begin() -> linear_iterator
        requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        return m_e.storage().begin() + data_offset();
    }

    template <class CT, class... S>
    template <class T>
    auto xview<CT, S...>::linear_end() -> linear_iterator
        requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        return m_e.storage().begin() + data_offset() + this->size();
    }

    template <class CT, class... S>
    template <class T>
    auto xview<CT, S...>::linear_begin() const -> const_linear_iterator
        requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        return linear_cbegin();
    }

    template <class CT, class... S>
    template <class T>
    auto xview<CT, S...>::linear_end() const -> const_linear_iterator
        requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        return linear_cend();
    }

    template <class CT, class... S>
    template <class T>
    auto xview<CT, S...>::linear_cbegin() const -> const_linear_iterator
        requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        return m_e.storage().cbegin() + data_offset();
    }

    template <class CT, class... S>
    template <class T>
    auto xview<CT, S...>::linear_cend() const -> const_linear_iterator
        requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        return m_e.storage().cbegin() + data_offset() + this->size();
    }

    template <class CT, class... S>
    template <class T>
    auto xview<CT, S...>::linear_rbegin() -> reverse_linear_iterator
        requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        return reverse_linear_iterator(linear_end());
    }

    template <class CT, class... S>
    template <class T>
    auto xview<CT, S...>::linear_rend() -> reverse_linear_iterator
        requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        return reverse_linear_iterator(linear_begin());
    }

    template <class CT, class... S>
    template <class T>
    auto xview<CT, S...>::linear_rbegin() const -> const_reverse_linear_iterator
        requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        return linear_crbegin();
    }

    template <class CT, class... S>
    template <class T>
    auto xview<CT, S...>::linear_rend() const -> const_reverse_linear_iterator
        requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        return linear_crend();
    }

    template <class CT, class... S>
    template <class T>
    auto xview<CT, S...>::linear_crbegin() const -> const_reverse_linear_iterator
        requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        return const_reverse_linear_iterator(linear_end());
    }

    template <class CT, class... S>
    template <class T>
    auto xview<CT, S...>::linear_crend() const -> const_reverse_linear_iterator
        requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        return const_reverse_linear_iterator(linear_begin());
    }

    /**
     * Return the strides for the underlying container of the view.
     */
    template <class CT, class... S>
    template <class T>
    inline auto xview<CT, S...>::strides() const
        -> const inner_strides_type& requires(has_data_interface_concept<T>and strided_view_concept<CT, S...>) {
            if (!m_strides_computed)
            {
                compute_strides(std::integral_constant<bool, has_trivial_strides>{});
                m_strides_computed = true;
            }
            return m_strides;
        }

    template <class CT, class... S>
    template <class T>
    inline auto xview<CT, S...>::backstrides() const
        -> const inner_strides_type& requires(has_data_interface_concept<T>and strided_view_concept<CT, S...>) {
            if (!m_strides_computed)
            {
                compute_strides(std::integral_constant<bool, has_trivial_strides>{});
                m_strides_computed = true;
            }
            return m_backstrides;
        }

    /**
     * Return the pointer to the underlying buffer.
     */
    template <class CT, class... S>
    template <class T>
    inline auto xview<CT, S...>::data() const -> const_pointer
        requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        return m_e.data();
    }

    template <class CT, class... S>
    template <class T>
    inline auto xview<CT, S...>::data() -> pointer
        requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        return m_e.data();
    }

    template <class CT, class... S>
    template <std::size_t... I>
    inline std::size_t xview<CT, S...>::data_offset_impl(std::index_sequence<I...>) const noexcept
    {
        auto temp = std::array<std::ptrdiff_t, sizeof...(S)>(
            {(static_cast<ptrdiff_t>(xt::value(std::get<I>(m_slices), 0)))...}
        );

        std::ptrdiff_t result = 0;
        std::size_t i = 0;
        for (; i < std::min(sizeof...(S), m_e.strides().size()); ++i)
        {
            result += temp[i] * m_e.strides()[i - newaxis_count_before<S...>(i)];
        }
        for (; i < sizeof...(S); ++i)
        {
            result += temp[i];
        }
        return static_cast<std::size_t>(result) + m_e.data_offset();
    }

    /**
     * Return the offset to the first element of the view in the underlying container.
     */
    template <class CT, class... S>
    template <class T>
    inline std::size_t xview<CT, S...>::data_offset() const noexcept
        requires(has_data_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        if (!m_strides_computed)
        {
            compute_strides(std::integral_constant<bool, has_trivial_strides>{});
            m_strides_computed = true;
        }
        return m_data_offset;
    }

    //@}

    template <class CT, class... S>
    inline auto xview<CT, S...>::underlying_size(size_type dim) const -> size_type
    {
        return m_e.shape()[dim];
    }

    template <class CT, class... S>
    inline auto xview<CT, S...>::operator&() & -> xtl::xclosure_pointer<self_type&>
    {
        return xtl::closure_pointer(*this);
    }

    template <class CT, class... S>
    inline auto xview<CT, S...>::operator&() const& -> xtl::xclosure_pointer<const self_type&>
    {
        return xtl::closure_pointer(*this);
    }

    template <class CT, class... S>
    inline auto xview<CT, S...>::operator&() && -> xtl::xclosure_pointer<self_type>
    {
        return xtl::closure_pointer(std::move(*this));
    }

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
    template <class CT, class... S>
    template <class ST>
    inline bool xview<CT, S...>::broadcast_shape(ST& shape, bool) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Checks whether the xview can be linearly assigned to an expression
     * with the specified strides.
     * @return a boolean indicating whether a linear assign is possible
     */
    template <class CT, class... S>
    template <class ST>
    inline bool xview<CT, S...>::has_linear_assign(const ST& str) const
    {
        if constexpr (is_strided_view)
        {
            return str.size() == strides().size() && std::equal(str.cbegin(), str.cend(), strides().begin());
        }
        else
        {
            return false;
        }
    }

    //@}

    template <class CT, class... S>
    template <class It>
    inline It xview<CT, S...>::data_xbegin_impl(It begin) const noexcept
    {
        return begin + data_offset();
    }

    template <class CT, class... S>
    template <class It>
    inline It xview<CT, S...>::data_xend_impl(It begin, layout_type l, size_type offset) const noexcept
    {
        return strided_data_end(*this, begin, l, offset);
    }

    template <class CT, class... S>
    inline auto xview<CT, S...>::data_xbegin() noexcept -> container_iterator
    {
        return data_xbegin_impl(data());
    }

    template <class CT, class... S>
    inline auto xview<CT, S...>::data_xbegin() const noexcept -> const_container_iterator
    {
        return data_xbegin_impl(data());
    }

    template <class CT, class... S>
    inline auto xview<CT, S...>::data_xend(layout_type l, size_type offset) noexcept -> container_iterator
    {
        return data_xend_impl(data() + data_offset(), l, offset);
    }

    template <class CT, class... S>
    inline auto xview<CT, S...>::data_xend(layout_type l, size_type offset) const noexcept
        -> const_container_iterator
    {
        return data_xend_impl(data() + data_offset(), l, offset);
    }

    // Assign to operator enabled for contigous views
    template <class CT, class... S>
    template <class E, class T>
    void xview<CT, S...>::assign_to(xexpression<E>& e, bool force_resize) const
        requires(has_data_interface_concept<T> and contiguous_view_concept<E, S...>)
    {
        auto& de = e.derived_cast();
        de.resize(shape(), force_resize);
        std::copy(data() + data_offset(), data() + data_offset() + de.size(), de.template begin<static_layout>());
    }

    template <class CT, class... S>
    template <class E, std::size_t... I>
    inline auto xview<CT, S...>::build_view_impl(E&& e, std::index_sequence<I...>) const -> rebind_t<E>
    {
        return rebind_t<E>(std::forward<E>(e), std::get<I>(m_slices)...);
    }

    template <class CT, class... S>
    template <class E>
    inline auto xview<CT, S...>::build_view(E&& e) const -> rebind_t<E>
    {
        return build_view_impl(std::forward<E>(e), std::make_index_sequence<sizeof...(S)>());
    }

    template <class CT, class... S>
    template <class align, class simd, class T>
    inline auto xview<CT, S...>::store_simd(size_type i, const simd& e) -> void
        requires(has_simd_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        return m_e.template store_simd<xt_simd::unaligned_mode>(data_offset() + i, e);
    }

    template <class CT, class... S>
    template <class align, class requested_type, std::size_t N, class T>
    inline auto xview<CT, S...>::load_simd(size_type i) const -> simd_return_type<requested_type>
        requires(has_simd_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        return m_e.template load_simd<xt_simd::unaligned_mode, requested_type>(data_offset() + i);
    }

    template <class CT, class... S>
    template <class T>
    inline auto xview<CT, S...>::data_element(size_type i) -> reference
        requires(has_simd_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        return m_e.data_element(data_offset() + i);
    }

    template <class CT, class... S>
    template <class T>
    inline auto xview<CT, S...>::data_element(size_type i) const -> const_reference
        requires(has_simd_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        return m_e.data_element(data_offset() + i);
    }

    template <class CT, class... S>
    template <class T>
    inline auto xview<CT, S...>::flat(size_type i) -> reference
        requires(has_simd_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        XTENSOR_ASSERT(is_contiguous());
        return m_e.flat(data_offset() + i);
    }

    template <class CT, class... S>
    template <class T>
    inline auto xview<CT, S...>::flat(size_type i) const -> const_reference
        requires(has_simd_interface_concept<T> and strided_view_concept<CT, S...>)
    {
        XTENSOR_ASSERT(is_contiguous());
        return m_e.flat(data_offset() + i);
    }

    template <class CT, class... S>
    template <class... Args>
    inline auto xview<CT, S...>::make_index_sequence(Args...) const noexcept
    {
        return std::make_index_sequence<
            (sizeof...(Args) + integral_count<S...>() > newaxis_count<S...>()
                 ? sizeof...(Args) + integral_count<S...>() - newaxis_count<S...>()
                 : 0)>();
    }

    template <class CT, class... S>
    template <std::size_t... I>
    inline auto xview<CT, S...>::compute_strides_impl(std::index_sequence<I...>) const noexcept
    {
        std::size_t original_dim = m_e.dimension();
        return std::array<std::ptrdiff_t, sizeof...(I)>(
            {(static_cast<std::ptrdiff_t>(xt::step_size(std::get<integral_skip<S...>(I)>(m_slices), 1))
              * ((integral_skip<S...>(I) - newaxis_count_before<S...>(integral_skip<S...>(I))) < original_dim
                     ? m_e.strides()[integral_skip<S...>(I) - newaxis_count_before<S...>(integral_skip<S...>(I))]
                     : 1))...}
        );
    }

    template <class CT, class... S>
    inline void xview<CT, S...>::compute_strides(std::false_type) const
    {
        m_strides = xtl::make_sequence<inner_strides_type>(this->dimension(), 0);
        m_backstrides = xtl::make_sequence<inner_strides_type>(this->dimension(), 0);

        constexpr std::size_t n_strides = sizeof...(S) - integral_count<S...>();

        auto slice_strides = compute_strides_impl(std::make_index_sequence<n_strides>());

        for (std::size_t i = 0; i < n_strides; ++i)
        {
            m_strides[i] = slice_strides[i];
            // adapt strides for shape[i] == 1 to make consistent with rest of xtensor
            detail::adapt_strides(shape(), m_strides, &m_backstrides, i);
        }
        for (std::size_t i = n_strides; i < this->dimension(); ++i)
        {
            m_strides[i] = m_e.strides()[i + integral_count<S...>() - newaxis_count<S...>()];
            detail::adapt_strides(shape(), m_strides, &m_backstrides, i);
        }

        m_data_offset = data_offset_impl(std::make_index_sequence<sizeof...(S)>());
    }

    template <class CT, class... S>
    inline void xview<CT, S...>::compute_strides(std::true_type) const
    {
    }

    template <class CT, class... S>
    inline auto xview<CT, S...>::access() -> reference
    {
        return access_impl(make_index_sequence());
    }

    template <class CT, class... S>
    template <class Arg, class... Args>
    inline auto xview<CT, S...>::access(Arg arg, Args... args) -> reference
    {
        if (sizeof...(Args) >= this->dimension())
        {
            return access(args...);
        }
        return access_impl(make_index_sequence(arg, args...), arg, args...);
    }

    template <class CT, class... S>
    inline auto xview<CT, S...>::access() const -> const_reference
    {
        return access_impl(make_index_sequence());
    }

    template <class CT, class... S>
    template <class Arg, class... Args>
    inline auto xview<CT, S...>::access(Arg arg, Args... args) const -> const_reference
    {
        if (sizeof...(Args) >= this->dimension())
        {
            return access(args...);
        }
        return access_impl(make_index_sequence(arg, args...), arg, args...);
    }

    template <class CT, class... S>
    template <typename std::decay_t<CT>::size_type... I, class... Args>
    inline auto xview<CT, S...>::unchecked_impl(std::index_sequence<I...>, Args... args) -> reference
    {
        return m_e.unchecked(index<I>(args...)...);
    }

    template <class CT, class... S>
    template <typename std::decay_t<CT>::size_type... I, class... Args>
    inline auto xview<CT, S...>::unchecked_impl(std::index_sequence<I...>, Args... args) const
        -> const_reference
    {
        return m_e.unchecked(index<I>(args...)...);
    }

    template <class CT, class... S>
    template <typename std::decay_t<CT>::size_type... I, class... Args>
    inline auto xview<CT, S...>::access_impl(std::index_sequence<I...>, Args... args) -> reference
    {
        return m_e(index<I>(args...)...);
    }

    template <class CT, class... S>
    template <typename std::decay_t<CT>::size_type... I, class... Args>
    inline auto xview<CT, S...>::access_impl(std::index_sequence<I...>, Args... args) const -> const_reference
    {
        return m_e(index<I>(args...)...);
    }

    template <class CT, class... S>
    template <typename std::decay_t<CT>::size_type I, class... Args>
    inline auto xview<CT, S...>::index(Args... args) const -> size_type
    {
        if constexpr (lesser_condition<I>::value)
        {
            return sliced_access<I - integral_count_before<S...>(I) + newaxis_count_before<S...>(I + 1)>(
                std::get<I + newaxis_count_before<S...>(I + 1)>(m_slices),
                args...
            );
        }
        else
        {
            return argument<I - integral_count<S...>() + newaxis_count<S...>()>(args...);
        }
    }

    template <class CT, class... S>
    template <typename std::decay_t<CT>::size_type I, class T>
    inline auto xview<CT, S...>::sliced_access(const xslice<T>& slice) const -> size_type
    {
        return static_cast<size_type>(slice.derived_cast()(0));
    }

    template <class CT, class... S>
    template <typename std::decay_t<CT>::size_type I, class T, class Arg, class... Args>
    inline auto xview<CT, S...>::sliced_access(const xslice<T>& slice, Arg arg, Args... args) const -> size_type
    {
        using ST = typename T::size_type;
        return static_cast<size_type>(
            slice.derived_cast()(argument<I>(static_cast<ST>(arg), static_cast<ST>(args)...))
        );
    }

    template <class CT, class... S>
    template <typename std::decay_t<CT>::size_type I, class T, class... Args>
    inline auto xview<CT, S...>::sliced_access(const T& squeeze, Args...) const -> disable_xslice<T, size_type>
    {
        return static_cast<size_type>(squeeze);
    }

    template <class CT, class... S>
    template <class It>
    inline auto xview<CT, S...>::make_index(It first, It last) const -> base_index_type
    {
        auto index = xtl::make_sequence<base_index_type>(m_e.dimension(), 0);
        using diff_type = typename std::iterator_traits<It>::difference_type;
        using ivalue_type = typename base_index_type::value_type;
        auto func1 = [&first](const auto& s) noexcept
        {
            return get_slice_value(s, first);
        };
        auto func2 = [](const auto& s) noexcept
        {
            return xt::value(s, 0);
        };

        auto s = static_cast<diff_type>(
            (std::min)(static_cast<size_type>(std::distance(first, last)), this->dimension())
        );
        auto first_copy = last - s;
        for (size_type i = 0; i != m_e.dimension(); ++i)
        {
            size_type k = newaxis_skip<S...>(i);

            // need to advance captured `first`
            first = first_copy;
            std::advance(first, static_cast<diff_type>(k - xt::integral_count_before<S...>(i)));

            if (first < last)
            {
                index[i] = k < sizeof...(S) ? apply<size_type>(k, func1, m_slices)
                                            : static_cast<ivalue_type>(*first);
            }
            else
            {
                index[i] = k < sizeof...(S) ? apply<size_type>(k, func2, m_slices) : ivalue_type(0);
            }
        }
        return index;
    }

    template <class CT, class... S>
    inline auto xview<CT, S...>::compute_shape(std::true_type) const -> inner_shape_type
    {
        return inner_shape_type(m_e.shape());
    }

    template <class CT, class... S>
    inline auto xview<CT, S...>::compute_shape(std::false_type) const -> inner_shape_type
    {
        std::size_t dim = m_e.dimension() - integral_count<S...>() + newaxis_count<S...>();
        auto shape = xtl::make_sequence<inner_shape_type>(dim, 0);
        auto func = [](const auto& s) noexcept
        {
            return get_size(s);
        };
        for (size_type i = 0; i != dim; ++i)
        {
            size_type index = integral_skip<S...>(i);
            shape[i] = index < sizeof...(S) ? apply<size_type>(index, func, m_slices)
                                            : m_e.shape()[index - newaxis_count_before<S...>(index)];
        }
        return shape;
    }

    namespace xview_detail
    {
        template <class V, class T>
        inline void run_assign_temporary_impl(V& v, const T& t, std::true_type /* enable strided assign */)
        {
            strided_loop_assigner<true>::run(v, t);
        }

        template <class V, class T>
        inline void
        run_assign_temporary_impl(V& v, const T& t, std::false_type /* fallback to iterator assign */)
        {
            std::copy(t.cbegin(), t.cend(), v.begin());
        }
    }

    template <class CT, class... S>
    inline void xview<CT, S...>::assign_temporary_impl(temporary_type&& tmp)
    {
        constexpr bool fast_assign = detail::is_strided_view<xexpression_type, S...>::value
                                     && xassign_traits<xview<CT, S...>, temporary_type>::simd_strided_assign();
        xview_detail::run_assign_temporary_impl(*this, tmp, std::integral_constant<bool, fast_assign>{});
    }

    namespace detail
    {
        template <class E, class... S>
        inline std::size_t get_underlying_shape_index(std::size_t I)
        {
            return I - newaxis_count_before<get_slice_type<E, S>...>(I);
        }

        template <class... S>
        struct check_slice;

        template <>
        struct check_slice<>
        {
            using type = void_t<>;
        };

        template <class S, class... SL>
        struct check_slice<S, SL...>
        {
            static_assert(!std::is_same<S, xellipsis_tag>::value, "ellipsis not supported vith xview");
            using type = typename check_slice<SL...>::type;
        };

        template <class E, std::size_t... I, class... S>
        inline auto make_view_impl(E&& e, std::index_sequence<I...>, S&&... slices)
        {
            // Checks that no ellipsis slice is used
            using view_type = xview<xtl::closure_type_t<E>, get_slice_type<std::decay_t<E>, S>...>;
            return view_type(
                std::forward<E>(e),
                get_slice_implementation(
                    e,
                    std::forward<S>(slices),
                    get_underlying_shape_index<std::decay_t<E>, S...>(I)
                )...
            );
        }
    }

    /**
     * Constructs and returns a view on the specified xexpression. Users
     * should not directly construct the slices but call helper functions
     * instead.
     * @param e the xexpression to adapt
     * @param slices the slices list describing the view. \c view accepts negative
     * indices, in that case indexing is done in reverse order.
     * @sa range, all, newaxis
     */
    template <class E, class... S>
    inline auto view(E&& e, S&&... slices)
    {
        return detail::make_view_impl(
            std::forward<E>(e),
            std::make_index_sequence<sizeof...(S)>(),
            std::forward<S>(slices)...
        );
    }

    namespace detail
    {
        class row_impl
        {
        public:

            template <class E>
            inline static auto make(E&& e, const std::ptrdiff_t index)
            {
                const auto shape = e.shape();
                check_dimension(shape);
                return view(e, index, xt::all());
            }

        private:

            template <class S>
            inline static void check_dimension(const S& shape)
            {
                if (shape.size() != 2)
                {
                    XTENSOR_THROW(
                        std::invalid_argument,
                        "A row can only be accessed on an expression with exact two dimensions"
                    );
                }
            }

            template <class T, std::size_t N>
            inline static void check_dimension(const std::array<T, N>&)
            {
                static_assert(N == 2, "A row can only be accessed on an expression with exact two dimensions");
            }
        };

        class column_impl
        {
        public:

            template <class E>
            inline static auto make(E&& e, const std::ptrdiff_t index)
            {
                const auto shape = e.shape();
                check_dimension(shape);
                return view(e, xt::all(), index);
            }

        private:

            template <class S>
            inline static void check_dimension(const S& shape)
            {
                if (shape.size() != 2)
                {
                    XTENSOR_THROW(
                        std::invalid_argument,
                        "A column can only be accessed on an expression with exact two dimensions"
                    );
                }
            }

            template <class T, std::size_t N>
            inline static void check_dimension(const std::array<T, N>&)
            {
                static_assert(N == 2, "A column can only be accessed on an expression with exact two dimensions");
            }
        };
    }

    /**
     * Constructs and returns a row (sliced view) on the specified expression.
     * Users should not directly construct the slices but call helper functions
     * instead. This function is only allowed on expressions with two dimensions.
     * @param e the xexpression to adapt
     * @param index 0-based index of the row, negative indices will return the
     * last rows in reverse order.
     * @throws std::invalid_argument if the expression has more than 2 dimensions.
     */
    template <class E>
    inline auto row(E&& e, std::ptrdiff_t index)
    {
        return detail::row_impl::make(e, index);
    }

    /**
     * Constructs and returns a column (sliced view) on the specified expression.
     * Users should not directly construct the slices but call helper functions
     * instead. This function is only allowed on expressions with two dimensions.
     * @param e the xexpression to adapt
     * @param index 0-based index of the column, negative indices will return the
     * last columns in reverse order.
     * @throws std::invalid_argument if the expression has more than 2 dimensions.
     */
    template <class E>
    inline auto col(E&& e, std::ptrdiff_t index)
    {
        return detail::column_impl::make(e, index);
    }

    /***************
     * stepper api *
     ***************/

    template <class CT, class... S>
    template <class ST>
    inline auto xview<CT, S...>::stepper_begin(const ST& shape) -> stepper
    {
        const size_type offset = shape.size() - this->dimension();
        if constexpr (is_strided_view)
        {
            return stepper(this, data_xbegin(), offset);
        }
        else
        {
            return stepper(this, m_e.stepper_begin(m_e.shape()), offset);
        }
    }

    template <class CT, class... S>
    template <class ST>
    inline auto xview<CT, S...>::stepper_end(const ST& shape, layout_type l) -> stepper
    {
        const size_type offset = shape.size() - this->dimension();
        if constexpr (is_strided_view)
        {
            return stepper(this, data_xend(l, offset), offset);
        }
        else
        {
            return stepper(this, m_e.stepper_end(m_e.shape(), l), offset, true, l);
        }
    }

    template <class CT, class... S>
    template <class ST>
    inline auto xview<CT, S...>::stepper_begin(const ST& shape) const -> const_stepper
    {
        const size_type offset = shape.size() - this->dimension();
        if constexpr (is_strided_view)
        {
            return const_stepper(this, data_xbegin(), offset);
        }
        else
        {
            const xexpression_type& e = m_e;
            return const_stepper(this, e.stepper_begin(m_e.shape()), offset);
        }
    }

    template <class CT, class... S>
    template <class ST>
    inline auto xview<CT, S...>::stepper_end(const ST& shape, layout_type l) const -> const_stepper
    {
        const size_type offset = shape.size() - this->dimension();
        if constexpr (is_strided_view)
        {
            return const_stepper(this, data_xend(l, offset), offset);
        }
        else
        {
            const xexpression_type& e = m_e;
            return const_stepper(this, e.stepper_end(m_e.shape(), l), offset, true, l);
        }
    }

    /********************************
     * xview_stepper implementation *
     ********************************/

    template <bool is_const, class CT, class... S>
    inline xview_stepper<is_const, CT, S...>::xview_stepper(
        view_type* view,
        substepper_type it,
        size_type offset,
        bool end,
        layout_type l
    )
        : p_view(view)
        , m_it(it)
        , m_offset(offset)
    {
        if (!end)
        {
            std::fill(m_index_keeper.begin(), m_index_keeper.end(), 0);
            auto func = [](const auto& s) noexcept
            {
                return xt::value(s, 0);
            };
            for (size_type i = 0; i < sizeof...(S); ++i)
            {
                if (!is_newaxis_slice(i))
                {
                    size_type s = apply<size_type>(i, func, p_view->slices());
                    size_type index = i - newaxis_count_before<S...>(i);
                    m_it.step(index, s);
                }
            }
        }
        else
        {
            to_end_impl(l);
        }
    }

    template <bool is_const, class CT, class... S>
    inline auto xview_stepper<is_const, CT, S...>::operator*() const -> reference
    {
        return *m_it;
    }

    template <bool is_const, class CT, class... S>
    inline void xview_stepper<is_const, CT, S...>::step(size_type dim)
    {
        auto func = [this](size_type index, size_type offset)
        {
            m_it.step(index, offset);
        };
        common_step_forward(dim, func);
    }

    template <bool is_const, class CT, class... S>
    inline void xview_stepper<is_const, CT, S...>::step_back(size_type dim)
    {
        auto func = [this](size_type index, size_type offset)
        {
            m_it.step_back(index, offset);
        };
        common_step_backward(dim, func);
    }

    template <bool is_const, class CT, class... S>
    inline void xview_stepper<is_const, CT, S...>::step(size_type dim, size_type n)
    {
        auto func = [this](size_type index, size_type offset)
        {
            m_it.step(index, offset);
        };
        common_step_forward(dim, n, func);
    }

    template <bool is_const, class CT, class... S>
    inline void xview_stepper<is_const, CT, S...>::step_back(size_type dim, size_type n)
    {
        auto func = [this](size_type index, size_type offset)
        {
            m_it.step_back(index, offset);
        };
        common_step_backward(dim, n, func);
    }

    template <bool is_const, class CT, class... S>
    inline void xview_stepper<is_const, CT, S...>::reset(size_type dim)
    {
        auto func = [this](size_type index, size_type offset)
        {
            m_it.step_back(index, offset);
        };
        common_reset(dim, func, false);
    }

    template <bool is_const, class CT, class... S>
    inline void xview_stepper<is_const, CT, S...>::reset_back(size_type dim)
    {
        auto func = [this](size_type index, size_type offset)
        {
            m_it.step(index, offset);
        };
        common_reset(dim, func, true);
    }

    template <bool is_const, class CT, class... S>
    inline void xview_stepper<is_const, CT, S...>::to_begin()
    {
        std::fill(m_index_keeper.begin(), m_index_keeper.end(), 0);
        m_it.to_begin();
    }

    template <bool is_const, class CT, class... S>
    inline void xview_stepper<is_const, CT, S...>::to_end(layout_type l)
    {
        m_it.to_end(l);
        to_end_impl(l);
    }

    template <bool is_const, class CT, class... S>
    inline bool xview_stepper<is_const, CT, S...>::is_newaxis_slice(size_type index) const noexcept
    {
        // A bit tricky but avoids a lot of template instantiations
        return newaxis_count_before<S...>(index + 1) != newaxis_count_before<S...>(index);
    }

    template <bool is_const, class CT, class... S>
    inline void xview_stepper<is_const, CT, S...>::to_end_impl(layout_type l)
    {
        auto func = [](const auto& s) noexcept
        {
            return xt::value(s, get_size(s) - 1);
        };
        auto size_func = [](const auto& s) noexcept
        {
            return get_size(s);
        };

        for (size_type i = 0; i < sizeof...(S); ++i)
        {
            if (!is_newaxis_slice(i))
            {
                size_type s = apply<size_type>(i, func, p_view->slices());
                size_type ix = apply<size_type>(i, size_func, p_view->slices());
                m_index_keeper[i] = ix - size_type(1);
                size_type index = i - newaxis_count_before<S...>(i);
                s = p_view->underlying_size(index) - 1 - s;
                m_it.step_back(index, s);
            }
        }
        if (l == layout_type::row_major)
        {
            for (size_type i = sizeof...(S); i > 0; --i)
            {
                if (!is_newaxis_slice(i - 1))
                {
                    m_index_keeper[i - 1]++;
                    break;
                }
            }
        }
        else if (l == layout_type::column_major)
        {
            for (size_type i = 0; i < sizeof...(S); ++i)
            {
                if (!is_newaxis_slice(i))
                {
                    m_index_keeper[i]++;
                    break;
                }
            }
        }
        else
        {
            XTENSOR_THROW(std::runtime_error, "Iteration only allowed in row or column major.");
        }
    }

    template <bool is_const, class CT, class... S>
    template <class F>
    void xview_stepper<is_const, CT, S...>::common_step_forward(size_type dim, F f)
    {
        if (dim >= m_offset)
        {
            auto func = [&dim, this](const auto& s) noexcept
            {
                return step_size(s, this->m_index_keeper[dim]++, 1);
            };
            size_type index = integral_skip<S...>(dim);
            if (!is_newaxis_slice(index))
            {
                size_type step_size = index < sizeof...(S) ? apply<size_type>(index, func, p_view->slices())
                                                           : 1;
                index -= newaxis_count_before<S...>(index);
                f(index, step_size);
            }
        }
    }

    template <bool is_const, class CT, class... S>
    template <class F>
    void xview_stepper<is_const, CT, S...>::common_step_forward(size_type dim, size_type n, F f)
    {
        if (dim >= m_offset)
        {
            auto func = [&dim, &n, this](const auto& s) noexcept
            {
                auto st_size = step_size(s, this->m_index_keeper[dim], n);
                this->m_index_keeper[dim] += n;
                return size_type(st_size);
            };

            size_type index = integral_skip<S...>(dim);
            if (!is_newaxis_slice(index))
            {
                size_type step_size = index < sizeof...(S) ? apply<size_type>(index, func, p_view->slices())
                                                           : n;
                index -= newaxis_count_before<S...>(index);
                f(index, step_size);
            }
        }
    }

    template <bool is_const, class CT, class... S>
    template <class F>
    void xview_stepper<is_const, CT, S...>::common_step_backward(size_type dim, F f)
    {
        if (dim >= m_offset)
        {
            auto func = [&dim, this](const auto& s) noexcept
            {
                this->m_index_keeper[dim]--;
                return step_size(s, this->m_index_keeper[dim], 1);
            };
            size_type index = integral_skip<S...>(dim);
            if (!is_newaxis_slice(index))
            {
                size_type step_size = index < sizeof...(S) ? apply<size_type>(index, func, p_view->slices())
                                                           : 1;
                index -= newaxis_count_before<S...>(index);
                f(index, step_size);
            }
        }
    }

    template <bool is_const, class CT, class... S>
    template <class F>
    void xview_stepper<is_const, CT, S...>::common_step_backward(size_type dim, size_type n, F f)
    {
        if (dim >= m_offset)
        {
            auto func = [&dim, &n, this](const auto& s) noexcept
            {
                this->m_index_keeper[dim] -= n;
                return step_size(s, this->m_index_keeper[dim], n);
            };

            size_type index = integral_skip<S...>(dim);
            if (!is_newaxis_slice(index))
            {
                size_type step_size = index < sizeof...(S) ? apply<size_type>(index, func, p_view->slices())
                                                           : n;
                index -= newaxis_count_before<S...>(index);
                f(index, step_size);
            }
        }
    }

    template <bool is_const, class CT, class... S>
    template <class F>
    void xview_stepper<is_const, CT, S...>::common_reset(size_type dim, F f, bool backwards)
    {
        auto size_func = [](const auto& s) noexcept
        {
            return get_size(s);
        };
        auto end_func = [](const auto& s) noexcept
        {
            return xt::value(s, get_size(s) - 1) - xt::value(s, 0);
        };

        size_type index = integral_skip<S...>(dim);
        if (!is_newaxis_slice(index))
        {
            if (dim < m_index_keeper.size())
            {
                size_type size = index < sizeof...(S) ? apply<size_type>(index, size_func, p_view->slices())
                                                      : p_view->shape()[dim];
                m_index_keeper[dim] = backwards ? size - 1 : 0;
            }

            size_type reset_n = index < sizeof...(S) ? apply<size_type>(index, end_func, p_view->slices())
                                                     : p_view->shape()[dim] - 1;
            index -= newaxis_count_before<S...>(index);
            f(index, reset_n);
        }
    }
}

#endif
