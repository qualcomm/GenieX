#ifndef XTENSOR_XBLOCKWISE_REDUCER_HPP
#define XTENSOR_XBLOCKWISE_REDUCER_HPP

#include "../core/xmultiindex_iterator.hpp"
#include "../core/xshape.hpp"
#include "../reducers/xblockwise_reducer_functors.hpp"
#include "../reducers/xreducer.hpp"
#include "xtl/xclosure.hpp"
#include "xtl/xsequence.hpp"

namespace xt
{

    template <class CT, class F, class X, class O>
    class xblockwise_reducer
    {
    public:

        using self_type = xblockwise_reducer<CT, F, X, O>;
        using raw_options_type = std::decay_t<O>;
        using keep_dims = xtl::mpl::contains<raw_options_type, xt::keep_dims_type>;
        using xexpression_type = std::decay_t<CT>;
        using shape_type = typename xreducer_shape_type<typename xexpression_type::shape_type, std::decay_t<X>, keep_dims>::type;
        using functor_type = F;
        using value_type = typename functor_type::value_type;
        using input_shape_type = typename xexpression_type::shape_type;
        using input_chunk_index_type = filter_fixed_shape_t<input_shape_type>;
        using input_grid_strides = filter_fixed_shape_t<input_shape_type>;
        using axes_type = X;
        using chunk_shape_type = filter_fixed_shape_t<shape_type>;


        template <class E, class BS, class XX, class OO, class FF>
        xblockwise_reducer(E&& e, BS&& block_shape, XX&& axes, OO&& options, FF&& functor);

        const input_shape_type& input_shape() const;
        const axes_type& axes() const;

        std::size_t dimension() const;

        const shape_type& shape() const;

        const chunk_shape_type& chunk_shape() const;

        template <class R>
        void assign_to(R& result) const;

    private:

        using mapping_type = filter_fixed_shape_t<shape_type>;
        using input_chunked_view_type = xchunked_view<const std::decay_t<CT>&>;
        using input_const_chunked_iterator_type = typename input_chunked_view_type::const_chunk_iterator;
        using input_chunk_range_type = std::array<xmultiindex_iterator<input_chunk_index_type>, 2>;

        template <class CI>
        void assign_to_chunk(CI& result_chunk_iter) const;

        template <class CI>
        input_chunk_range_type compute_input_chunk_range(CI& result_chunk_iter) const;

        input_const_chunked_iterator_type get_input_chunk_iter(input_chunk_index_type input_chunk_index) const;
        void init_shapes();

        CT m_e;
        xchunked_view<const std::decay_t<CT>&> m_e_chunked_view;
        axes_type m_axes;
        raw_options_type m_options;
        functor_type m_functor;
        shape_type m_result_shape;
        chunk_shape_type m_result_chunk_shape;
        mapping_type m_mapping;
        input_grid_strides m_input_grid_strides;
    };

    template <class CT, class F, class X, class O>
    template <class E, class BS, class XX, class OO, class FF>
    xblockwise_reducer<CT, F, X, O>::xblockwise_reducer(E&& e, BS&& block_shape, XX&& axes, OO&& options, FF&& functor)
        : m_e(std::forward<E>(e))
        , m_e_chunked_view(m_e, std::forward<BS>(block_shape))
        , m_axes(std::forward<XX>(axes))
        , m_options(std::forward<OO>(options))
        , m_functor(std::forward<FF>(functor))
        , m_result_shape()
        , m_result_chunk_shape()
        , m_mapping()
        , m_input_grid_strides()
    {
        init_shapes();
        resize_container(m_input_grid_strides, m_e.dimension());
        std::size_t stride = 1;

        for (std::size_t i = m_input_grid_strides.size(); i != 0; --i)
        {
            m_input_grid_strides[i - 1] = stride;
            stride *= m_e_chunked_view.grid_shape()[i - 1];
        }
    }

    template <class CT, class F, class X, class O>
    inline auto xblockwise_reducer<CT, F, X, O>::input_shape() const -> const input_shape_type&
    {
        return m_e.shape();
    }

    template <class CT, class F, class X, class O>
    inline auto xblockwise_reducer<CT, F, X, O>::axes() const -> const axes_type&
    {
        return m_axes;
    }

    template <class CT, class F, class X, class O>
    inline std::size_t xblockwise_reducer<CT, F, X, O>::dimension() const
    {
        return m_result_shape.size();
    }

    template <class CT, class F, class X, class O>
    inline auto xblockwise_reducer<CT, F, X, O>::shape() const -> const shape_type&
    {
        return m_result_shape;
    }

    template <class CT, class F, class X, class O>
    inline auto xblockwise_reducer<CT, F, X, O>::chunk_shape() const -> const chunk_shape_type&
    {
        return m_result_chunk_shape;
    }

    template <class CT, class F, class X, class O>
    template <class R>
    inline void xblockwise_reducer<CT, F, X, O>::assign_to(R& result) const
    {
        auto result_chunked_view = as_chunked(result, m_result_chunk_shape);
        for (auto chunk_iter = result_chunked_view.chunk_begin(); chunk_iter != result_chunked_view.chunk_end();
             ++chunk_iter)
        {
            assign_to_chunk(chunk_iter);
        }
    }

    template <class CT, class F, class X, class O>
    auto xblockwise_reducer<CT, F, X, O>::get_input_chunk_iter(input_chunk_index_type input_chunk_index) const
        -> input_const_chunked_iterator_type
    {
        std::size_t chunk_linear_index = 0;
        for (std::size_t i = 0; i < m_e_chunked_view.dimension(); ++i)
        {
            chunk_linear_index += input_chunk_index[i] * m_input_grid_strides[i];
        }
        return input_const_chunked_iterator_type(m_e_chunked_view, std::move(input_chunk_index), chunk_linear_index);
    }

    template <class CT, class F, class X, class O>
    template <class CI>
    void xblockwise_reducer<CT, F, X, O>::assign_to_chunk(CI& result_chunk_iter) const
    {
        auto result_chunk_view = *result_chunk_iter;
        auto reduction_variable = m_functor.reduction_variable(result_chunk_view);

        // get the range of input chunks we need to compute the desired ouput chunk
        auto range = compute_input_chunk_range(result_chunk_iter);

        // iterate over input chunk (indics)
        auto first = true;
        // std::for_each(std::get<0>(range), std::get<1>(range), [&](auto && input_chunk_index)
        auto iter = std::get<0>(range);
        while (iter != std::get<1>(range))
        {
            const auto& input_chunk_index = *iter;
            // get input chunk iterator from chunk index
            auto chunked_input_iter = this->get_input_chunk_iter(input_chunk_index);
            auto input_chunk_view = *chunked_input_iter;

            // compute the per block result
            auto block_res = m_functor.compute(input_chunk_view, m_axes, m_options);

            // merge
            m_functor.merge(block_res, first, result_chunk_view, reduction_variable);
            first = false;
            ++iter;
        }

        // finalize (ie smth like normalization)
        m_functor.finalize(reduction_variable, result_chunk_view, *this);
    }

    template <class CT, class F, class X, class O>
    template <class CI>
    auto xblockwise_reducer<CT, F, X, O>::compute_input_chunk_range(CI& result_chunk_iter) const
        -> input_chunk_range_type
    {
        auto input_chunks_begin = xtl::make_sequence<input_chunk_index_type>(m_e_chunked_view.dimension(), 0);
        auto input_chunks_end = xtl::make_sequence<input_chunk_index_type>(m_e_chunked_view.dimension());

        XTENSOR_ASSERT(input_chunks_begin.size() == m_e_chunked_view.dimension());
        XTENSOR_ASSERT(input_chunks_end.size() == m_e_chunked_view.dimension());

        std::copy(
            m_e_chunked_view.grid_shape().begin(),
            m_e_chunked_view.grid_shape().end(),
            input_chunks_end.begin()
        );

        const auto& chunk_index = result_chunk_iter.chunk_index();
        for (std::size_t result_ax_index = 0; result_ax_index < m_result_shape.size(); ++result_ax_index)
        {
            if (m_result_shape[result_ax_index] != 1)
            {
                const auto input_ax_index = m_mapping[result_ax_index];
                input_chunks_begin[input_ax_index] = chunk_index[result_ax_index];
                input_chunks_end[input_ax_index] = chunk_index[result_ax_index] + 1;
            }
        }
        return input_chunk_range_type{
            multiindex_iterator_begin<input_chunk_index_type>(input_chunks_begin, input_chunks_end),
            multiindex_iterator_end<input_chunk_index_type>(input_chunks_begin, input_chunks_end)
        };
    }

    template <class CT, class F, class X, class O>
    void xblockwise_reducer<CT, F, X, O>::init_shapes()
    {
        const auto& shape = m_e.shape();
        const auto dimension = m_e.dimension();
        const auto& block_shape = m_e_chunked_view.chunk_shape();
        if (xtl::mpl::contains<raw_options_type, xt::keep_dims_type>::value)
        {
            resize_container(m_result_shape, dimension);
            resize_container(m_result_chunk_shape, dimension);
            resize_container(m_mapping, dimension);
            for (std::size_t i = 0; i < dimension; ++i)
            {
                m_mapping[i] = i;
                if (std::find(m_axes.begin(), m_axes.end(), i) == m_axes.end())
                {
                    // i not in m_axes!
                    m_result_shape[i] = shape[i];
                    m_result_chunk_shape[i] = block_shape[i];
                }
                else
                {
                    m_result_shape[i] = 1;
                    m_result_chunk_shape[i] = 1;
                }
            }
        }
        else
        {
            const auto result_dim = dimension - m_axes.size();
            resize_container(m_result_shape, result_dim);
            resize_container(m_result_chunk_shape, result_dim);
            resize_container(m_mapping, result_dim);

            for (std::size_t i = 0, idx = 0; i < dimension; ++i)
            {
                if (std::find(m_axes.begin(), m_axes.end(), i) == m_axes.end())
                {
                    // i not in axes!
                    m_result_shape[idx] = shape[i];
                    m_result_chunk_shape[idx] = block_shape[i];
                    m_mapping[idx] = i;
                    ++idx;
                }
            }
        }
    }

    template <class E, class CS, class A, class O, class FF>
    inline auto blockwise_reducer(E&& e, CS&& chunk_shape, A&& axes, O&& raw_options, FF&& functor)
    {
        using functor_type = std::decay_t<FF>;
        using closure_type = xtl::const_closure_type_t<E>;
        using axes_type = std::decay_t<A>;

        return xblockwise_reducer<closure_type, functor_type, axes_type, O>(
            std::forward<E>(e),
            std::forward<CS>(chunk_shape),
            std::forward<A>(axes),
            std::forward<O>(raw_options),
            std::forward<FF>(functor)
        );
    }

    namespace blockwise
    {

#define XTENSOR_BLOCKWISE_REDUCER_FUNC(FNAME, FUNCTOR)                                                        \
    template <                                                                                                \
        class T = void,                                                                                       \
        class E,                                                                                              \
        class BS,                                                                                             \
        class X,                                                                                              \
        class O = DEFAULT_STRATEGY_REDUCERS,                                                                  \
        XTL_REQUIRES(std::negation<is_reducer_options<X>>, std::negation<xtl::is_integral<std::decay_t<X>>>)> \
    auto FNAME(E&& e, BS&& block_shape, X&& axes, O options = O())                                            \
    {                                                                                                         \
        using input_expression_type = std::decay_t<E>;                                                        \
        using functor_type = FUNCTOR<typename input_expression_type::value_type, T>;                          \
        return blockwise_reducer(                                                                             \
            std::forward<E>(e),                                                                               \
            std::forward<BS>(block_shape),                                                                    \
            std::forward<X>(axes),                                                                            \
            std::forward<O>(options),                                                                         \
            functor_type()                                                                                    \
        );                                                                                                    \
    }                                                                                                         \
    template <                                                                                                \
        class T = void,                                                                                       \
        class E,                                                                                              \
        class BS,                                                                                             \
        class X,                                                                                              \
        class O = DEFAULT_STRATEGY_REDUCERS,                                                                  \
        XTL_REQUIRES(xtl::is_integral<std::decay_t<X>>)>                                                      \
    auto FNAME(E&& e, BS&& block_shape, X axis, O options = O())                                              \
    {                                                                                                         \
        std::array<X, 1> axes{axis};                                                                          \
        using input_expression_type = std::decay_t<E>;                                                        \
        using functor_type = FUNCTOR<typename input_expression_type::value_type, T>;                          \
        return blockwise_reducer(                                                                             \
            std::forward<E>(e),                                                                               \
            std::forward<BS>(block_shape),                                                                    \
            axes,                                                                                             \
            std::forward<O>(options),                                                                         \
            functor_type()                                                                                    \
        );                                                                                                    \
    }                                                                                                         \
    template <                                                                                                \
        class T = void,                                                                                       \
        class E,                                                                                              \
        class BS,                                                                                             \
        class O = DEFAULT_STRATEGY_REDUCERS,                                                                  \
        XTL_REQUIRES(is_reducer_options<O>, std::negation<xtl::is_integral<std::decay_t<O>>>)>                \
    auto FNAME(E&& e, BS&& block_shape, O options = O())                                                      \
    {                                                                                                         \
        using input_expression_type = std::decay_t<E>;                                                        \
        using axes_type = filter_fixed_shape_t<typename input_expression_type::shape_type>;                   \
        axes_type axes = xtl::make_sequence<axes_type>(e.dimension());                                        \
        XTENSOR_ASSERT(axes.size() == e.dimension());                                                         \
        std::iota(axes.begin(), axes.end(), 0);                                                               \
        using functor_type = FUNCTOR<typename input_expression_type::value_type, T>;                          \
        return blockwise_reducer(                                                                             \
            std::forward<E>(e),                                                                               \
            std::forward<BS>(block_shape),                                                                    \
            std::move(axes),                                                                                  \
            std::forward<O>(options),                                                                         \
            functor_type()                                                                                    \
        );                                                                                                    \
    }                                                                                                         \
    template <class T = void, class E, class BS, class I, std::size_t N, class O = DEFAULT_STRATEGY_REDUCERS> \
    auto FNAME(E&& e, BS&& block_shape, const I(&axes)[N], O options = O())                                   \
    {                                                                                                         \
        using input_expression_type = std::decay_t<E>;                                                        \
        using functor_type = FUNCTOR<typename input_expression_type::value_type, T>;                          \
        using axes_type = std::array<std::size_t, N>;                                                         \
        auto ax = xt::forward_normalize<axes_type>(e, axes);                                                  \
        return blockwise_reducer(                                                                             \
            std::forward<E>(e),                                                                               \
            std::forward<BS>(block_shape),                                                                    \
            std::move(ax),                                                                                    \
            std::forward<O>(options),                                                                         \
            functor_type()                                                                                    \
        );                                                                                                    \
    }
        XTENSOR_BLOCKWISE_REDUCER_FUNC(sum, xt::detail::blockwise::sum_functor)
        XTENSOR_BLOCKWISE_REDUCER_FUNC(prod, xt::detail::blockwise::prod_functor)
        XTENSOR_BLOCKWISE_REDUCER_FUNC(amin, xt::detail::blockwise::amin_functor)
        XTENSOR_BLOCKWISE_REDUCER_FUNC(amax, xt::detail::blockwise::amax_functor)
        XTENSOR_BLOCKWISE_REDUCER_FUNC(mean, xt::detail::blockwise::mean_functor)
        XTENSOR_BLOCKWISE_REDUCER_FUNC(variance, xt::detail::blockwise::variance_functor)
        XTENSOR_BLOCKWISE_REDUCER_FUNC(stddev, xt::detail::blockwise::stddev_functor)

#undef XTENSOR_BLOCKWISE_REDUCER_FUNC


// norm reducers do *not* allow to to pass a template
// parameter to specifiy the internal computation type
#define XTENSOR_BLOCKWISE_NORM_REDUCER_FUNC(FNAME, FUNCTOR)                                                                     \
    template <                                                                                                                  \
        class E,                                                                                                                \
        class BS,                                                                                                               \
        class X,                                                                                                                \
        class O = DEFAULT_STRATEGY_REDUCERS,                                                                                    \
        XTL_REQUIRES(std::negation<is_reducer_options<X>>, std::negation<xtl::is_integral<std::decay_t<X>>>)>                   \
    auto FNAME(E&& e, BS&& block_shape, X&& axes, O options = O())                                                              \
    {                                                                                                                           \
        using input_expression_type = std::decay_t<E>;                                                                          \
        using functor_type = FUNCTOR<typename input_expression_type::value_type>;                                               \
        return blockwise_reducer(                                                                                               \
            std::forward<E>(e),                                                                                                 \
            std::forward<BS>(block_shape),                                                                                      \
            std::forward<X>(axes),                                                                                              \
            std::forward<O>(options),                                                                                           \
            functor_type()                                                                                                      \
        );                                                                                                                      \
    }                                                                                                                           \
    template <class E, class BS, class X, class O = DEFAULT_STRATEGY_REDUCERS, XTL_REQUIRES(xtl::is_integral<std::decay_t<X>>)> \
    auto FNAME(E&& e, BS&& block_shape, X axis, O options = O())                                                                \
    {                                                                                                                           \
        std::array<X, 1> axes{axis};                                                                                            \
        using input_expression_type = std::decay_t<E>;                                                                          \
        using functor_type = FUNCTOR<typename input_expression_type::value_type>;                                               \
        return blockwise_reducer(                                                                                               \
            std::forward<E>(e),                                                                                                 \
            std::forward<BS>(block_shape),                                                                                      \
            axes,                                                                                                               \
            std::forward<O>(options),                                                                                           \
            functor_type()                                                                                                      \
        );                                                                                                                      \
    }                                                                                                                           \
    template <                                                                                                                  \
        class E,                                                                                                                \
        class BS,                                                                                                               \
        class O = DEFAULT_STRATEGY_REDUCERS,                                                                                    \
        XTL_REQUIRES(is_reducer_options<O>, std::negation<xtl::is_integral<std::decay_t<O>>>)>                                  \
    auto FNAME(E&& e, BS&& block_shape, O options = O())                                                                        \
    {                                                                                                                           \
        using input_expression_type = std::decay_t<E>;                                                                          \
        using axes_type = filter_fixed_shape_t<typename input_expression_type::shape_type>;                                     \
        axes_type axes = xtl::make_sequence<axes_type>(e.dimension());                                                          \
        XTENSOR_ASSERT(axes.size() == e.dimension());                                                                           \
        std::iota(axes.begin(), axes.end(), 0);                                                                                 \
        using functor_type = FUNCTOR<typename input_expression_type::value_type>;                                               \
        return blockwise_reducer(                                                                                               \
            std::forward<E>(e),                                                                                                 \
            std::forward<BS>(block_shape),                                                                                      \
            std::move(axes),                                                                                                    \
            std::forward<O>(options),                                                                                           \
            functor_type()                                                                                                      \
        );                                                                                                                      \
    }                                                                                                                           \
    template <class E, class BS, class I, std::size_t N, class O = DEFAULT_STRATEGY_REDUCERS>                                   \
    auto FNAME(E&& e, BS&& block_shape, const I(&axes)[N], O options = O())                                                     \
    {                                                                                                                           \
        using input_expression_type = std::decay_t<E>;                                                                          \
        using functor_type = FUNCTOR<typename input_expression_type::value_type>;                                               \
        using axes_type = std::array<std::size_t, N>;                                                                           \
        auto ax = xt::forward_normalize<axes_type>(e, axes);                                                                    \
        return blockwise_reducer(                                                                                               \
            std::forward<E>(e),                                                                                                 \
            std::forward<BS>(block_shape),                                                                                      \
            std::move(ax),                                                                                                      \
            std::forward<O>(options),                                                                                           \
            functor_type()                                                                                                      \
        );                                                                                                                      \
    }
        XTENSOR_BLOCKWISE_NORM_REDUCER_FUNC(norm_l0, xt::detail::blockwise::norm_l0_functor)
        XTENSOR_BLOCKWISE_NORM_REDUCER_FUNC(norm_l1, xt::detail::blockwise::norm_l1_functor)
        XTENSOR_BLOCKWISE_NORM_REDUCER_FUNC(norm_l2, xt::detail::blockwise::norm_l2_functor)
        XTENSOR_BLOCKWISE_NORM_REDUCER_FUNC(norm_sq, xt::detail::blockwise::norm_sq_functor)
        XTENSOR_BLOCKWISE_NORM_REDUCER_FUNC(norm_linf, xt::detail::blockwise::norm_linf_functor)

#undef XTENSOR_BLOCKWISE_NORM_REDUCER_FUNC


#define XTENSOR_BLOCKWISE_NORM_REDUCER_FUNC(FNAME, FUNCTOR)                                                                     \
    template <                                                                                                                  \
        class E,                                                                                                                \
        class BS,                                                                                                               \
        class X,                                                                                                                \
        class O = DEFAULT_STRATEGY_REDUCERS,                                                                                    \
        XTL_REQUIRES(std::negation<is_reducer_options<X>>, std::negation<xtl::is_integral<std::decay_t<X>>>)>                   \
    auto FNAME(E&& e, BS&& block_shape, double p, X&& axes, O options = O())                                                    \
    {                                                                                                                           \
        using input_expression_type = std::decay_t<E>;                                                                          \
        using functor_type = FUNCTOR<typename input_expression_type::value_type>;                                               \
        return blockwise_reducer(                                                                                               \
            std::forward<E>(e),                                                                                                 \
            std::forward<BS>(block_shape),                                                                                      \
            std::forward<X>(axes),                                                                                              \
            std::forward<O>(options),                                                                                           \
            functor_type(p)                                                                                                     \
        );                                                                                                                      \
    }                                                                                                                           \
    template <class E, class BS, class X, class O = DEFAULT_STRATEGY_REDUCERS, XTL_REQUIRES(xtl::is_integral<std::decay_t<X>>)> \
    auto FNAME(E&& e, BS&& block_shape, double p, X axis, O options = O())                                                      \
    {                                                                                                                           \
        std::array<X, 1> axes{axis};                                                                                            \
        using input_expression_type = std::decay_t<E>;                                                                          \
        using functor_type = FUNCTOR<typename input_expression_type::value_type>;                                               \
        return blockwise_reducer(                                                                                               \
            std::forward<E>(e),                                                                                                 \
            std::forward<BS>(block_shape),                                                                                      \
            axes,                                                                                                               \
            std::forward<O>(options),                                                                                           \
            functor_type(p)                                                                                                     \
        );                                                                                                                      \
    }                                                                                                                           \
    template <                                                                                                                  \
        class E,                                                                                                                \
        class BS,                                                                                                               \
        class O = DEFAULT_STRATEGY_REDUCERS,                                                                                    \
        XTL_REQUIRES(is_reducer_options<O>, std::negation<xtl::is_integral<std::decay_t<O>>>)>                                  \
    auto FNAME(E&& e, BS&& block_shape, double p, O options = O())                                                              \
    {                                                                                                                           \
        using input_expression_type = std::decay_t<E>;                                                                          \
        using axes_type = filter_fixed_shape_t<typename input_expression_type::shape_type>;                                     \
        axes_type axes = xtl::make_sequence<axes_type>(e.dimension());                                                          \
        XTENSOR_ASSERT(axes.size() == e.dimension());                                                                           \
        std::iota(axes.begin(), axes.end(), 0);                                                                                 \
        using functor_type = FUNCTOR<typename input_expression_type::value_type>;                                               \
        return blockwise_reducer(                                                                                               \
            std::forward<E>(e),                                                                                                 \
            std::forward<BS>(block_shape),                                                                                      \
            std::move(axes),                                                                                                    \
            std::forward<O>(options),                                                                                           \
            functor_type(p)                                                                                                     \
        );                                                                                                                      \
    }                                                                                                                           \
    template <class E, class BS, class I, std::size_t N, class O = DEFAULT_STRATEGY_REDUCERS>                                   \
    auto FNAME(E&& e, BS&& block_shape, double p, const I(&axes)[N], O options = O())                                           \
    {                                                                                                                           \
        using input_expression_type = std::decay_t<E>;                                                                          \
        using functor_type = FUNCTOR<typename input_expression_type::value_type>;                                               \
        using axes_type = std::array<std::size_t, N>;                                                                           \
        auto ax = xt::forward_normalize<axes_type>(e, axes);                                                                    \
        return blockwise_reducer(                                                                                               \
            std::forward<E>(e),                                                                                                 \
            std::forward<BS>(block_shape),                                                                                      \
            std::move(ax),                                                                                                      \
            std::forward<O>(options),                                                                                           \
            functor_type(p)                                                                                                     \
        );                                                                                                                      \
    }

        XTENSOR_BLOCKWISE_NORM_REDUCER_FUNC(norm_lp_to_p, xt::detail::blockwise::norm_lp_to_p_functor);
        XTENSOR_BLOCKWISE_NORM_REDUCER_FUNC(norm_lp, xt::detail::blockwise::norm_lp_functor);

#undef XTENSOR_BLOCKWISE_NORM_REDUCER_FUNC
    }

}

#endif
