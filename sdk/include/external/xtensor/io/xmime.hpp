/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_MIME_HPP
#define XTENSOR_MIME_HPP

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../io/xio.hpp"

namespace xt
{
    template <class P, class T>
    void compute_0d_table(std::stringstream& out, P& /*printer*/, const T& expr)
    {
        out << "<table style='border-style:solid;border-width:1px;'><tbody>";
        out << "<tr><td style='font-family:monospace;'><pre>";
        out << expr();
        out << "</pre></td></tr>";
        out << "</tbody></table>";
    }

    template <class P>
    void compute_1d_row(std::stringstream& out, P& printer, const std::size_t& row_idx)
    {
        out << "<tr><td style='font-family:monospace;' title='" << row_idx << "'><pre>";
        printer.print_next(out);
        out << "</pre></td></tr>";
    }

    template <class P, class T>
    void compute_1d_table(std::stringstream& out, P& printer, const T& expr, const std::size_t& edgeitems)
    {
        const auto& dim = expr.shape()[0];

        out << "<table style='border-style:solid;border-width:1px;'><tbody>";
        if (edgeitems == 0 || 2 * edgeitems >= dim)
        {
            for (std::size_t row_idx = 0; row_idx < dim; ++row_idx)
            {
                compute_1d_row(out, printer, row_idx);
            }
        }
        else
        {
            for (std::size_t row_idx = 0; row_idx < edgeitems; ++row_idx)
            {
                compute_1d_row(out, printer, row_idx);
            }
            out << "<tr><td><center>\u22ee</center></td></tr>";
            for (std::size_t row_idx = dim - edgeitems; row_idx < dim; ++row_idx)
            {
                compute_1d_row(out, printer, row_idx);
            }
        }
        out << "</tbody></table>";
    }

    template <class P>
    void compute_2d_element(
        std::stringstream& out,
        P& printer,
        const std::string& idx_str,
        const std::size_t& row_idx,
        const std::size_t& column_idx
    )
    {
        out << "<td style='font-family:monospace;' title='(" << idx_str << row_idx << ", " << column_idx
            << ")'><pre>";
        printer.print_next(out);
        out << "</pre></td>";
    }

    template <class P, class T>
    void compute_2d_row(
        std::stringstream& out,
        P& printer,
        const T& expr,
        const std::size_t& edgeitems,
        const std::string& idx_str,
        const std::size_t& row_idx
    )
    {
        const auto& dim = expr.shape()[expr.dimension() - 1];

        out << "<tr>";
        if (edgeitems == 0 || 2 * edgeitems >= dim)
        {
            for (std::size_t column_idx = 0; column_idx < dim; ++column_idx)
            {
                compute_2d_element(out, printer, idx_str, row_idx, column_idx);
            }
        }
        else
        {
            for (std::size_t column_idx = 0; column_idx < edgeitems; ++column_idx)
            {
                compute_2d_element(out, printer, idx_str, row_idx, column_idx);
            }
            out << "<td><center>\u22ef</center></td>";
            for (std::size_t column_idx = dim - edgeitems; column_idx < dim; ++column_idx)
            {
                compute_2d_element(out, printer, idx_str, row_idx, column_idx);
            }
        }
        out << "</tr>";
    }

    template <class P, class T, class I>
    void compute_2d_table(
        std::stringstream& out,
        P& printer,
        const T& expr,
        const std::size_t& edgeitems,
        const std::vector<I>& idx
    )
    {
        const auto& dim = expr.shape()[expr.dimension() - 2];
        const auto& last_dim = expr.shape()[expr.dimension() - 1];
        std::string idx_str;
        std::for_each(
            idx.cbegin(),
            idx.cend(),
            [&idx_str](const auto& i)
            {
                idx_str += std::to_string(i) + ", ";
            }
        );

        std::size_t nb_ellipsis = 2 * edgeitems + 1;
        if (last_dim <= 2 * edgeitems + 1)
        {
            nb_ellipsis = last_dim;
        }

        out << "<table style='border-style:solid;border-width:1px;'><tbody>";
        if (edgeitems == 0 || 2 * edgeitems >= dim)
        {
            for (std::size_t row_idx = 0; row_idx < dim; ++row_idx)
            {
                compute_2d_row(out, printer, expr, edgeitems, idx_str, row_idx);
            }
        }
        else
        {
            for (std::size_t row_idx = 0; row_idx < edgeitems; ++row_idx)
            {
                compute_2d_row(out, printer, expr, edgeitems, idx_str, row_idx);
            }
            out << "<tr>";
            for (std::size_t column_idx = 0; column_idx < nb_ellipsis; ++column_idx)
            {
                if (column_idx == edgeitems && nb_ellipsis != last_dim)
                {
                    out << "<td><center>\u22f1</center></td>";
                }
                else
                {
                    out << "<td><center>\u22ee</center></td>";
                }
            }
            out << "</tr>";
            for (std::size_t row_idx = dim - edgeitems; row_idx < dim; ++row_idx)
            {
                compute_2d_row(out, printer, expr, edgeitems, idx_str, row_idx);
            }
        }
        out << "</tbody></table>";
    }

    template <class P, class T, class I>
    void compute_nd_row(
        std::stringstream& out,
        P& printer,
        const T& expr,
        const std::size_t& edgeitems,
        const std::vector<I>& idx
    )
    {
        out << "<tr><td>";
        compute_nd_table_impl(out, printer, expr, edgeitems, idx);
        out << "</td></tr>";
    }

    template <class P, class T, class I>
    void compute_nd_table_impl(
        std::stringstream& out,
        P& printer,
        const T& expr,
        const std::size_t& edgeitems,
        const std::vector<I>& idx
    )
    {
        const auto& displayed_dimension = idx.size();
        const auto& expr_dim = expr.dimension();
        const auto& dim = expr.shape()[displayed_dimension];

        if (expr_dim - displayed_dimension == 2)
        {
            return compute_2d_table(out, printer, expr, edgeitems, idx);
        }

        std::vector<I> idx2 = idx;
        idx2.resize(displayed_dimension + 1);

        out << "<table style='border-style:solid;border-width:1px;'>";
        if (edgeitems == 0 || 2 * edgeitems >= dim)
        {
            for (std::size_t i = 0; i < dim; ++i)
            {
                idx2[displayed_dimension] = i;
                compute_nd_row(out, printer, expr, edgeitems, idx2);
            }
        }
        else
        {
            for (std::size_t i = 0; i < edgeitems; ++i)
            {
                idx2[displayed_dimension] = i;
                compute_nd_row(out, printer, expr, edgeitems, idx2);
            }
            out << "<tr><td><center>\u22ef</center></td></tr>";
            for (std::size_t i = dim - edgeitems; i < dim; ++i)
            {
                idx2[displayed_dimension] = i;
                compute_nd_row(out, printer, expr, edgeitems, idx2);
            }
        }
        out << "</table>";
    }

    template <class P, class T>
    void compute_nd_table(std::stringstream& out, P& printer, const T& expr, const std::size_t& edgeitems)
    {
        if (expr.dimension() == 0)
        {
            compute_0d_table(out, printer, expr);
        }
        else if (expr.dimension() == 1)
        {
            compute_1d_table(out, printer, expr, edgeitems);
        }
        else
        {
            std::vector<std::size_t> empty_vector;
            compute_nd_table_impl(out, printer, expr, edgeitems, empty_vector);
        }
    }

    template <class E>
    nlohmann::json mime_bundle_repr_impl(const E& expr)
    {
        std::stringstream out;

        std::size_t edgeitems = 0;
        std::size_t size = compute_size(expr.shape());
        if (size > static_cast<std::size_t>(print_options::print_options().threshold))
        {
            edgeitems = static_cast<std::size_t>(print_options::print_options().edge_items);
        }

        if (print_options::print_options().precision != -1)
        {
            out.precision(print_options::print_options().precision);
        }

        detail::printer<E> printer(out.precision());

        xstrided_slice_vector slice_vector;
        detail::recurser_run(printer, expr, slice_vector, edgeitems);
        printer.init();

        compute_nd_table(out, printer, expr, edgeitems);

        auto bundle = nlohmann::json::object();
        bundle["text/html"] = out.str();
        return bundle;
    }

    template <class F, class CT>
    class xfunctor_view;

    template <class F, class CT>
    nlohmann::json mime_bundle_repr(const xfunctor_view<F, CT>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class F, class... CT>
    class xfunction;

    template <class F, class... CT>
    nlohmann::json mime_bundle_repr(const xfunction<F, CT...>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class EC, layout_type L, class SC, class Tag>
    class xarray_container;

    template <class EC, layout_type L, class SC, class Tag>
    nlohmann::json mime_bundle_repr(const xarray_container<EC, L, SC, Tag>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    class xtensor_container;

    template <class EC, std::size_t N, layout_type L, class Tag>
    nlohmann::json mime_bundle_repr(const xtensor_container<EC, N, L, Tag>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class ET, class S, layout_type L, bool SH, class Tag>
    class xfixed_container;

    template <class ET, class S, layout_type L, bool SH, class Tag>
    nlohmann::json mime_bundle_repr(const xfixed_container<ET, S, L, SH, Tag>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class F, class CT, class X, class O>
    class xreducer;

    template <class F, class CT, class X, class O>
    nlohmann::json mime_bundle_repr(const xreducer<F, CT, X, O>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class VE, class FE>
    class xoptional_assembly;

    template <class VE, class FE>
    nlohmann::json mime_bundle_repr(const xoptional_assembly<VE, FE>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class VEC, class FEC>
    class xoptional_assembly_adaptor;

    template <class VEC, class FEC>
    nlohmann::json mime_bundle_repr(const xoptional_assembly_adaptor<VEC, FEC>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class CT>
    class xscalar;

    template <class CT>
    nlohmann::json mime_bundle_repr(const xscalar<CT>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class CT, class X>
    class xbroadcast;

    template <class CT, class X>
    nlohmann::json mime_bundle_repr(const xbroadcast<CT, X>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class F, class R, class S>
    class xgenerator;

    template <class F, class R, class S>
    nlohmann::json mime_bundle_repr(const xgenerator<F, R, S>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class CT, class... S>
    class xview;

    template <class CT, class... S>
    nlohmann::json mime_bundle_repr(const xview<CT, S...>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class CT, class S, layout_type L, class FST>
    class xstrided_view;

    template <class CT, class S, layout_type L, class FST>
    nlohmann::json mime_bundle_repr(const xstrided_view<CT, S, L, FST>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class CTD, class CTM>
    class xmasked_view;

    template <class CTD, class CTM>
    nlohmann::json mime_bundle_repr(const xmasked_view<CTD, CTM>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class T, class B>
    class xmasked_value;

    template <class T, class B>
    nlohmann::json mime_bundle_repr(const xmasked_value<T, B>& v)
    {
        auto bundle = nlohmann::json::object();
        std::stringstream tmp;
        tmp << v;
        bundle["text/plain"] = tmp.str();
        return bundle;
    }
}

#endif
