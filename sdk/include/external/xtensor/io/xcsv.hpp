/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_CSV_HPP
#define XTENSOR_CSV_HPP

#include <exception>
#include <istream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>

#include "../containers/xtensor.hpp"
#include "../core/xtensor_config.hpp"

namespace xt
{

    /**************************************
     * load_csv and dump_csv declarations *
     **************************************/

    template <class T, class A = std::allocator<T>>
    using xcsv_tensor = xtensor_container<std::vector<T, A>, 2, layout_type::row_major>;

    template <class T, class A = std::allocator<T>>
    xcsv_tensor<T, A> load_csv(
        std::istream& stream,
        const char delimiter = ',',
        const std::size_t skip_rows = 0,
        const std::ptrdiff_t max_rows = -1,
        const std::string comments = "#"
    );

    template <class E>
    void dump_csv(std::ostream& stream, const xexpression<E>& e);

    /*****************************************
     * load_csv and dump_csv implementations *
     *****************************************/

    namespace detail
    {
        template <class T>
        inline T lexical_cast(const std::string& cell)
        {
            T res;
            std::istringstream iss(cell);
            iss >> res;
            return res;
        }

        template <>
        inline std::string lexical_cast(const std::string& cell)
        {
            size_t first = cell.find_first_not_of(' ');
            if (first == std::string::npos)
            {
                return cell;
            }

            size_t last = cell.find_last_not_of(' ');
            return cell.substr(first, last == std::string::npos ? cell.size() : last + 1);
        }

        template <>
        inline float lexical_cast<float>(const std::string& cell)
        {
            return std::stof(cell);
        }

        template <>
        inline double lexical_cast<double>(const std::string& cell)
        {
            return std::stod(cell);
        }

        template <>
        inline long double lexical_cast<long double>(const std::string& cell)
        {
            return std::stold(cell);
        }

        template <>
        inline int lexical_cast<int>(const std::string& cell)
        {
            return std::stoi(cell);
        }

        template <>
        inline long lexical_cast<long>(const std::string& cell)
        {
            return std::stol(cell);
        }

        template <>
        inline long long lexical_cast<long long>(const std::string& cell)
        {
            return std::stoll(cell);
        }

        template <>
        inline unsigned int lexical_cast<unsigned int>(const std::string& cell)
        {
            return static_cast<unsigned int>(std::stoul(cell));
        }

        template <>
        inline unsigned long lexical_cast<unsigned long>(const std::string& cell)
        {
            return std::stoul(cell);
        }

        template <>
        inline unsigned long long lexical_cast<unsigned long long>(const std::string& cell)
        {
            return std::stoull(cell);
        }

        template <class ST, class T, class OI>
        ST load_csv_row(std::istream& row_stream, OI output, std::string cell, const char delimiter = ',')
        {
            ST length = 0;
            while (std::getline(row_stream, cell, delimiter))
            {
                *output++ = lexical_cast<T>(cell);
                ++length;
            }
            return length;
        }
    }

    /**
     * @brief Load tensor from CSV.
     *
     * Returns an \ref xexpression for the parsed CSV
     * @param stream the input stream containing the CSV encoded values
     * @param delimiter the character used to separate values. [default: ',']
     * @param skip_rows the number of lines to skip from the beginning. [default: 0]
     * @param max_rows the number of lines to read after skip_rows lines; the default is to read all the
     * lines. [default: -1]
     * @param comments the string used to indicate the start of a comment. [default: "#"]
     */
    template <class T, class A>
    xcsv_tensor<T, A> load_csv(
        std::istream& stream,
        const char delimiter,
        const std::size_t skip_rows,
        const std::ptrdiff_t max_rows,
        const std::string comments
    )
    {
        using tensor_type = xcsv_tensor<T, A>;
        using storage_type = typename tensor_type::storage_type;
        using size_type = typename tensor_type::size_type;
        using inner_shape_type = typename tensor_type::inner_shape_type;
        using inner_strides_type = typename tensor_type::inner_strides_type;
        using output_iterator = std::back_insert_iterator<storage_type>;

        storage_type data;
        size_type nbrow = 0, nbcol = 0, nhead = 0;
        {
            output_iterator output(data);
            std::string row, cell;
            while (std::getline(stream, row))
            {
                if (nhead < skip_rows)
                {
                    ++nhead;
                    continue;
                }
                if (std::equal(comments.begin(), comments.end(), row.begin()))
                {
                    continue;
                }
                if (0 < max_rows && max_rows <= static_cast<const long long>(nbrow))
                {
                    break;
                }
                std::stringstream row_stream(row);
                nbcol = detail::load_csv_row<size_type, T, output_iterator>(row_stream, output, cell, delimiter);
                ++nbrow;
            }
        }
        inner_shape_type shape = {nbrow, nbcol};
        inner_strides_type strides;  // no need for initializer list for stack-allocated strides_type
        size_type data_size = compute_strides(shape, layout_type::row_major, strides);
        // Sanity check for data size.
        if (data.size() != data_size)
        {
            XTENSOR_THROW(std::runtime_error, "Inconsistent row lengths in CSV");
        }
        return tensor_type(std::move(data), std::move(shape), std::move(strides));
    }

    /**
     * @brief Dump tensor to CSV.
     *
     * @param stream the output stream to write the CSV encoded values
     * @param e the tensor expression to serialize
     */
    template <class E>
    void dump_csv(std::ostream& stream, const xexpression<E>& e)
    {
        using size_type = typename E::size_type;
        const E& ex = e.derived_cast();
        if (ex.dimension() != 2)
        {
            XTENSOR_THROW(std::runtime_error, "Only 2-D expressions can be serialized to CSV");
        }
        size_type nbrows = ex.shape()[0], nbcols = ex.shape()[1];
        auto st = ex.stepper_begin(ex.shape());
        for (size_type r = 0; r != nbrows; ++r)
        {
            for (size_type c = 0; c != nbcols; ++c)
            {
                stream << *st;
                if (c != nbcols - 1)
                {
                    st.step(1);
                    stream << ',';
                }
                else
                {
                    st.reset(1);
                    st.step(0);
                    stream << std::endl;
                }
            }
        }
    }

    struct xcsv_config
    {
        char delimiter;
        std::size_t skip_rows;
        std::ptrdiff_t max_rows;
        std::string comments;

        xcsv_config()
            : delimiter(',')
            , skip_rows(0)
            , max_rows(-1)
            , comments("#")
        {
        }
    };

    template <class E>
    void load_file(std::istream& stream, xexpression<E>& e, const xcsv_config& config)
    {
        e.derived_cast() = load_csv<typename E::value_type>(
            stream,
            config.delimiter,
            config.skip_rows,
            config.max_rows,
            config.comments
        );
    }

    template <class E>
    void dump_file(std::ostream& stream, const xexpression<E>& e, const xcsv_config&)
    {
        dump_csv(stream, e);
    }
}

#endif
