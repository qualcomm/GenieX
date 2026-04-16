/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright Leon Merten Lohse
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_NPY_HPP
#define XTENSOR_NPY_HPP

// Derived from https://github.com/llohse/libnpy by Leon Merten Lohse,
// relicensed from MIT License with permission

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

#include <xtl/xplatform.hpp>
#include <xtl/xsequence.hpp>

#include "../containers/xadapt.hpp"
#include "../containers/xarray.hpp"
#include "../core/xeval.hpp"
#include "../core/xstrides.hpp"
#include "../core/xtensor_config.hpp"

namespace xt
{
    using namespace std::string_literals;

    namespace detail
    {

        const char magic_string[] = "\x93NUMPY";
        const std::size_t magic_string_length = sizeof(magic_string) - 1;

        template <class O>
        inline void write_magic(O& ostream, unsigned char v_major = 1, unsigned char v_minor = 0)
        {
            ostream.write(magic_string, magic_string_length);
            ostream.put(char(v_major));
            ostream.put(char(v_minor));
        }

        inline void read_magic(std::istream& istream, unsigned char* v_major, unsigned char* v_minor)
        {
            std::unique_ptr<char[]> buf(new char[magic_string_length + 2]);
            istream.read(buf.get(), magic_string_length + 2);

            if (!istream)
            {
                XTENSOR_THROW(std::runtime_error, "io error: failed reading file");
            }

            for (std::size_t i = 0; i < magic_string_length; i++)
            {
                if (buf[i] != magic_string[i])
                {
                    XTENSOR_THROW(std::runtime_error, "this file do not have a valid npy format.");
                }
            }

            *v_major = static_cast<unsigned char>(buf[magic_string_length]);
            *v_minor = static_cast<unsigned char>(buf[magic_string_length + 1]);
        }

        template <class T>
        inline char map_type()
        {
            if (std::is_same<T, float>::value)
            {
                return 'f';
            }
            if (std::is_same<T, double>::value)
            {
                return 'f';
            }
            if (std::is_same<T, long double>::value)
            {
                return 'f';
            }

            if (std::is_same<T, char>::value)
            {
                return 'i';
            }
            if (std::is_same<T, signed char>::value)
            {
                return 'i';
            }
            if (std::is_same<T, short>::value)
            {
                return 'i';
            }
            if (std::is_same<T, int>::value)
            {
                return 'i';
            }
            if (std::is_same<T, long>::value)
            {
                return 'i';
            }
            if (std::is_same<T, long long>::value)
            {
                return 'i';
            }

            if (std::is_same<T, unsigned char>::value)
            {
                return 'u';
            }
            if (std::is_same<T, unsigned short>::value)
            {
                return 'u';
            }
            if (std::is_same<T, unsigned int>::value)
            {
                return 'u';
            }
            if (std::is_same<T, unsigned long>::value)
            {
                return 'u';
            }
            if (std::is_same<T, unsigned long long>::value)
            {
                return 'u';
            }

            if (std::is_same<T, bool>::value)
            {
                return 'b';
            }

            if (std::is_same<T, std::complex<float>>::value)
            {
                return 'c';
            }
            if (std::is_same<T, std::complex<double>>::value)
            {
                return 'c';
            }
            if (std::is_same<T, std::complex<long double>>::value)
            {
                return 'c';
            }

            XTENSOR_THROW(std::runtime_error, "Type not known.");
        }

        template <class T>
        inline char get_endianess()
        {
            constexpr char little_endian_char = '<';
            constexpr char big_endian_char = '>';
            constexpr char no_endian_char = '|';

            if (sizeof(T) <= sizeof(char))
            {
                return no_endian_char;
            }

            switch (xtl::endianness())
            {
                case xtl::endian::little_endian:
                    return little_endian_char;
                case xtl::endian::big_endian:
                    return big_endian_char;
                default:
                    return no_endian_char;
            }
        }

        template <class T>
        inline std::string build_typestring()
        {
            std::stringstream ss;
            ss << get_endianess<T>() << map_type<T>() << sizeof(T);
            return ss.str();
        }

        // Safety check function
        inline void parse_typestring(std::string typestring)
        {
            std::regex re("'([<>|])([ifucb])(\\d+)'");
            std::smatch sm;

            std::regex_match(typestring, sm, re);
            if (sm.size() != 4)
            {
                XTENSOR_THROW(std::runtime_error, "invalid typestring");
            }
        }

        // Helpers for the improvised parser
        inline std::string unwrap_s(std::string s, char delim_front, char delim_back)
        {
            if ((s.back() == delim_back) && (s.front() == delim_front))
            {
                return s.substr(1, s.length() - 2);
            }
            else
            {
                XTENSOR_THROW(std::runtime_error, "unable to unwrap");
            }
        }

        inline std::string get_value_from_map(std::string mapstr)
        {
            std::size_t sep_pos = mapstr.find_first_of(":");
            if (sep_pos == std::string::npos)
            {
                return "";
            }

            return mapstr.substr(sep_pos + 1);
        }

        inline void pop_char(std::string& s, char c)
        {
            if (s.back() == c)
            {
                s.pop_back();
            }
        }

        inline void
        parse_header(std::string header, std::string& descr, bool* fortran_order, std::vector<std::size_t>& shape)
        {
            // The first 6 bytes are a magic string: exactly "x93NUMPY".
            //
            // The next 1 byte is an unsigned byte: the major version number of the file
            // format, e.g. x01.
            //
            // The next 1 byte is an unsigned byte: the minor version number of the file
            // format, e.g. x00. Note: the version of the file format is not tied to the
            // version of the NumPy package.
            //
            // The next 2 bytes form a little-endian unsigned short int: the length of the
            // header data HEADER_LEN.
            //
            // The next HEADER_LEN bytes form the header data describing the array's
            // format. It is an ASCII string which contains a Python literal expression of
            // a dictionary. It is terminated by a newline ('n') and padded with spaces
            // ('x20') to make the total length of the magic string + 4 + HEADER_LEN be
            // evenly divisible by 16 for alignment purposes.
            //
            // The dictionary contains three keys:
            //
            // "descr" : dtype.descr
            // An object that can be passed as an argument to the numpy.dtype()
            // constructor to create the array's dtype.
            // "fortran_order" : bool
            // Whether the array data is Fortran-contiguous or not. Since
            // Fortran-contiguous arrays are a common form of non-C-contiguity, we allow
            // them to be written directly to disk for efficiency.
            // "shape" : tuple of int
            // The shape of the array.
            // For repeatability and readability, this dictionary is formatted using
            // pprint.pformat() so the keys are in alphabetic order.

            // remove trailing newline
            if (header.back() != '\n')
            {
                XTENSOR_THROW(std::runtime_error, "invalid header");
            }
            header.pop_back();

            // remove all whitespaces
            header.erase(std::remove(header.begin(), header.end(), ' '), header.end());

            // unwrap dictionary
            header = unwrap_s(header, '{', '}');

            // find the positions of the 3 dictionary keys
            std::size_t keypos_descr = header.find("'descr'");
            std::size_t keypos_fortran = header.find("'fortran_order'");
            std::size_t keypos_shape = header.find("'shape'");

            // make sure all the keys are present
            if (keypos_descr == std::string::npos)
            {
                XTENSOR_THROW(std::runtime_error, "missing 'descr' key");
            }
            if (keypos_fortran == std::string::npos)
            {
                XTENSOR_THROW(std::runtime_error, "missing 'fortran_order' key");
            }
            if (keypos_shape == std::string::npos)
            {
                XTENSOR_THROW(std::runtime_error, "missing 'shape' key");
            }

            // Make sure the keys are in order.
            // Note that this violates the standard, which states that readers *must* not
            // depend on the correct order here.
            // TODO: fix
            if (keypos_descr >= keypos_fortran || keypos_fortran >= keypos_shape)
            {
                XTENSOR_THROW(std::runtime_error, "header keys in wrong order");
            }

            // get the 3 key-value pairs
            std::string keyvalue_descr;
            keyvalue_descr = header.substr(keypos_descr, keypos_fortran - keypos_descr);
            pop_char(keyvalue_descr, ',');

            std::string keyvalue_fortran;
            keyvalue_fortran = header.substr(keypos_fortran, keypos_shape - keypos_fortran);
            pop_char(keyvalue_fortran, ',');

            std::string keyvalue_shape;
            keyvalue_shape = header.substr(keypos_shape, std::string::npos);
            pop_char(keyvalue_shape, ',');

            // get the values (right side of `:')
            std::string descr_s = get_value_from_map(keyvalue_descr);
            std::string fortran_s = get_value_from_map(keyvalue_fortran);
            std::string shape_s = get_value_from_map(keyvalue_shape);

            parse_typestring(descr_s);
            descr = unwrap_s(descr_s, '\'', '\'');

            // convert literal Python bool to C++ bool
            if (fortran_s == "True")
            {
                *fortran_order = true;
            }
            else if (fortran_s == "False")
            {
                *fortran_order = false;
            }
            else
            {
                XTENSOR_THROW(std::runtime_error, "invalid fortran_order value");
            }

            // parse the shape Python tuple ( x, y, z,)

            // first clear the vector
            shape.clear();
            shape_s = unwrap_s(shape_s, '(', ')');

            // a tokenizer would be nice...
            std::size_t pos = 0;
            for (;;)
            {
                std::size_t pos_next = shape_s.find_first_of(',', pos);
                std::string dim_s;

                if (pos_next != std::string::npos)
                {
                    dim_s = shape_s.substr(pos, pos_next - pos);
                }
                else
                {
                    dim_s = shape_s.substr(pos);
                }

                if (dim_s.length() == 0)
                {
                    if (pos_next != std::string::npos)
                    {
                        XTENSOR_THROW(std::runtime_error, "invalid shape");
                    }
                }
                else
                {
                    std::stringstream ss;
                    ss << dim_s;
                    std::size_t tmp;
                    ss >> tmp;
                    shape.push_back(tmp);
                }

                if (pos_next != std::string::npos)
                {
                    pos = ++pos_next;
                }
                else
                {
                    break;
                }
            }
        }

        template <class O, class S>
        inline void write_header(O& out, const std::string& descr, bool fortran_order, const S& shape)
        {
            std::ostringstream ss_header;
            std::string s_fortran_order;
            if (fortran_order)
            {
                s_fortran_order = "True";
            }
            else
            {
                s_fortran_order = "False";
            }

            std::string s_shape;
            std::ostringstream ss_shape;
            ss_shape << "(";
            for (auto shape_it = std::begin(shape); shape_it != std::end(shape); ++shape_it)
            {
                ss_shape << *shape_it << ", ";
            }
            s_shape = ss_shape.str();
            if (std::size(shape) > 1)
            {
                s_shape = s_shape.erase(s_shape.size() - 2);
            }
            else if (std::size(shape) == 1)
            {
                s_shape = s_shape.erase(s_shape.size() - 1);
            }
            s_shape += ")";

            ss_header << "{'descr': '" << descr << "', 'fortran_order': " << s_fortran_order
                      << ", 'shape': " << s_shape << ", }";

            std::size_t header_len_pre = ss_header.str().length() + 1;
            std::size_t metadata_len = magic_string_length + 2 + 2 + header_len_pre;

            unsigned char version[2] = {1, 0};
            if (metadata_len >= 255 * 255)
            {
                metadata_len = magic_string_length + 2 + 4 + header_len_pre;
                version[0] = 2;
                version[1] = 0;
            }
            std::size_t padding_len = 64 - (metadata_len % 64);
            std::string padding(padding_len, ' ');
            ss_header << padding;
            ss_header << std::endl;

            std::string header = ss_header.str();

            // write magic
            write_magic(out, version[0], version[1]);

            // write header length
            if (version[0] == 1 && version[1] == 0)
            {
                char header_len_le16[2];
                uint16_t header_len = uint16_t(header.length());

                header_len_le16[0] = char((header_len >> 0) & 0xff);
                header_len_le16[1] = char((header_len >> 8) & 0xff);
                out.write(reinterpret_cast<char*>(header_len_le16), 2);
            }
            else
            {
                char header_len_le32[4];
                uint32_t header_len = uint32_t(header.length());

                header_len_le32[0] = char((header_len >> 0) & 0xff);
                header_len_le32[1] = char((header_len >> 8) & 0xff);
                header_len_le32[2] = char((header_len >> 16) & 0xff);
                header_len_le32[3] = char((header_len >> 24) & 0xff);
                out.write(reinterpret_cast<char*>(header_len_le32), 4);
            }

            out << header;
        }

        inline std::string read_header_1_0(std::istream& istream)
        {
            // read header length and convert from little endian
            char header_len_le16[2];
            istream.read(header_len_le16, 2);

            uint16_t header_length = uint16_t(header_len_le16[0] << 0) | uint16_t(header_len_le16[1] << 8);

            if ((magic_string_length + 2 + 2 + header_length) % 16 != 0)
            {
                // TODO: display warning
            }

            std::unique_ptr<char[]> buf(new char[header_length]);
            istream.read(buf.get(), header_length);
            std::string header(buf.get(), header_length);

            return header;
        }

        inline std::string read_header_2_0(std::istream& istream)
        {
            // read header length and convert from little endian
            char header_len_le32[4];
            istream.read(header_len_le32, 4);

            uint32_t header_length = uint32_t(header_len_le32[0] << 0) | uint32_t(header_len_le32[1] << 8)
                                     | uint32_t(header_len_le32[2] << 16) | uint32_t(header_len_le32[3] << 24);

            if ((magic_string_length + 2 + 4 + header_length) % 16 != 0)
            {
                // TODO: display warning
            }

            std::unique_ptr<char[]> buf(new char[header_length]);
            istream.read(buf.get(), header_length);
            std::string header(buf.get(), header_length);

            return header;
        }

        struct npy_file
        {
            npy_file() = default;

            npy_file(std::vector<std::size_t>& shape, bool fortran_order, std::string typestring)
                : m_shape(shape)
                , m_fortran_order(fortran_order)
                , m_typestring(typestring)
            {
                // Allocate memory
                m_word_size = std::size_t(atoi(&typestring[2]));
                m_n_bytes = compute_size(shape) * m_word_size;
                m_buffer = std::allocator<char>{}.allocate(m_n_bytes);
            }

            ~npy_file()
            {
                if (m_buffer != nullptr)
                {
                    std::allocator<char>{}.deallocate(m_buffer, m_n_bytes);
                }
            }

            // delete copy constructor
            npy_file(const npy_file&) = delete;
            npy_file& operator=(const npy_file&) = delete;

            // implement move constructor and assignment
            npy_file(npy_file&& rhs)
                : m_shape(std::move(rhs.m_shape))
                , m_fortran_order(std::move(rhs.m_fortran_order))
                , m_word_size(std::move(rhs.m_word_size))
                , m_n_bytes(std::move(rhs.m_n_bytes))
                , m_typestring(std::move(rhs.m_typestring))
                , m_buffer(rhs.m_buffer)
            {
                rhs.m_buffer = nullptr;
            }

            npy_file& operator=(npy_file&& rhs)
            {
                if (this != &rhs)
                {
                    m_shape = std::move(rhs.m_shape);
                    m_fortran_order = std::move(rhs.m_fortran_order);
                    m_word_size = std::move(rhs.m_word_size);
                    m_n_bytes = std::move(rhs.m_n_bytes);
                    m_typestring = std::move(rhs.m_typestring);
                    m_buffer = rhs.m_buffer;
                    rhs.m_buffer = nullptr;
                }
                return *this;
            }

            template <class T, layout_type L>
            auto cast_impl(bool check_type)
            {
                if (m_buffer == nullptr)
                {
                    XTENSOR_THROW(std::runtime_error, "This npy_file has already been cast.");
                }
                T* ptr = reinterpret_cast<T*>(&m_buffer[0]);
                std::vector<std::size_t> strides(m_shape.size());
                std::size_t sz = compute_size(m_shape);

                // check if the typestring matches the given one
                if (check_type && m_typestring != detail::build_typestring<T>())
                {
                    XTENSOR_THROW(
                        std::runtime_error,
                        "Cast error: formats not matching "s + m_typestring + " vs "s
                            + detail::build_typestring<T>()
                    );
                }

                if ((L == layout_type::column_major && !m_fortran_order)
                    || (L == layout_type::row_major && m_fortran_order))
                {
                    XTENSOR_THROW(
                        std::runtime_error,
                        "Cast error: layout mismatch between npy file and requested layout."
                    );
                }

                compute_strides(
                    m_shape,
                    m_fortran_order ? layout_type::column_major : layout_type::row_major,
                    strides
                );
                std::vector<std::size_t> shape(m_shape);

                return std::make_tuple(ptr, sz, std::move(shape), std::move(strides));
            }

            template <class T, layout_type L = layout_type::dynamic>
            auto cast(bool check_type = true) &&
            {
                auto cast_elems = cast_impl<T, L>(check_type);
                m_buffer = nullptr;
                return adapt(
                    std::move(std::get<0>(cast_elems)),
                    std::get<1>(cast_elems),
                    acquire_ownership(),
                    std::get<2>(cast_elems),
                    std::get<3>(cast_elems)
                );
            }

            template <class T, layout_type L = layout_type::dynamic>
            auto cast(bool check_type = true) const&
            {
                auto cast_elems = cast_impl<T, L>(check_type);
                return adapt(
                    std::get<0>(cast_elems),
                    std::get<1>(cast_elems),
                    no_ownership(),
                    std::get<2>(cast_elems),
                    std::get<3>(cast_elems)
                );
            }

            template <class T, layout_type L = layout_type::dynamic>
            auto cast(bool check_type = true) &
            {
                auto cast_elems = cast_impl<T, L>(check_type);
                return adapt(
                    std::get<0>(cast_elems),
                    std::get<1>(cast_elems),
                    no_ownership(),
                    std::get<2>(cast_elems),
                    std::get<3>(cast_elems)
                );
            }

            char* ptr()
            {
                return m_buffer;
            }

            std::size_t n_bytes()
            {
                return m_n_bytes;
            }

            std::vector<std::size_t> m_shape;
            bool m_fortran_order;
            std::size_t m_word_size;
            std::size_t m_n_bytes;
            std::string m_typestring;
            char* m_buffer;
        };

        inline npy_file load_npy_file(std::istream& stream)
        {
            // check magic bytes an version number
            unsigned char v_major, v_minor;
            detail::read_magic(stream, &v_major, &v_minor);

            std::string header;

            if (v_major == 1 && v_minor == 0)
            {
                header = detail::read_header_1_0(stream);
            }
            else if (v_major == 2 && v_minor == 0)
            {
                header = detail::read_header_2_0(stream);
            }
            else
            {
                XTENSOR_THROW(std::runtime_error, "unsupported file format version");
            }

            // parse header
            bool fortran_order;
            std::string typestr;

            std::vector<std::size_t> shape;
            detail::parse_header(header, typestr, &fortran_order, shape);

            npy_file result(shape, fortran_order, typestr);
            // read the data
            stream.read(result.ptr(), std::streamsize((result.n_bytes())));
            return result;
        }

        template <class O, class E>
        inline void dump_npy_stream(O& stream, const xexpression<E>& e)
        {
            using value_type = typename E::value_type;
            const E& ex = e.derived_cast();
            auto&& eval_ex = eval(ex);
            bool fortran_order = false;
            if (eval_ex.layout() == layout_type::column_major && eval_ex.dimension() > 1)
            {
                fortran_order = true;
            }

            std::string typestring = detail::build_typestring<value_type>();

            auto shape = eval_ex.shape();
            detail::write_header(stream, typestring, fortran_order, shape);

            std::size_t size = compute_size(shape);
            stream.write(
                reinterpret_cast<const char*>(eval_ex.data()),
                std::streamsize((sizeof(value_type) * size))
            );
        }
    }  // namespace detail

    /**
     * Save xexpression to NumPy npy format
     *
     * @param filename The filename or path to dump the data
     * @param e the xexpression
     */
    template <typename E>
    inline void dump_npy(const std::string& filename, const xexpression<E>& e)
    {
        std::ofstream stream(filename, std::ofstream::binary);
        if (!stream)
        {
            XTENSOR_THROW(std::runtime_error, "IO Error: failed to open file: "s + filename);
        }

        detail::dump_npy_stream(stream, e);
    }

    /**
     * Save xexpression to NumPy npy format in a string
     *
     * @param e the xexpression
     */
    template <typename E>
    inline std::string dump_npy(const xexpression<E>& e)
    {
        std::stringstream stream;
        detail::dump_npy_stream(stream, e);
        return stream.str();
    }

    /**
     * Loads a npy file (the NumPy storage format)
     *
     * @param stream An input stream from which to load the file
     * @tparam T select the type of the npy file (note: currently there is
     *           no dynamic casting if types do not match)
     * @tparam L select layout_type::column_major if you stored data in
     *           Fortran format
     * @return xarray with contents from npy file
     */
    template <typename T, layout_type L = layout_type::dynamic>
    inline auto load_npy(std::istream& stream)
    {
        detail::npy_file file = detail::load_npy_file(stream);
        return std::move(file).cast<T, L>();
    }

    /**
     * Loads a npy file (the NumPy storage format)
     *
     * @param filename The filename or path to the file
     * @tparam T select the type of the npy file (note: currently there is
     *           no dynamic casting if types do not match)
     * @tparam L select layout_type::column_major if you stored data in
     *           Fortran format
     * @return xarray with contents from npy file
     */
    template <typename T, layout_type L = layout_type::dynamic>
    inline auto load_npy(const std::string& filename)
    {
        std::ifstream stream(filename, std::ifstream::binary);
        if (!stream)
        {
            XTENSOR_THROW(std::runtime_error, "io error: failed to open a file.");
        }
        return load_npy<T, L>(stream);
    }

}  // namespace xt

#endif
