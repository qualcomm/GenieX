/***************************************************************************
* Copyright (c) Sylvain Corlay and Johan Mabille and Wolf Vollprecht       *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_BASE64_HPP
#define XTL_BASE64_HPP

#include <array>
#include <cstddef>
#include <string>

#include "xsequence.hpp"

namespace xtl
{
    inline std::string base64decode(const std::string& input)
    {
        std::array<int, 256> T;
        T.fill(-1);
        for (std::size_t i = 0; i < 64; ++i)
        {
            T[std::size_t("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[i])] = int(i);
        }

        std::string output;
        int val = 0;
        int valb = -8;
        for (char c : input)
        {
            if (T[std::size_t(c)] == -1)
            {
                break;
            }
            val = (val << 6) + T[std::size_t(c)];
            valb += 6;
            if (valb >= 0)
            {
                output.push_back(char((val >> valb) & 0xFF));
                valb -= 8;
            }
        }
        return output;
    }

    inline std::string base64encode(const std::string& input)
    {
        std::string output;
        int val = 0;
        int valb = -6;
        for (char sc : input)
        {
            unsigned char c = static_cast<unsigned char>(sc);
            val = (val << 8) + c;
            valb += 8;
            while (valb >= 0)
            {
                output.push_back("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[(val >> valb) & 0x3F]);
                valb -= 6;
            }
        }
        if (valb > -6)
        {
            output.push_back("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[((val << 8) >> (valb + 8)) & 0x3F]);
        }
        while (output.size() % 4)
        {
            output.push_back('=');
        }
        return output;
    }
}
#endif
