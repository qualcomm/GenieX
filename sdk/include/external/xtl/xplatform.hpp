/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_XPLATFORM_HPP
#define XTL_XPLATFORM_HPP

#include <cstring>
#include <cstdint>

namespace xtl
{
    enum class endian
    {
        big_endian,
        little_endian,
        mixed
    };

    inline endian endianness()
    {
        uint32_t utmp = 0x01020304;
        char btmp[sizeof(utmp)];
        std::memcpy(&btmp[0], &utmp, sizeof(utmp));
        switch(btmp[0])
        {
        case 0x01:
            return endian::big_endian;
        case 0x04:
            return endian::little_endian;
        default:
            return endian::mixed;
        }
    }
}

#endif
