#include "doctest.h"
#include "geniex.h"
#include "logging.h"
#include "util.h"

Setup<int, int> setup_guard(SetupMap<int>{}, nullptr, nullptr, nullptr);

TEST_CASE("Version") {
    const char* version = geniex_version();
    REQUIRE(version != nullptr);
    GENIEX_LOG_INFO("ML Version: {}", version);
}

TEST_MAIN()
