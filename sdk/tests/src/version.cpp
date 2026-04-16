#include "doctest.h"
#include "logging.h"
#include "ml.h"
#include "util.h"

Setup<int, int> setup_guard(SetupMap<int>{}, nullptr, nullptr, nullptr);

TEST_CASE("Version") {
    const char* version = ml_version();
    REQUIRE(version != nullptr);
    GENIEX_LOG_INFO("ML Version: {}", version);
}

TEST_MAIN()
