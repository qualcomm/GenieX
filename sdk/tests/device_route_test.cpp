// SPDX-License-Identifier: BSD-3-Clause
#include "geniex.h"
#include "logging.h"
#include <cstdlib>
#include <iostream>

geniex_log_callback geniex_log = nullptr;
geniex_LogLevel geniex_log_level = GENIEX_LOG_LEVEL_TRACE;
extern "C" void geniex_free(void* ptr) { std::free(ptr); }
namespace { int failures=0; void check(bool v,const char* e,int l){if(!v){++failures;std::cerr<<"CHECK failed "<<l<<": "<<e<<'\n';}} }
#define CHECK(x) check(!!(x),#x,__LINE__)
int main(){
 geniex_ResolveDeviceInput in{"llama_cpp",nullptr,"gpu",-1}; geniex_ResolveDeviceOutput out{};
 CHECK(geniex_resolve_device(&in,&out)==GENIEX_SUCCESS);
 CHECK(out.requested_route==GENIEX_ROUTE_GPU); CHECK(out.selected_route==GENIEX_ROUTE_GPU);
 if(out.device_id) geniex_free(out.device_id); if(out.warning) geniex_free(out.warning);
 in.plugin_id="qairt"; in.mode="cpu"; out={};
 CHECK(geniex_resolve_device(&in,&out)==GENIEX_ERROR_COMMON_INVALID_DEVICE);
 in.mode="auto"; out={}; CHECK(geniex_resolve_device(&in,&out)==GENIEX_SUCCESS);
 CHECK(out.requested_route==GENIEX_ROUTE_AUTO); CHECK(out.selected_route==GENIEX_ROUTE_NPU);
 if(out.device_id) geniex_free(out.device_id); if(out.warning) geniex_free(out.warning);
 return failures?1:0;
}
