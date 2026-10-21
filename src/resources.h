#pragma once
#include <windows.h>
#include <psapi.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <intrin.h>
#include <cstring>
#include <iomanip>
#include <set>
#include <sstream>
#include "core.h"

namespace wllm {
inline json cpu_capabilities() {
    int f[4]{}; __cpuid(f, 0x80000000);
    char brand[49]{};
    if (uint32_t(f[0]) >= 0x80000004u) {
        for (int i = 0; i < 3; ++i) { __cpuid(f, 0x80000002 + i); std::memcpy(brand + 16*i, f, 16); }
    }
    __cpuid(f, 1);
    const bool os_avx = (uint32_t(f[2]) & (1u << 27)) && (uint32_t(f[2]) & (1u << 28));
    const uint64_t xcr = os_avx ? _xgetbv(0) : 0;
    const bool avx_state = (xcr & 6) == 6;
    __cpuidex(f, 7, 0);
    json result = {{"source", "CPUID/XGETBV/GetLogicalProcessorInformationEx"}, {"brand", brand},
        {"avx2_usable", avx_state && bool(uint32_t(f[1]) & (1u << 5))},
        {"avx512_f_usable", (xcr & 0xe6) == 0xe6 && bool(uint32_t(f[1]) & (1u << 16))},
        {"logical_processors", GetActiveProcessorCount(ALL_PROCESSOR_GROUPS)}, {"physical_cores", nullptr}};
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    if (bytes > 0) {
        std::vector<char> buffer(bytes);
        if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &bytes)) {
            uint32_t cores = 0;
            for (DWORD offset = 0; offset < bytes;) {
                auto * entry = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
                if (!entry->Size || entry->Size > bytes - offset) break;
                if (entry->Relationship == RelationProcessorCore) ++cores;
                offset += entry->Size;
            }
            result["physical_cores"] = cores;
        }
    }
    return result;
}

// These quantities are separate accounting views; never add process, system and GPU usage.
inline json system_resources() {
    PERFORMANCE_INFORMATION p{};
    p.cb = sizeof(p);
    if (!GetPerformanceInfo(&p, sizeof(p))) throw std::runtime_error("GetPerformanceInfo failed");
    SYSTEM_POWER_STATUS power{};
    const bool has_power = GetSystemPowerStatus(&power) != 0;
    return {{"source", "Win32 GetPerformanceInfo/GetSystemPowerStatus"},
        {"physical_total_bytes", uint64_t(p.PhysicalTotal) * p.PageSize},
        {"physical_available_bytes", uint64_t(p.PhysicalAvailable) * p.PageSize},
        {"commit_total_bytes", uint64_t(p.CommitTotal) * p.PageSize},
        {"commit_limit_bytes", uint64_t(p.CommitLimit) * p.PageSize},
        {"commit_headroom_bytes", uint64_t(p.CommitLimit - std::min(p.CommitTotal, p.CommitLimit)) * p.PageSize},
        {"system_cache_bytes", uint64_t(p.SystemCache) * p.PageSize},
        {"process_count", p.ProcessCount},
        {"ac_line_status", has_power && power.ACLineStatus != 255 ? json(power.ACLineStatus) : json(nullptr)},
        {"logical_processors", GetActiveProcessorCount(ALL_PROCESSOR_GROUPS)}};
}

inline std::string adapter_name(const wchar_t * name) {
    const int count = WideCharToMultiByte(CP_UTF8, 0, name, -1, nullptr, 0, nullptr, nullptr);
    if (count <= 0) return {};
    std::string result(count, '\0');
    WideCharToMultiByte(CP_UTF8, 0, name, -1, result.data(), count, nullptr, nullptr);
    result.pop_back();
    return result;
}

inline json dxgi_adapters() {
    using Microsoft::WRL::ComPtr;
    ComPtr<IDXGIFactory1> factory;
    const HRESULT created = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(created)) return {{"source", "DXGI"}, {"status", "unavailable"}, {"hresult", int64_t(created)}, {"devices", json::array()}};
    json devices = json::array();
    std::set<std::string> seen;
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT enumerated = factory->EnumAdapters1(index, &adapter);
        if (enumerated == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(enumerated)) return {{"source", "DXGI"}, {"status", "partial"}, {"hresult", int64_t(enumerated)}, {"devices", devices}};
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc))) continue;
        std::ostringstream id;
        id << std::hex << std::setfill('0') << std::setw(8) << uint32_t(desc.AdapterLuid.HighPart)
           << ':' << std::setw(8) << desc.AdapterLuid.LowPart;
        if (!seen.insert(id.str()).second) continue;
        json item = {{"adapter_luid", id.str()}, {"name", adapter_name(desc.Description)},
            {"vendor_id", desc.VendorId}, {"device_id", desc.DeviceId}, {"subsystem_id", desc.SubSysId},
            {"software", bool(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)},
            {"dedicated_video_bytes", uint64_t(desc.DedicatedVideoMemory)},
            {"dedicated_system_bytes", uint64_t(desc.DedicatedSystemMemory)},
            {"shared_system_limit_bytes", uint64_t(desc.SharedSystemMemory)},
            {"capacity_additive_to_ram", false}, {"model_execution_verified", false}};
        LARGE_INTEGER driver{};
        if (SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver)))
            item["driver_version"] = std::to_string(HIWORD(driver.HighPart)) + "." + std::to_string(LOWORD(driver.HighPart)) + "." +
                std::to_string(HIWORD(driver.LowPart)) + "." + std::to_string(LOWORD(driver.LowPart));
        else item["driver_version"] = nullptr;
        ComPtr<IDXGIAdapter3> adapter3;
        item["probe_process_memory"] = json::object();
        if (SUCCEEDED(adapter.As(&adapter3))) {
            for (const auto group : {DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL}) {
                DXGI_QUERY_VIDEO_MEMORY_INFO info{};
                if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, group, &info))) {
                    item["probe_process_memory"][group == DXGI_MEMORY_SEGMENT_GROUP_LOCAL ? "local" : "nonlocal"] = {
                        {"budget_bytes", info.Budget}, {"current_usage_bytes", info.CurrentUsage},
                        {"current_reservation_bytes", info.CurrentReservation},
                        {"available_for_reservation_bytes", info.AvailableForReservation}};
                }
            }
        }
        devices.push_back(std::move(item));
    }
    return {{"source", "DXGI EnumAdapters1, deduplicated by adapter LUID"}, {"status", "enumerated"}, {"devices", devices},
        {"budget_scope", "QueryVideoMemoryInfo reports this probe process, not global free GPU memory"}};
}
} // namespace wllm
