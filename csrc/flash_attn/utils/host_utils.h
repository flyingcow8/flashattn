#pragma once


namespace flash {

inline __host__ mcDeviceProp_t mcGetCurrentDeviceProperties() {
    int deviceId{};
    mcGetDevice(&deviceId);
    mcDeviceProp_t dprops;
    mcGetDeviceProperties(&dprops, deviceId);
    return dprops;
}

}
