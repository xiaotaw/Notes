#include <soundio/soundio.h>
#include <map>
#include <regex>
#include <string>
#include <iostream>
/*
struct SoundIoDeviceDeleter
{
    void operator()(SoundIoDevice *s) const
    {
        soundio_device_unref(s);
    }
};
*/

int main()
{    
    enum SoundIoBackend backend = SoundIoBackendNone;
    struct SoundIo *soundio = soundio_create();
    if(backend == SoundIoBackendNone){
        soundio_connect(soundio);
    }
    else{
        soundio_connect_backend(soundio, backend);
    }

    soundio_flush_events(soundio);
    
    const int inputCount = soundio_input_device_count(soundio);
    for (int i = 0; i < inputCount; i++)
    {
        struct SoundIoDevice *device = soundio_get_input_device(soundio, i);
        if (device)
        {
            // Each device is listed twice - a 'raw' device and a not-'raw' device.
            // We only want the non-raw ones.
            if (device->is_raw)
            {
                continue;
            }

            // On ALSA/Pulse, the device ID contains the serial number, 
            // so we can just extract it from the name
            /*
            static const std::regex nameRegex(".*Kinect.*_([0-9]+)-.*");
            std::cmatch match;
            if (!std::regex_match(device->id, match, nameRegex))
            {
                continue;
            }

            foundDevices = true;

            (*result)[device->id] = match.str(1);
            */
            std::cout << device->id << std::endl;
        }
        soundio_device_unref(device);
    }
    //std::map<std::string, std::string> soundio_id2sn;
    //GetSoundIoBackendIdToSerialNumberMapping(soundio, &soundio_id2sn);
    return 0;
}
