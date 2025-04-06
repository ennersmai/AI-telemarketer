import sounddevice as sd
import numpy as np
import time

def audio_callback(indata, frames, time_info, status):
    volume_norm = np.linalg.norm(indata) * 10
    print(f"Audio level: {volume_norm:.2f}")

# Get all devices with input channels
devices = sd.query_devices()
input_devices = [i for i, dev in enumerate(devices) if dev['max_input_channels'] > 0]

print(f"Found {len(input_devices)} input devices:")
for i in input_devices:
    dev = sd.query_devices(i)
    print(f"Device {i}: {dev['name']} - {dev['max_input_channels']} input channels")

for device_idx in input_devices:
    dev = sd.query_devices(device_idx)
    print(f"\nTesting device {device_idx}: {dev['name']}")
    default_sr = int(dev['default_samplerate'])
    print(f"Default Sample Rate: {default_sr} Hz")
    print(f"Channels: {dev['max_input_channels']}")
    try:
        with sd.InputStream(device=device_idx, 
                           channels=min(dev['max_input_channels'], 2),
                           callback=audio_callback, 
                           blocksize=1024, 
                           samplerate=default_sr):
            print(f"Recording from {dev['name']} (at {default_sr} Hz) for 3 seconds...")
            for i in range(3):
                time.sleep(1)
                print(f"Testing... {i+1}s")
        print(f"Device {device_idx} test completed.")
    except Exception as e:
        print(f"Error testing device {device_idx}: {e}")
    
print("\nAll device tests completed.")
