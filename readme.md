```bash
    sudo apt-get install python3-tk
```

```bash
    sudo apt install pulseaudio pulseaudio-utils
```


```bash
    sudo apt install alsa-base alsa-utils
```



sudo chmod -R 777 main.py

import sounddevice as sd

print(sd.query_hostapis())
print(sd.query_devices())
