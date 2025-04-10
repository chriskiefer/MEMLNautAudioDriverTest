#include "src/memllib/audio/AudioDriver.hpp"

volatile bool core1_ready = false;

void setup()
{
    Serial.begin(115200);
    Serial.println("Core 0: Waiting for Core 1 setup...");
    while (!core1_ready) {
        delay(1);
    }
    Serial.println("Core 0: Starting main loop");
}

void loop()
{

}

void setup1()
{
    Serial.println("Core 1: Starting setup");
    AudioDriver_Output::Setup();
    AudioDriver_Output::SetCallback(AudioDriver_Output::silence_);
    core1_ready = true;
    Serial.println("Core 1: Setup complete");
}

void loop1()
{

}
