#include "src/memllib/audio/AudioDriver.hpp"
#include "src/memllib/memlnaut/Pins.hpp"
#include "src/memllib/memlnaut/MEMLNaut.hpp"
#include "src/memllib/memlnaut/MEMLNautTest.hpp"

#include <memory>

volatile bool core1_ready = false;

void setup()
{
    // Initialise main serial monitor
    Serial.begin(115200);
    // Initialise pins
    Pins::initializePins();
    // Initialise MEMLNaut singleton
    MEMLNaut::Initialize();
    MEMLNautTest::Setup();

    Serial.println("Core 0: Waiting for Core 1 setup...");
    while (!core1_ready) {
        delay(1);
    }
    Serial.println("Core 0: Starting main loop");
}

void loop()
{
    MEMLNaut::Instance()->loop();
    delay(10);
    
    // Print a dot every second
    static unsigned long lastPrint = 0;
    if (lastPrint++ > 100) {
        Serial.println(".");
        lastPrint = 0;
    }
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
