#include "src/memlp/MLP.h"
#include "src/memlp/Dataset.hpp"

// Define minimal network parameters
const int INPUT_SIZE = 2;
const int HIDDEN_SIZE = 2;  // Minimum required for XOR
const int OUTPUT_SIZE = 1;
const float LEARNING_RATE = 0.1f;

// Network architecture
const std::vector<size_t> LAYERS = {INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};
const std::vector<ACTIVATION_FUNCTIONS> ACTIVATIONS = {
    ACTIVATION_FUNCTIONS::RELU,
    ACTIVATION_FUNCTIONS::SIGMOID
};

// XOR training data
const std::vector<std::vector<float>> XOR_INPUTS = {
    {0.0f, 0.0f},
    {0.0f, 1.0f},
    {1.0f, 0.0f},
    {1.0f, 1.0f}
};
const std::vector<std::vector<float>> XOR_OUTPUTS = {
    {0.0f},
    {1.0f},
    {1.0f},
    {0.0f}
};

// Global objects
MLP<float>* mlp;
Dataset dataset;
bool trained = false;

void setup() {
    Serial.begin(115200);
    while (!Serial) {
        ; // Wait for serial port to connect. Needed for native USB port only
    }

    // Initialize the MLP
    mlp = new MLP<float>(LAYERS, ACTIVATIONS);
    
    // Load XOR data into dataset
    dataset.Clear();
    for(size_t i = 0; i < XOR_INPUTS.size(); i++) {
        if(!dataset.Add(XOR_INPUTS[i], XOR_OUTPUTS[i])) {
            Serial.println("Failed to add training example!");
        }
    }

    Serial.println("Training XOR network...");
    
    // Get training data with bias term
    auto training_data = std::make_pair(
        dataset.GetFeatures(true), // true to include bias
        dataset.GetLabels()
    );

    // Train network
    float final_loss = mlp->Train(
        training_data,
        LEARNING_RATE,
        5000,      // max iterations 
        0.001f,    // min error threshold
        true       // log output
    );

    Serial.print("Training complete! Final loss: ");
    Serial.println(final_loss);
    trained = true;
}

void loop() {
    if(!trained) return;

    // Test all XOR combinations
    for(const auto& input : XOR_INPUTS) {
        // Add bias term
        std::vector<float> input_with_bias = input;
        input_with_bias.push_back(1.0f);
        
        // Get network output
        std::vector<float> output;
        mlp->GetOutput(input_with_bias, &output);

        // Print result
        Serial.print(input[0], 1);
        Serial.print(" XOR ");
        Serial.print(input[1], 1);
        Serial.print(" = ");
        Serial.println(output[0], 3);
    }

    delay(2000); // Wait 2 seconds before next test
}
