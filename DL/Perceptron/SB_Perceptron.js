// Perceptron is created with n weights and learning constant
class SimplePerceptron {
  constructor(n, learningRate) {
    // Array of weights for inputs
    this.weights = new Array(n);
    // Start with random weights
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = random(-1, 1);
    }
    this.learningRate = learningRate; // learning rate/constant
  }

  // Function to train the Perceptron
  // Weights are adjusted based on "actualOutput" answer
  trainThePerceptron(inputs, actualOutput) {
    // Guess the result
    let guess = this.feedforwardIntoNN(inputs);
    // Compute the factor for changing the weight based on the error
    // Error = desired output - guessed output
    // Note this can only be 0, -2, or 2
    let error = actualOutput - guess;
    // Adjust weights based on weightChange * input
    for (let i = 0; i < this.weights.length; i++) {
      // Multiply by learning constant
      this.weights[i] += this.learningRate * error * inputs[i];
    }
  }

  // Guess -1 or 1 based on input values
  feedforwardIntoNN(inputs) {
    // Sum all values
    let sum = 0;
    for (let i = 0; i < this.weights.length; i++) {
      sum += inputs[i] * this.weights[i];
    }
    // Result is sign of the sum, -1 or 1
    return this.activate(sum);
  }

  activate(sum) {
    if (sum > 0) return 1;
    else return -1;
  }

  // Return weights
  getWeights() {
    return this.weights;
  }
}
