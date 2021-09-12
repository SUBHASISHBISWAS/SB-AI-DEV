class NeuralNetwork
{
  constructor(numI,numH,numO)
  {
    this.input_nodes=numI;
    this.hidden_nodes=numH;
    this.output_nodes=numO;

    this.weights_ih=new Matrix(this.hidden_nodes,this.input_nodes);
    this.weights_ho=new Matrix(this.output_nodes,this.hidden_nodes);

    this.weights_ih.randomize();
    this.weights_ho.randomize();

    this.bias_ih=new Matrix(this.hidden_nodes,1);
    this.bias_ho=new Matrix(this.output_nodes,1);

    this.bias_ih.randomize();
    this.bias_ho.randomize();
    this.learning_rate=0.1;
  }

  feedForward(input_array)
  {
    let inputs = Matrix.convertArrayToMatrix(input_array);
    //inputs.print()
    //Generate Hidden output
    let hidden=Matrix.multiply(this.weights_ih,inputs);
    hidden.add(this.bias_ih);
    hidden.map(this.sigmoid);

    let output=Matrix.multiply(this.weights_ho,hidden);
    output.add(this.bias_ho);
    output.map(this.sigmoid);

    return output.convertMatrixToArray();
    //return guess;
  }

  sigmoid(x)
  {
    return 1/(1+Math.exp(-x));
  }

  dsigmoid(x)
  {
    return x*(1-x);
    //return sigmoid(x)*(1-sigmoid(x))
    // σ(x)*(1-σ(x))
  }

  train(input_array,actual_outputs_array)
  {

      //Convert Array to Single column Matrix
      let inputs = Matrix.convertArrayToMatrix(input_array);
      let actual_outputs=Matrix.convertArrayToMatrix(actual_outputs_array);

      //Generate Hidden output -FeedForward To OutputLayer
      let hidden_layer_outputs=Matrix.multiply(this.weights_ih,inputs);
      hidden_layer_outputs.add(this.bias_ih);
      hidden_layer_outputs.map(this.sigmoid);



      //Generate Output's output-FeedForward To Output
      let guess_outputs=Matrix.multiply(this.weights_ho,hidden_layer_outputs);
      guess_outputs.add(this.bias_ho);
      guess_outputs.map(this.sigmoid);



      //calculating output layer Error =(actual_outputs-guess_outputs)
      let output_layer_errors=Matrix.substract(actual_outputs,guess_outputs);

                  /* Adjust Output Layer Weights through Back-Prpgation*/

      //Calculate Gradient of Output Layer
      let output_gradient=Matrix.map(guess_outputs,this.dsigmoid);
      output_gradient.multiply(output_layer_errors);
      output_gradient.multiply(this.learning_rate);



      // Calculate output layer Delta Weights ∆W
      let hidden_layer_outputs_T=Matrix.transpose(hidden_layer_outputs);
      let weight_ho_deltas=Matrix.multiply(output_gradient,hidden_layer_outputs_T);

      //Adjust the output layer weights W=W+∆W
      this.weights_ho.add(weight_ho_deltas);


            /* Adjust Output Layer Weights through Back-Prpgation*/

      //Calculate Hidden layer output error
      let who_t=Matrix.transpose(this.weights_ho);
      let hidden_layer_errors=Matrix.multiply(who_t,output_layer_errors)


      //Calculate Gradient of Hidden Layer
      let hidden_layer_gradient=Matrix.map(hidden_layer_outputs,this.dsigmoid);
      hidden_layer_gradient.multiply(hidden_layer_errors);
      hidden_layer_gradient.multiply(this.learning_rate);


      // Calculate output layer Delta Weights ∆W
      let inputs_T=Matrix.transpose(inputs);
      let weight_ih_deltas=Matrix.multiply(hidden_layer_gradient,inputs_T);


      //Adjust the hidden layer weights W=W+∆W
      this.weights_ih.add(weight_ih_deltas);

  }
}
