var nn;

function setup() {
	createCanvas(windowWidth, windowHeight);

/*
	brain=new NeuralNetwork(3,4,2)

	var m =new Matrix(2,3);
	m.randomize();
	m.print()

	var n =new Matrix(3,2);
	n.randomize();
	n.print();

	let v=Matrix.multiply(m,n);

	v.print()

	var m =new Matrix(2,2);
	m.randomize();
	m.print()

	function makeDouble(v)
	{
		return v*2;
	}

	m.map(makeDouble)
	m.print()

	console.table(m.data)
	var v=m.transpose();
	console.table(v.data)

	*/

	nn = new NeuralNetwork(2,2,2);
	inputs=[1,0];
	targets=[1,0];
	nn.train(inputs,targets)
}



function draw()
{

}
