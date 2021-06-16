let puffinImage;
let mobileNet;

function modelCreated() {
  console.log("Ml5 Model is Ready");
}

function resultsArrived(error, result) {
  if (error) {
    console.error(error);
  } else {
    console.log(result);
    let imageLabel = result[0].label;
    let confidence = result[0].confidence;
    fill(0);
    textSize(50);
    text(imageLabel, 10, height - 100);
    text(confidence, 20, height - 50);
    createP(imageLabel);
    createP(confidence);
  }
}

function imageLoaded() {
  image(puffinImage, 0, 0, width, height);

  mobileNet.predict(puffinImage, resultsArrived);
}
function setup() {
  createCanvas(640, 480);

  puffinImage = createImg("Puffin.jpg", imageLoaded);
  puffinImage.hide();
  background(0);

  mobileNet = ml5.imageClassifier("MobileNet", modelCreated);
}

function draw() {}
