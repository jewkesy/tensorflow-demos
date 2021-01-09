// Daniel Shiffman
// http://codingtra.in
// http://patreon.com/codingtrain

// Polynomial Regression with TensorFlow.js
// Video: https://youtu.be/tIXDik5SGsI

let x_vals = [];
let y_vals = [];

let screenWidth = 800
let screenHeight = 800

let a, b, c, d;
let dragging = false;

let scaleY = 800;

const learningRate = 0.2;
const optimizer = tf.train.adam(learningRate);

let fromDate = "2010-11-30";

let countDiv, predictDiv;
let currCount = 0;

function setup() {
  getData()
  createCanvas(screenWidth, screenHeight);
  a = tf.variable(tf.scalar(random(-1, 1)));
  b = tf.variable(tf.scalar(random(-1, 1)));
  c = tf.variable(tf.scalar(random(-1, 1)));
  d = tf.variable(tf.scalar(random(-1, 1)));
  countDiv = createDiv('Todays Count: ...');
  predictDiv = createDiv('Prediction: ...');
  setInterval(getData, 3000); 
}

function loss(pred, labels) {
  return pred
    .sub(labels)
    .square()
    .mean();
}

function predict(x) {
  const xs = tf.tensor1d(x);
  // y = ax^3 + bx^2 + cx + d
  const ys = xs
    .pow(tf.scalar(3))
    .mul(a)
    .add(xs.square().mul(b))
    .add(xs.mul(c))
    .add(d);
  return ys;
}

function getData() {
  noLoop();  // to reduce cpu
  fetch("https://e94s2o5jyb.execute-api.eu-west-1.amazonaws.com/prod/getHomePageContent?getdailygames=true&prefix=pc&limit=0&locale=&timefrom="+fromDate)
    .then(response => response.json())
    .then(json => {
      getDailyGamesHistory(json);
    }).catch(error => {
    console.error('There has been a problem with your fetch operation:', error);
  });;  
}

function mousePressed() {
  dragging = true;
}

function mouseReleased() {
  dragging = false;
}

function draw() {
  if (dragging) {
    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);
    x_vals.push(x);
    y_vals.push(y);
  } else {
    tf.tidy(() => {
      if (x_vals.length > 0) {
        const ys = tf.tensor1d(y_vals);
        optimizer.minimize(() => loss(predict(x_vals), ys));
      }
    });
  }

  background(0);

  stroke(255);
  strokeWeight(4);
  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], 0, 1, 0, width);
    let py = map(y_vals[i], 0, 1, height, 0);
    if (i == x_vals.length-1) stroke(255,0,0)
    point(px, py);
  }

  const curveX = [];
  for (let x = -1; x <= 1; x += 0.05) {
    curveX.push(x);
  }

  const ys = tf.tidy(() => predict(curveX));
  let curveY = ys.dataSync();
  ys.dispose();

  beginShape();
  noFill();
  stroke(0,255,0);
  strokeWeight(2);
  var lastX;
  var lastY;
  for (let i = 0; i < curveX.length; i++) {
    let x = map(curveX[i], 0, 1, 0, width*1.037);
    let y = map(curveY[i], 0, 1, height, 0);
    vertex(x, y);
    lastY = y
  }
  getNextDayPrediction(lastY)
  endShape();

  // console.log(tf.memory().numTensors);
}

function getNextDayPrediction(yVal) {

  let pix = screenHeight-Math.floor(yVal);

  let predict = Math.floor((scaleY/screenHeight)*pix)
  predictDiv.html('Prediction: ' + predict)
  // console.log(pix, scaleY, predict)
}

function getDailyGamesHistory(source) {
  let dailyGames = [];
  for (var i = 0; i < source.g.length; i++) { 
    var x = source.g[i]
    for (const [key, value] of Object.entries(x)) {
      // if (!value.games) continue;
      if (key.indexOf('d_') == 0) {
        if (!value.games) value.games = 0;
        dailyGames.push({d: key, games:  value.games})
      }
    }
  }

  dailyGames = dailyGames.sort(compare);

  if (dailyGames[dailyGames.length-1].games == currCount) {  // to reduce cpu
    noLoop();
    return;
  } else {
    loop();
    currCount = dailyGames[dailyGames.length-1].games;
    x_vals=[]
    y_vals = []
  }

  countDiv.html('Todays Count: ' + dailyGames[dailyGames.length-1].games);

  let retVal = [];
  for (var i = 0; i < dailyGames.length; i++) {
    retVal.push(dailyGames[i].games)
  }

  scaleY = Math.max.apply(Math, retVal);
  var normVals = normaliseArray(retVal, height)

  var timeline = []
  for (var i = 0; i < normVals.length; i++) {
    timeline.push(i+1)
  }

  timeline = normaliseArray(timeline, width)

  for (var i = 0; i < normVals.length; i++) {
    normVals[i] = height-normVals[i]

    let x = map(timeline[i], 0, width*1.02, 0, 1);
    let y = map(normVals[i], 0, height, 1, 0);
    x_vals.push(x)
    y_vals.push(y)
  }
}

function normaliseArray(array, max) {
  var x = Math.max.apply(Math, array);
  var ratio = x / max;
  var retVal = []
  for (var i = 0; i < array.length; i++) {
    // var a = array[i];
    // var b = Math.round(array[i] / ratio)
    // console.log(a,b)
      retVal.push( Math.round(array[i] / ratio));
  }

  return retVal;
}

function compare( a, b ) {
  if ( a.d < b.d ){
    return -1;
  }
  if ( a.d > b.d ){
    return 1;
  }
  return 0;
}
