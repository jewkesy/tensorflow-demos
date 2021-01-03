// Daniel Shiffman
// http://codingtra.in
// http://patreon.com/codingtrain

// Linear Regression with TensorFlow.js
// Video: https://www.youtube.com/watch?v=dLp10CFIvxI

let x_vals = [];
let y_vals = [];

let screenWidth = 600
let screenHeight = 600

let m, b;
let fromDate = "2010-11-30";

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function setup() {
  createCanvas(screenWidth, screenHeight);
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
  fetch("https://e94s2o5jyb.execute-api.eu-west-1.amazonaws.com/prod/getHomePageContent?getdailygames=true&prefix=pc&limit=0&locale=&timefrom="+fromDate)
    .then(response => response.json())
    .then(json => {
      getDailyGamesHistory(json);
    }).catch(error => {
    console.error('There has been a problem with your fetch operation:', error);
  });;  
}

function loss(pred, labels) {
  return pred
    .sub(labels)
    .square()
    .mean();
}

function predict(x) {
  const xs = tf.tensor1d(x);
  // y = mx + b;
  const ys = xs.mul(m).add(b);
  return ys;
}

function mousePressed() {
  // console.log(mouseX, mouseY)
  // let x = map(mouseX, 0, width, 0, 1);
  // let y = map(mouseY, 0, height, 1, 0);
  // console.log(x, y)
  // x_vals.push(x);
  // y_vals.push(y);
  // console.log(x_vals)
}

function draw() {
  tf.tidy(() => {
    if (x_vals.length > 0) {
      const ys = tf.tensor1d(y_vals);
      optimizer.minimize(() => loss(predict(x_vals), ys));
    }
  });

  background(0);

  stroke(255);
  strokeWeight(3);
  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], 0, 1, 0, width);
    let py = map(y_vals[i], 0, 1, height, 0);
    point(px, py);
  }

  const lineX = [0, 1];

  const ys = tf.tidy(() => predict(lineX));
  let lineY = ys.dataSync();
  // console.log(lineY)
  ys.dispose();

  let x1 = map(lineX[0], 0, 1, 0, width);
  let x2 = map(lineX[1], 0, 1, 0, width);

  let y1 = map(lineY[0], 0, 1, height, 0);
  let y2 = map(lineY[1], 0, 1, height, 0);

  strokeWeight(2);
  line(x1, y1, x2, y2);

  // console.log(tf.memory().numTensors);
  // noLoop();
}

function getDailyGamesHistory(source) {
  let dailyGames = [];
  for (var i = 0; i < source.g.length; i++) { 
    var x = source.g[i]
    // console.info(x.year, x.month)
    for (const [key, value] of Object.entries(x)) {
      // if (!value.games) continue;
      // console.log(key)
      if (key.indexOf('d_') == 0) {
        if (!value.games) value.games = 0;
        dailyGames.push({d: key, games:  value.games})
      }
    }
  }

  dailyGames.sort(compare);
  console.log(dailyGames)

  let retVal = [];
  for (var i = 0; i < dailyGames.length; i++) {
    retVal.push(dailyGames[i].games)
  }

  console.log(dailyGames[0], retVal[0])
  console.log(retVal)

  var normVals = normaliseArray(retVal, height)

  var timeline = []
  for (var i = 0; i < normVals.length; i++) {
    timeline.push(i+1)
  }

  timeline = normaliseArray(timeline, width)

  for (var i = 0; i < normVals.length; i++) {
    normVals[i] = height-normVals[i]

    let x = map(timeline[i], 0, width, 0, 1);
    let y = map(normVals[i], 0, height, 1, 0);
    x_vals.push(x)
    y_vals.push(y)
  }


}

function normaliseArray(array, max) {

  var ratio = Math.max.apply(Math, array) / max;

  for (var i = 0; i < array.length; i++) {
      array[i] = Math.round(array[i] / ratio);
  }

  return array;
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
