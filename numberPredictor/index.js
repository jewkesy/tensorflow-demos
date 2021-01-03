const tf = require('@tensorflow/tfjs-node');
var console = require('tracer').colorConsole();
var fs = require('fs');
var source = JSON.parse(fs.readFileSync('./numberPredictor/stats.json', 'utf8'));


let stats = getDailyGamesHistory(source);

const predict = data => {
  const weights = tf.tensor([2.5])
  const prediction = data.dot(weights)
  return prediction
}

const dailyStuff = stats

const data = tf.tensor([dailyStuff[1]])
data.print()

const prediction = predict(data)


console.log(
  `Predicted next day game count: ${data.dataSync()[0]}}
prediction: ${prediction.dataSync()}
  `
)


function getDailyGamesHistory(source) {
	let dailyGames = [];
	for (var i = 1; i < source.g.length; i++) { //ignore today
		var x = source.g[i]
		// console.info(x.year, x.month)
		for (const [key, value] of Object.entries(x)) {
			if (!value.games) continue;
			// console.log(key)
			if (key.indexOf('d_') == 0) {
				dailyGames.push({d: key, games:  value.games})
			}
		}
	}

	dailyGames.sort(compare);
	// console.log(dailyGames)

	let retVal = [];
	for (var i = 0; i < dailyGames.length; i++) {
		retVal.push(dailyGames[i].games)
	}

	// console.log(retVal);
	return retVal

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

// objs.sort( compare );
