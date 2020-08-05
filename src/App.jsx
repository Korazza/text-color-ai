import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import trainingData from './../../data/training.json';
import './App.scss';

const App = () => {
	const [model, setModel] = useState(new tf.Sequential());
	const [color, setColor] = useState({
		r: 0,
		g: 0,
		b: 0,
	});
	const [prediction, setPrediction] = useState(0);

	const setRandomColor = () => {
		let r = Math.floor(Math.random() * Math.floor(255));
		let g = Math.floor(Math.random() * Math.floor(255));
		let b = Math.floor(Math.random() * Math.floor(255));
		setColor({ r, g, b });
		let output = model.predict(tf.tensor1d([color.r, color.g, color.b]));
		setPrediction(output.read().dataSync());
	};

	const addTrainingColor = (input, output) => {
		trainingData.push({ input, output });
		setRandomColor();
		let X = tf.tensor1d([2, 4]);
		let y = tf.tensor1d([2, 4]);
		model.fit(X, y);
	};

	useEffect(() => {
		setModel(tf.sequential());
		setRandomColor();
	}, []);

	useEffect(() => {
		model.add(
			tf.layers.dense({ units: 12, inputShape: [3, trainingData.length] })
		);
		model.add(tf.layers.dense({ units: 6 }));
		model.add(tf.layers.dense({ units: 1 }));
		model.compile({
			loss: 'meanSquaredError',
			optimizer: 'adam',
		});
	}, [model]);

	return (
		<>
			<div
				style={{ backgroundColor: `rgb(${color.r},${color.g},${color.b})` }}
				className="color"
			>
				<div className="black">Black</div>
				<div className="white">White</div>
				<div
					style={{
						color: prediction > 0.5 ? '#fff' : '#000',
					}}
				>
					Prediction
				</div>
			</div>
			<button onClick={() => addTrainingColor(color, 0)}>Choose Black</button>
			<button onClick={() => addTrainingColor(color, 1)}>Choose White</button>
			<div>
				[{color.r}, {color.g}, {color.b}]
			</div>
			<pre>{JSON.stringify(trainingData)}</pre>
		</>
	);
};

export default App;
