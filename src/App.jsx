import React, { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import trainingData from './data/training.json';
import './App.scss';

const App = () => {
	const model = useRef(tf.sequential());
	const randomColor = () => {
		return [
			Math.floor(Math.random() * 255),
			Math.floor(Math.random() * 255),
			Math.floor(Math.random() * 255),
		];
	};
	const [color, setColor] = useState([0, 0, 0]);
	const [loss, setLoss] = useState(0);
	const [accuracy, setAccuracy] = useState(0);
	const [prediction, setPrediction] = useState(0);

	useEffect(() => {
		(async () => {
			model.current.add(
				tf.layers.dense({ inputShape: [3], units: 6, activation: 'relu' })
			);
			model.current.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
			model.current.compile({
				loss: 'meanSquaredError',
				optimizer: tf.train.adam(0.05),
				metrics: ['accuracy'],
			});
			await train();
			setColor(randomColor());
		})();
	}, []);

	useEffect(() => {
		tf.engine().startScope();
		const input = tf.tensor2d([color]).div(255.0);
		const output = model.current.predict(input);
		setPrediction(output.dataSync());
		tf.engine().endScope();
	}, [color]);

	const train = async () => {
		tf.engine().startScope();
		const epochs = 200;
		const X = tf.tensor2d(trainingData.map((color) => color.input)).div(255.0);
		const y = tf.tensor1d(trainingData.map((color) => color.output));
		const history = (await model.current.fit(X, y, { epochs })).history;
		setLoss(history.loss[epochs - 1]);
		setAccuracy(history.acc[epochs - 1]);
		tf.engine().endScope();
	};

	const componentToHex = (c) => {
		var hex = c.toString(16);
		return hex.length === 1 ? '0' + hex : hex;
	};

	const rgb2hex = ([r, g, b]) => {
		return '#' + componentToHex(r) + componentToHex(g) + componentToHex(b);
	};

	function hex2rgb(hex) {
		var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
		return result
			? [
					parseInt(result[1], 16),
					parseInt(result[2], 16),
					parseInt(result[3], 16),
			  ]
			: null;
	}

	return (
		<>
			<div
				style={{
					backgroundColor: `rgb(${color[0]},${color[1]},${color[2]})`,
				}}
				className="color"
			>
				<div
					style={{
						color: prediction > 0.5 ? '#fff' : '#000',
					}}
				>
					{prediction > 0.5 ? 'White' : 'Black'}
				</div>
			</div>
			<div className="container">
				<label htmlFor="color">Color</label>
				<input
					type="color"
					id="color"
					value={rgb2hex(color)}
					onChange={(e) => setColor(hex2rgb(e.target.value))}
				/>
				<button onClick={() => setColor(randomColor())}>Random Color</button>
				<div>Prediction: {prediction}</div>
				<div>Loss: {loss * 100}%</div>
				<div>Accuracy: {accuracy * 100}%</div>
			</div>
		</>
	);
};

export default App;
