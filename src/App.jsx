import React, { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import './App.css';

const App = () => {
	const model = useRef(null);
	const randomColor = () => {
		return [
			Math.floor(Math.random() * 255),
			Math.floor(Math.random() * 255),
			Math.floor(Math.random() * 255),
		];
	};
	const [color, setColor] = useState([255, 255, 255]);
	const [prediction, setPrediction] = useState(0);

	useEffect(() => {
		(async () => {
			model.current = await tf.loadLayersModel('./model/model.json');
			console.log(model.current);
			setColor(randomColor());
		})();
	}, []);

	useEffect(() => {
		(async () => {
			if (!model.current) return;
			tf.engine().startScope();
			const input = tf.tensor2d([color]).div(255.0);
			const output = await model.current.predict(input).data();
			console.log(output);
			setPrediction(output);
			tf.engine().endScope();
		})();
	}, [color]);

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
			</div>
		</>
	);
};

export default App;
