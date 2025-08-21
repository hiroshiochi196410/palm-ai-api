const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const multer = require('multer');
const sharp = require('sharp');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const upload = multer({ limits: { fileSize: 10 * 1024 * 1024 } });

let model = null;
let loaded = false;

async function loadModel() {
  try {
    console.log('Loading model...');
    const modelPath = path.join(__dirname, 'model.json');
    model = await tf.loadLayersModel(`file://${modelPath}`);
    loaded = true;
    console.log('Model loaded successfully!');
  } catch (error) {
    console.error('Model load error:', error);
  }
}

function generateReading(hand) {
  const readings = {
    'left': 'Your left hand reveals inner potential and creativity.',
    'right': 'Your right hand shows drive and determination.'
  };
  return readings[hand] || 'Great energy detected in your palm.';
}

app.get('/', (req, res) => {
  res.json({ status: 'OK', loaded: loaded });
});

app.get('/health', (req, res) => {
  res.json({ status: loaded ? 'healthy' : 'loading' });
});

app.post('/analyze-palm', upload.single('image'), async (req, res) => {
  if (!loaded) return res.status(503).json({ error: 'Model loading' });
  if (!req.file) return res.status(400).json({ error: 'No image' });
  
  try {
    const buffer = await sharp(req.file.buffer).resize(224, 224).raw().toBuffer();
    const array = new Float32Array(buffer.length);
    for (let i = 0; i < buffer.length; i++) array[i] = buffer[i] / 255.0;
    
    const tensor = tf.tensor4d(array, [1, 224, 224, 3]);
    const prediction = await model.predict(tensor).data();
    const confidence = Math.max(prediction[0], 1 - prediction[0]) * 100;
    const hand = prediction[0] > 0.5 ? 'right' : 'left';
    const handJa = prediction[0] > 0.5 ? '右手' : '左手';
    
    const reading = generateReading(hand);
    
    res.json({
      success: true,
      hand: handJa,
      handEn: hand,
      confidence: Math.round(confidence),
      palmReading: reading,
      lineMessage: `${handJa}を検出！確信度: ${Math.round(confidence)}%\n\n${reading}`
    });
    
    tensor.dispose();
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/test-webhook', (req, res) => {
  res.json({ success: true, message: 'Test OK', data: req.body });
});

loadModel().then(() => {
  app.listen(PORT, () => console.log(`Server: ${PORT}`));
});
