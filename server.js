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
app.post('/line-webhook', async (req, res) => {
  try {
    const events = req.body.events || [];
    
    for (const event of events) {
      if (event.type === 'message') {
        if (event.message.type === 'image') {
          await handlePalmReading(event);
        } else if (event.message.type === 'text') {
          await handleTextMessage(event);
        }
      }
    }
    
    res.sendStatus(200);
  } catch (error) {
    console.error('LINE webhook error:', error);
    res.sendStatus(500);
  }
});

// 画像メッセージ処理（手相鑑定）
async function handlePalmReading(event) {
  try {
    // 1. LINE画像取得
    const imageResponse = await fetch(
      `https://api-data.line.me/v2/bot/message/${event.message.id}/content`,
      {
        headers: { 'Authorization': `Bearer ${process.env.LINE_ACCESS_TOKEN}` }
      }
    );
    
    if (!imageResponse.ok) {
      throw new Error('Failed to get image from LINE');
    }
    
    const imageBuffer = await imageResponse.arrayBuffer();
    
    // 2. 既存のpalm解析APIを内部呼び出し（multerを使わずに直接処理）
   const buffer = await sharp(Buffer.from(imageBuffer)).resize(224, 224).raw().toBuffer();
const array = new Float32Array(buffer.length);
for (let i = 0; i < buffer.length; i++) {
  array[i] = buffer[i] / 255.0;    }
    
    // TensorFlowモデルで予測（既存のロジックを使用）
    if (!loaded) {
      throw new Error('Model not loaded');
    }
    
    const tensor = tf.tensor4d(array, [1, 224, 224, 3]);
    const prediction = await model.predict(tensor).data();
    const confidence = Math.max(prediction[0], 1 - prediction[0]) * 100;
    const hand = prediction[0] > 0.5 ? 'right' : 'left';
    const handJa = prediction[0] > 0.5 ? '右手' : '左手';
    
    const palmData = {
      success: true,
      hand: handJa,
      handEn: hand,
      confidence: Math.round(confidence),
      palmReading: generateReading(hand)
    };
    
    // 3. ChatGPT鑑定
    const fortuneResult = await getChatGPTFortune(palmData);
    
    // 4. LINE返信
    await replyToLine(event.replyToken, fortuneResult);
    
  } catch (error) {
    console.error('Palm reading error:', error);
    await replyToLine(event.replyToken, 
      '申し訳ございません。手相の解析中にエラーが発生しました。🙏\n\n' +
      '📸 撮影のコツ：\n' +
      '• 明るい場所で撮影\n' +
      '• 手のひら全体が写るように\n' +
      '• ピンぼけしないよう注意\n\n' +
      'もう一度お試しください。'
    );
  }
}

// テキストメッセージ処理
async function handleTextMessage(event) {
  const text = event.message.text.toLowerCase();
  let replyText = '';
  
  if (text.includes('こんにち') || text.includes('はじめ') || text.includes('hello')) {
    replyText = 'こんにちは！手相鑑定botです✋\n\n' +
               '手のひらの写真を送っていただければ、AI分析による詳しい手相鑑定をお返しします。\n\n' +
               '📝 撮影のコツ：\n' +
               '• 明るい場所で撮影\n' +
               '• 手のひら全体が写るように\n' +
               '• ピンぼけしないよう注意\n\n' +
               'それでは、お手相を拝見させていただきますね！🔮';
  } else if (text.includes('使い方') || text.includes('ヘルプ') || text.includes('help')) {
    replyText = '📋 手相鑑定botの使い方\n\n' +
               '1️⃣ 手のひらの写真を撮影\n' +
               '2️⃣ このチャットに写真を送信\n' +
               '3️⃣ AI分析による鑑定結果をお返しします\n\n' +
               '💡 より正確な鑑定のために：\n' +
               '• 自然光で撮影（室内灯でもOK）\n' +
               '• 手を平らに広げて\n' +
               '• 指先から手首まで全体を写す\n' +
               '• ブレないよう固定して撮影\n\n' +
               '準備ができましたら、写真をお送りください📷';
  } else {
    replyText = 'お疲れさまです！✨\n\n' +
               '手相鑑定をご希望でしたら、手のひらの写真を送ってくださいね📸\n\n' +
               '使い方が分からない場合は「ヘルプ」と送信してください。';
  }
  
  await replyToLine(event.replyToken, replyText);
}

// ChatGPT鑑定
async function getChatGPTFortune(palmData) {
  try {
    const prompt = `
手相鑑定をお願いします。

【解析データ】
手: ${palmData.hand}
信頼度: ${palmData.confidence}%
基本鑑定: ${palmData.palmReading}

【鑑定スタイル】
- 温かく親しみやすい関西弁混じりの口調
- まず良い面を伝えてから、建設的なアドバイス
- 具体的で実践的な内容を含める
- 希望を持てるような前向きな結論で締める

【含めてほしい内容】
🌟 あなたの手相の特徴
💪 生命線：健康運・生命力
🧠 知能線：才能・適職
❤️ 感情線：恋愛・人間関係
🍀 総合運とアドバイス

【文字数】400-500文字程度

親身になって鑑定してください。
`;

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'gpt-4',
        messages: [
          {
            role: 'system',
            content: 'あなたは30年の経験を持つ親しみやすい手相鑑定師です。温かく、希望に満ちた鑑定を心がけています。'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        max_tokens: 800,
        temperature: 0.7
      })
    });
    
    if (!response.ok) {
      throw new Error(`ChatGPT API failed: ${response.status}`);
    }
    
    const result = await response.json();
    return result.choices[0].message.content;
    
  } catch (error) {
    console.error('ChatGPT error:', error);
    return `${palmData.hand}の手相を拝見させていただきました✋\n\n` +
           `${palmData.palmReading}\n\n` +
           `信頼度: ${palmData.confidence}%\n\n` +
           `※より詳しい鑑定は後ほどお返しいたします。しばらくお待ちください。`;
  }
}

// LINE返信
async function replyToLine(replyToken, message) {
  try {
    const response = await fetch('https://api.line.me/v2/bot/message/reply', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.LINE_ACCESS_TOKEN}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        replyToken: replyToken,
        messages: [{
          type: 'text',
          text: message
        }]
      })
    });
    
    if (!response.ok) {
      throw new Error(`LINE reply failed: ${response.status}`);
    }
  } catch (error) {
    console.error('LINE reply error:', error);
  }
}
loadModel().then(() => {

  app.listen(PORT, () => console.log(`Server: ${PORT}`));
});
