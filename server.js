// server.js
// --- Palm AI API (LINE Bot + TFJS + OpenAI) ---
// Node v20 で動作。global fetch 利用。

const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const multer = require('multer');
const sharp = require('sharp');
const cors = require('cors');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// ---- Env vars (前後空白を除去し、欠落時は警告) ----
const LINE_TOKEN = (process.env.LINE_CHANNEL_ACCESS_TOKEN || process.env.LINE_ACCESS_TOKEN || '').trim();
const OPENAI_KEY = (process.env.OPENAI_API_KEY || '').trim();
const OPENAI_MODEL = (process.env.OPENAI_MODEL || 'gpt-4').trim();

if (!LINE_TOKEN) console.error('[BOOT] ❌ LINE token missing: set LINE_CHANNEL_ACCESS_TOKEN');
if (!OPENAI_KEY) console.error('[BOOT] ❌ OPENAI_API_KEY is missing!');

const upload = multer({ limits: { fileSize: 10 * 1024 * 1024 } });

// ---- TFJS model ----
let model = null;
let loaded = false;

async function loadModel() {
  try {
    const modelPath = path.join(__dirname, 'model.json');
    if (!fs.existsSync(modelPath)) {
      throw new Error(`model.json not found at ${modelPath}（同階層に置いてください）`);
    }
    console.log('[MODEL] Loading…', modelPath);
    model = await tf.loadLayersModel(`file://${modelPath}`);
    loaded = true;
    console.log('[MODEL] ✅ Loaded');
  } catch (err) {
    loaded = false;
    console.error('[MODEL] ❌ Load error:', err);
  }
}

// 予測の前処理（RGBA を除去して 224x224 RGB, [0,1] 正規化）
async function imageToTensorRGB(buffer) {
  const raw = await sharp(buffer)
    .resize(224, 224, { fit: 'cover' })
    .removeAlpha()
    .raw()
    .toBuffer(); // Uint8

  // 224*224*3 バイト → Float32
  const float = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) float[i] = raw[i] / 255.0;
  return tf.tensor4d(float, [1, 224, 224, 3]);
}

function generateReading(hand) {
  const readings = {
    left:  '左手は「本質・先天運」。あなたの内面の強さと創造性が根っこにあります。',
    right: '右手は「未来・後天運」。行動力・実行力が伸びていくタイミングです。'
  };
  return readings[hand] || '手のひらから良いエネルギーを感じます。前向きに進めば運が味方します。';
}

// ---- Basic endpoints ----
app.get('/', (_req, res) => res.json({ status: 'OK', loaded }));
app.get('/health', (_req, res) => res.json({ status: loaded ? 'healthy' : 'loading' }));

// 画像単体API（開発用）
app.post('/analyze-palm', upload.single('image'), async (req, res) => {
  try {
    if (!loaded) return res.status(503).json({ error: 'Model loading' });
    if (!req.file) return res.status(400).json({ error: 'No image' });

    const tensor = await imageToTensorRGB(req.file.buffer);
    const pred = await model.predict(tensor).data();
    tensor.dispose();

    // pred[0]: 右手確率（例）という前提
    const hand = pred[0] > 0.5 ? 'right' : 'left';
    const handJa = hand === 'right' ? '右手' : '左手';
    const conf = Math.round(Math.max(pred[0], 1 - pred[0]) * 100);

    const reading = generateReading(hand);

    res.json({
      success: true,
      hand: handJa,
      handEn: hand,
      confidence: conf,
      palmReading: reading,
      lineMessage: `${handJa}を検出！確信度: ${conf}%\n\n${reading}`
    });
  } catch (err) {
    console.error('[ANALYZE] error:', err);
    res.status(500).json({ error: err.message });
  }
});

// ---- LINE Webhook ----
app.post('/test-webhook', (req, res) => res.json({ ok: true, data: req.body }));

app.post('/line-webhook', async (req, res) => {
  try {
    const events = req.body?.events || [];
    for (const event of events) {
      if (event.type !== 'message') continue;

      if (event.message.type === 'image') {
        await handlePalmReading(event);
      } else if (event.message.type === 'text') {
        await handleTextMessage(event);
      }
    }
    // LINE には 200 を即返す
    res.sendStatus(200);
  } catch (err) {
    console.error('[WEBHOOK] error:', err);
    res.sendStatus(500);
  }
});

// ---- Handlers ----
async function handlePalmReading(event) {
  try {
    if (!LINE_TOKEN) throw new Error('LINE token missing');

    // 1) 画像取得
    const imgRes = await fetch(
      `https://api-data.line.me/v2/bot/message/${event.message.id}/content`,
      { headers: { Authorization: `Bearer ${LINE_TOKEN}` } }
    );
    if (!imgRes.ok) {
      const body = await imgRes.text();
      throw new Error(`Get image failed: ${imgRes.status} ${body}`);
    }
    const imgBuf = Buffer.from(await imgRes.arrayBuffer());

    // 2) 推論
    if (!loaded) throw new Error('Model not loaded');
    const tensor = await imageToTensorRGB(imgBuf);
    const pred = await model.predict(tensor).data();
    tensor.dispose();

    const hand = pred[0] > 0.5 ? 'right' : 'left';
    const handJa = hand === 'right' ? '右手' : '左手';
    const conf = Math.round(Math.max(pred[0], 1 - pred[0]) * 100);

    const palmData = {
      hand: handJa,
      handEn: hand,
      confidence: conf,
      palmReading: generateReading(hand)
    };

    // 3) 鑑定テキスト
    const fortune = await getChatGPTFortune(palmData);

    // 4) 返信
    await replyToLine(event.replyToken, fortune);
  } catch (err) {
    console.error('[PALM] error:', err);
    await replyToLine(
      event.replyToken,
      '申し訳ございません。手相の解析中にエラーが発生しました🙏\n\n' +
        '📸 撮影のコツ\n' +
        '・明るい場所で（自然光がベスト）\n' +
        '・手のひら全体が入るように\n' +
        '・ピンぼけに注意\n\n' +
        'もう一度お試しください。'
    );
  }
}

async function handleTextMessage(event) {
  const text = (event.message.text || '').toLowerCase();
  let reply = '';

  if (/(こんにち|はじめ|hello)/.test(text)) {
    reply =
      'こんにちは！手相鑑定botです✋\n\n' +
      '手のひらの写真を送っていただければ、AI分析＋鑑定結果をお返しします。\n\n' +
      '📸 コツ：明るい場所／手のひら全体／ピンぼけ注意';
  } else if (/(使い方|ヘルプ|help)/.test(text)) {
    reply =
      '📋 使い方\n' +
      '1) 手のひらを撮影\n2) このトークに写真を送信\n3) AIの鑑定が返ってきます\n\n' +
      'より正確に：自然光・手を広げる・指先〜手首まで写す📷';
  } else {
    reply =
      'ご利用ありがとうございます！✨\n' +
      '手相鑑定をご希望なら、手のひらの写真を送ってくださいね📸\n' +
      '困ったら「ヘルプ」と送ってください。';
  }
  await replyToLine(event.replyToken, reply);
}

// ---- OpenAI ----
async function getChatGPTFortune(palmData) {
  try {
    if (!OPENAI_KEY) throw new Error('OpenAI key missing');

    const prompt = `
手相鑑定をお願いします。

【解析データ】
手: ${palmData.hand}
信頼度: ${palmData.confidence}%
基本鑑定: ${palmData.pal
