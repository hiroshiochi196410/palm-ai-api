// server.js
// Palm AI API (LINE Bot + TensorFlow.js + OpenAI)
// Node v20 以上。global fetch を使用。

const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const multer = require('multer');
const sharp = require('sharp');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
+// === Image preprocess: 入力を [1,224,224,3] に統一（expandDims は1回だけ）===
+function preprocess(buffer) {
+  return tf.tidy(() => {
+    let img = tf.node.decodeImage(buffer, 3);      // [H, W, 3]
+    img = tf.image.resizeBilinear(img, [224, 224]); // [224, 224, 3]
+    img = img.toFloat().sub(127.5).div(127.5);
                   // 0-1 正規化（必要に応じて変更）
+    return img.expandDims(0);                       // [1, 224, 224, 3]
+  });
+}
const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// ===== Env =====
const LINE_TOKEN = (process.env.LINE_CHANNEL_ACCESS_TOKEN || process.env.LINE_ACCESS_TOKEN || '').trim();
const OPENAI_KEY = (process.env.OPENAI_API_KEY || '').trim();
const OPENAI_MODEL = (process.env.OPENAI_MODEL || 'gpt-4o').trim();

if (!LINE_TOKEN) console.error('[BOOT] ❌ LINE token missing (set LINE_CHANNEL_ACCESS_TOKEN)');
if (!OPENAI_KEY) console.error('[BOOT] ❌ OPENAI_API_KEY missing');

const upload = multer({ limits: { fileSize: 10 * 1024 * 1024 } });

// ===== TFJS model =====
let model = null;
let loaded = false;

async function loadModel() {
  try {
    const modelPath = path.join(__dirname, 'models', 'model.json');
    if (!fs.existsSync(modelPath)) {
      throw new Error(`model.json not found: ${modelPath}（models/ に配置してください）`);
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

// RGBA→RGB / 224x224 / [0,1] 正規化
async function imageToTensorRGB(buffer) {
  const raw = await sharp(buffer)
    .resize(224, 224, { fit: 'cover' })
    .removeAlpha()
    .raw()
    .toBuffer(); // Uint8Array

  const float = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) float[i] = raw[i] / 255.0;
  return tf.tensor4d(float, [1, 224, 224, 3]);
}

function generateReading(hand) {
  const readings = {
    left:  '左手は「本質・先天運」。内面の強さと創造性が根っこにあります。',
    right: '右手は「未来・後天運」。行動力・実行力が伸びるタイミングです。'
  };
  return readings[hand] || '手のひらから良いエネルギー。前向きに進めば運が味方します。';
}

// ===== Basic endpoints =====
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

    const hand = pred[0] > 0.5 ? 'right' : 'left';     // 右確率を pred[0] と仮定
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

// ===== LINE Webhook =====
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
    res.sendStatus(200); // すぐ 200 を返す
  } catch (err) {
    console.error('[WEBHOOK] error:', err);
    res.sendStatus(500);
  }
});

// 画像メッセージ処理
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

    // 3) 鑑定
    const fortune = await getChatGPTFortune(palmData);

    // 4) 返信
    await replyToLine(event.replyToken, fortune);
  } catch (err) {
    console.error('[PALM] error:', err);
    await replyToLine(
      event.replyToken,
      '申し訳ございません。手相の解析中にエラーが発生しました🙏\n\n' +
        '📸 撮影のコツ\n' +
        '・明るい場所（自然光がベスト）\n' +
        '・手のひら全体が入るように\n' +
        '・ピンぼけに注意\n\n' +
        'もう一度お試しください。'
    );
  }
}

// テキストメッセージ処理
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

// OpenAI
async function getChatGPTFortune(palmData) {
  try {
    if (!OPENAI_KEY) throw new Error('OpenAI key missing');

    const prompt = `
手相鑑定をお願いします。

【解析データ】
手: ${palmData.hand}
信頼度: ${palmData.confidence}%
基本鑑定: ${palmData.palmReading}

【鑑定スタイル】
- 温かく親しみやすい関西弁まじり
- 良い面→建設的アドバイス
- 具体的・実践的
- 前向きな結論で締める

【含める】
🌟 手相の特徴
💪 生命線（健康・生命力）
🧠 知能線（才能・適職）
❤️ 感情線（恋愛・人間関係）
🍀 総合運とアドバイス

【文字数】400-500文字
`;

    const r = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${OPENAI_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: OPENAI_MODEL,
        messages: [
          { role: 'system', content: '親しみやすいベテラン手相鑑定師として丁寧に鑑定する。' },
          { role: 'user', content: prompt }
        ],
        max_tokens: 800,
        temperature: 0.7
      })
    });

    if (!r.ok) {
      const body = await r.text();
      throw new Error(`OpenAI ${r.status} ${body}`);
    }
    const json = await r.json();
    return json.choices?.[0]?.message?.content?.trim() || '鑑定結果を生成しました。';
  } catch (err) {
    console.error('[OPENAI] error:', err);
    return (
      `${palmData.hand}の手相を拝見しました✋\n\n` +
      `${palmData.palmReading}\n\n` +
      `信頼度: ${palmData.confidence}%\n\n` +
      `※詳細な鑑定は後ほどお届けします。`
    );
  }
}

// LINE返信
async function replyToLine(replyToken, message) {
  try {
    if (!LINE_TOKEN) throw new Error('LINE token missing');

    const r = await fetch('https://api.line.me/v2/bot/message/reply', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${LINE_TOKEN}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        replyToken,
        messages: [{ type: 'text', text: message }]
      })
    });
    if (!r.ok) {
      const body = await r.text();
      throw new Error(`LINE reply failed: ${r.status} ${body}`);
    }
  } catch (err) {
    console.error('[LINE] reply error:', err);
  }
}

// Boot
loadModel().finally(() => {
  app.listen(PORT, () => console.log(`[BOOT] Server listening on ${PORT}`));
});
