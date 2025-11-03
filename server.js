import express from "express";
import cors from "cors";
import fetch from "node-fetch";
import "dotenv/config";

const app = express();

// --- при желании сузим домены позже ---
app.use(cors());
app.use(express.json({ limit: "25mb" }));

app.get("/health", (_, res) => res.json({ ok: true, ts: Date.now() }));

// ---------- Модели из ENV (как у тебя в Pipedream) ----------
const MODELS = {
  realistic:  process.env.REPLICATE_MODEL_SLUG_REALISTIC,
  cartoon:    process.env.REPLICATE_MODEL_SLUG_CARTOON,
  futuristic: process.env.REPLICATE_MODEL_SLUG_FUTURISTIC,
  i2v_hd:     process.env.REPLICATE_MODEL_SLUG_I2V_HD,
  sdxl:       process.env.REPLICATE_MODEL_VERSION_SDXL,
  flux:       process.env.REPLICATE_MODEL_VERSION_FLUX,
  video:      process.env.REPLICATE_MODEL_VERSION_VIDEO,
  default:    process.env.REPLICATE_MODEL_VERSION
};

const REPLICATE_HEADERS = {
  authorization: `Token ${process.env.REPLICATE_API_TOKEN}`,
  "content-type": "application/json"
};

// ---------- helper: опрос Replicate до готового результата ----------
async function pollPrediction(predictionUrl, timeoutMs = 300000, stepMs = 2500) {
  const end = Date.now() + timeoutMs;
  while (Date.now() < end) {
    const r = await fetch(predictionUrl, { headers: REPLICATE_HEADERS });
    const data = await r.json();
    if (data.status === "succeeded" || data.status === "failed" || data.status === "canceled") {
      return data;
    }
    await new Promise(r => setTimeout(r, stepMs));
  }
  throw new Error("Prediction polling timeout");
}

// ================== OpenAI: ТЕКСТ ==================
app.post("/api/text", async (req, res) => {
  try {
    const { messages, model = "gpt-4o-mini", temperature = 0.7 } = req.body || {};
    const r = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        "content-type": "application/json"
      },
      body: JSON.stringify({ model, messages, temperature })
    });
    const data = await r.json();
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ================== Replicate: КАРТИНКИ ==================
app.post("/api/image", async (req, res) => {
  try {
    const { prompt, negative_prompt, style = "realistic", input = {}, wait = true } = req.body || {};
    const version = MODELS[style] || MODELS.default;
    if (!version) throw new Error("Model version not set in ENV");

    // создаём prediction
    const create = await fetch("https://api.replicate.com/v1/predictions", {
      method: "POST",
      headers: REPLICATE_HEADERS,
      body: JSON.stringify({
        version,
        input: { prompt, negative_prompt, ...input }
      })
    });
    const created = await create.json();

    // сразу вернуть ручку на поллинг, если wait=false
    if (!wait) return res.json(created);

    // иначе — дождаться результата и вернуть конечный output (URL/ы)
    const done = await pollPrediction(created.urls.get);
    res.json(done);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ================== Replicate: ВИДЕО / I2V ==================
app.post("/api/video", async (req, res) => {
  try {
    const { image_url, prompt, mode = "i2v_hd", input = {}, wait = true } = req.body || {};
    const version = MODELS[mode] || MODELS.video || MODELS.default;
    if (!version) throw new Error("Video model version not set in ENV");

    const create = await fetch("https://api.replicate.com/v1/predictions", {
      method: "POST",
      headers: REPLICATE_HEADERS,
      body: JSON.stringify({
        version,
        input: { image: image_url, prompt, ...input }
      })
    });
    const created = await create.json();

    if (!wait) return res.json(created);

    const done = await pollPrediction(created.urls.get, 600000, 3000); // до 10 мин
    res.json(done);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

const port = process.env.PORT || 8080;
app.listen(port, () => console.log(`HI-AI backend on :${port}`));
