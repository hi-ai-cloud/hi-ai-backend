import express from "express";
import cors from "cors";
import fetch from "node-fetch";
import "dotenv/config";

const app = express();
app.use(cors());
app.use(express.json({ limit: "20mb" }));

// Проверка, что сервер жив
app.get("/health", (req, res) => res.json({ ok: true }));

// Пример маршрута для OpenAI
app.post("/api/openai", async (req, res) => {
  try {
    const { messages, model = "gpt-4o-mini" } = req.body || {};
    const r = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        "content-type": "application/json"
      },
      body: JSON.stringify({
        model,
        messages,
        temperature: 0.7
      })
    });
    const data = await r.json();
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Пример маршрута для Replicate
app.post("/api/replicate", async (req, res) => {
  try {
    const { version, input } = req.body || {};
    const r = await fetch("https://api.replicate.com/v1/predictions", {
      method: "POST",
      headers: {
        authorization: `Token ${process.env.REPLICATE_API_TOKEN}`,
        "content-type": "application/json"
      },
      body: JSON.stringify({ version, input })
    });
    const data = await r.json();
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

const port = process.env.PORT || 8080;
app.listen(port, () => console.log(`HI-AI backend on :${port}`));
