// HI-AI HUB — Unified Backend (Brand Post + Image Studio + Video Studio + Video Reels)
// Express + OpenAI + Replicate (polling, data:URL safe, model routing via ENV)

import express from "express";
import cors from "cors";
import fetch from "node-fetch";
import OpenAI from "openai";
import "dotenv/config";

const app = express();
app.use(cors()); // при необходимости потом сузим по домену
app.use(express.json({ limit: "30mb" }));

/* ====================== UTILITIES ====================== */
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function fetchJson(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let t = "";
    try { t = await res.text(); } catch {}
    const err = new Error(`HTTP ${res.status} ${res.statusText} :: ${t}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}
function readBody(raw) {
  if (!raw) return {};
  if (typeof raw === "object") return raw;
  if (typeof raw === "string") { try { return JSON.parse(raw); } catch { return {}; } }
  return {};
}
function okUrl(v){ return typeof v === "string" && /^https?:\/\//i.test(v); }
function pickUrl(output){
  if (!output) return null;
  if (okUrl(output)) return output;
  if (Array.isArray(output)) {
    const f = output[0];
    if (okUrl(f)) return f;
    if (f && typeof f === "object") for (const v of Object.values(f)) if (okUrl(v)) return v;
  }
  if (typeof output === "object") for (const v of Object.values(output)) if (okUrl(v)) return v;
  return null;
}
async function urlToDataURL(url){
  const r = await fetch(url);
  if (!r.ok) {
    let msg = ""; try { msg = await r.text(); } catch {};
    throw new Error(`fetch failed ${r.status}${msg ? `: ${msg}` : ""}`);
  }
  const ct = r.headers.get("content-type") || "image/jpeg";
  const ab = await r.arrayBuffer();
  const b64 = Buffer.from(ab).toString("base64");
  return `data:${ct};base64,${b64}`;
}
function makeDataUrlSafe(dataUrl){
  const s = String(dataUrl || "").trim();
  if (!s.startsWith("data:")) return null;
  if (!/data:[^;]+;base64,/.test(s)) {
    return `data:image/png;base64,${s.replace(/^data:[^,]+,/, "")}`;
  }
  return s;
}

/* =============== REPLICATE HELPERS (polling) =============== */
const REPLICATE_HEADERS = {
  Authorization: `Token ${process.env.REPLICATE_API_TOKEN}`,
  "Content-Type": "application/json"
};

async function replicateCreate(version, input) {
  if (!process.env.REPLICATE_API_TOKEN) throw new Error("Missing REPLICATE_API_TOKEN");
  if (!version) throw new Error("Missing Replicate model version");
  return fetchJson("https://api.replicate.com/v1/predictions", {
    method: "POST",
    headers: REPLICATE_HEADERS,
    body: JSON.stringify({ version, input })
  });
}

async function pollPredictionByUrl(getUrl, { tries = 240, delayMs = 1500 } = {}) {
  let last = null;
  for (let i = 0; i < tries; i++) {
    last = await fetchJson(getUrl, { headers: { Authorization: `Token ${process.env.REPLICATE_API_TOKEN}` } });
    if (last.status === "succeeded") return last;
    if (last.status === "failed" || last.status === "canceled") {
      throw new Error(`Replicate failed: ${last?.error || last.status}`);
    }
    await sleep(delayMs);
  }
  throw new Error("Replicate timeout");
}

async function replicatePredict(version, input, pollCfg) {
  const job = await replicateCreate(version, input);
  const done = await pollPredictionByUrl(job?.urls?.get, pollCfg);
  return done;
}

/* ====================== MODEL MAPS ====================== */
const MODELS = {
  // из Pipedream ENV
  realistic:  process.env.REPLICATE_MODEL_SLUG_REALISTIC,    // slug or version (we use version for /image)
  cartoon:    process.env.REPLICATE_MODEL_SLUG_CARTOON,
  futuristic: process.env.REPLICATE_MODEL_SLUG_FUTURISTIC,
  i2v_hd:     process.env.REPLICATE_MODEL_SLUG_I2V_HD,

  sdxl:       process.env.REPLICATE_MODEL_VERSION_SDXL,      // version id
  flux:       process.env.REPLICATE_MODEL_VERSION_FLUX,
  video:      process.env.REPLICATE_MODEL_VERSION_VIDEO,
  i2v:        process.env.REPLICATE_MODEL_VERSION_I2V,       // optional direct
  def:        process.env.REPLICATE_MODEL_VERSION
};

function chooseImageModelKey({ idea, style, hint }) {
  const h = String(hint || "auto").toLowerCase();
  if (h === "sdxl" || h === "flux") return h;
  const st = String(style || "auto").toLowerCase();
  if (st === "cartoon3d" || st === "illustrated") return "flux";
  if (st === "futuristic" || st === "realistic")  return "sdxl";
  const isFun = /halloween|kids|pizza|party|fun|gymnastics/i.test(idea || "");
  return isFun ? "flux" : "sdxl";
}

/* ====================== HEALTH ====================== */
app.get("/health", (_, res) => res.json({ ok: true, ts: Date.now() }));

/* ====================== BRAND POST ====================== */
// POST /api/brand-post
app.post("/api/brand-post", async (req, res) => {
  try {
    const body = readBody(req.body);
    const idea = (body.idea || body.prompt || "").toString().trim();
    if (!idea) return res.status(400).json({ ok: false, error: "Missing 'idea' (or 'prompt')" });

    const style = (body.style || "auto").toString().toLowerCase();
    const ratio = (body.ratio || "1:1").toString().replace("-", ":");
    const options = typeof body.options === "object" ? body.options : { image: true };
    const category = (body.category || "General").toString();
    const subcategory = (body.subcategory || "").toString();
    const length = (body.length || "medium").toString().toLowerCase(); // "short" | "medium" | "long"
    const imageOnly = !!body.image_only;
    const textOnly  = !!body.text_only;
    const imageModelHint = (body.image_model_hint || "auto").toString().toLowerCase();

    const [w, h] = ratio === "9:16" ? [896, 1600] : ratio === "16:9" ? [1280, 720] : [1024, 1024];

    const modelKey = (()=> {
      if (imageModelHint === "sdxl" || imageModelHint === "flux") return imageModelHint;
      if (style === "cartoon3d" || style === "illustrated") return "flux";
      if (style === "futuristic" || style === "realistic")  return "sdxl";
      const isFun = /halloween|kids|pizza|party|fun|gymnastics/i.test(idea || "");
      return isFun ? "flux" : "sdxl";
    })();

    const lengthTargets = { short: 120, medium: 220, long: 400 };
    const maxChars = lengthTargets[length] || 220;
    const wantsEmoji = !!options.emojis;
    const wantsHash  = !!options.auto_hashtags;

    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

    let caption = null, vprompt = null, gptUsed = false;

    if (!imageOnly) {
      try {
        const user = `
Write a short social media caption and a clean visual prompt for an image generator.

Constraints:
- Tone preset: ${body.preset || "neutral"}
- CTA text: ${body.cta || "Learn more"}
- Category: ${category}${subcategory ? " / " + subcategory : ""}
- Emojis: ${wantsEmoji ? "ON (use 1–3 emojis total)" : "OFF (no emojis)"}
- Hashtags: ${wantsHash ? "ON (2–4 relevant at end)" : "OFF (no hashtags)"}
- Do NOT include any URLs in the caption.
- Aim for ~${maxChars} characters (hard cap: ${Math.round(maxChars * 1.1)}).
- Visual prompt must forbid text/letters/logos in image and keep text-safe area.
- Aspect ratio primary: ${ratio}
- Style hint: ${style}

Event idea: "${idea}"

Return STRICT JSON:
{
  "caption": "one paragraph under ${Math.round(maxChars * 1.1)} chars, CTA included, obey emoji/hashtag flags, no URLs",
  "visual_prompt": "clean prompt for image generator without text in image"
}`.trim();

        const gptResp = await openai.chat.completions.create({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: user }],
          temperature: 0.9,
          max_tokens: 500,
        });
        const text = gptResp?.choices?.[0]?.message?.content || "";
        const s = text.indexOf("{"), e = text.lastIndexOf("}");
        const parsed = JSON.parse(s >= 0 && e >= 0 ? text.slice(s, e + 1) : "{}");

        caption = (parsed.caption || "").toString().trim();
        vprompt = (parsed.visual_prompt || "").toString().trim();

        if (!caption || !vprompt) throw new Error("Empty fields in GPT JSON");
        if (caption.length > Math.round(maxChars * 1.1)) {
          const cut = caption.slice(0, Math.round(maxChars * 1.1));
          const idx = Math.max(cut.lastIndexOf(". "), cut.lastIndexOf(" "), Math.floor(cut.length * 0.9));
          caption = cut.slice(0, idx).trim() + "…";
        }

        gptUsed = true;
      } catch (e) {
        caption = `✨ ${idea}\n🚀 Learn more and take action today.\n\n➡️ Learn more\n\nhttps://hi-ai.ai #ai #automation #creativity`.slice(0, maxChars);
        vprompt = `${idea}. Modern minimalist beige & orange, warm light, clean bg, no text. AR ${ratio}.`;
      }
    } else {
      // image_only
      vprompt = `${idea}. ${
        (style === "cartoon3d" || style === "illustrated")
          ? "3D toon-shaded / flat illustrated, rounded forms, cel shading edges."
          : style === "futuristic"
          ? "Futuristic neon, glassmorphism, volumetric lights."
          : style === "realistic"
          ? "Photorealistic warm golden light, shallow DOF."
          : "Let AI choose best style; clean composition."
      } No text/logos on image. Aspect ratio ${ratio}. High detail.`;
    }

    // Replicate image (unless text_only)
    let image_url = null;
    if (options.image !== false && !textOnly) {
      try {
        const negative = "letters, text, words, watermark, logo, blurry, noisy, cluttered";
        if (modelKey === "flux") {
          const job = await replicatePredict(process.env.REPLICATE_MODEL_VERSION_FLUX, {
            prompt: vprompt, go_fast: false, megapixels: "1", prompt_strength: 0.85,
            num_outputs: 1, output_format: "png", output_quality: 90
          });
          image_url = Array.isArray(job.output) ? job.output[0] : job.output;
        } else {
          const job = await replicatePredict(process.env.REPLICATE_MODEL_VERSION_SDXL, {
            prompt: vprompt, negative_prompt: negative, width: w, height: h,
            num_inference_steps: 30, guidance_scale: 7.0, scheduler: "DPMSolverMultistep", num_outputs: 1
          });
          image_url = Array.isArray(job.output) ? job.output[0] : job.output;
        }
      } catch (e) { /* вернём что есть */ }
    }

    return res.json({
      ok: true, caption: textOnly || !caption ? caption || null : caption,
      vprompt, image_url, model_used: (modelKey || "default").toUpperCase(),
      gpt_used: !!gptUsed, length, mode: imageOnly ? "image_only" : textOnly ? "text_only" : "full"
    });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e.message || e) });
  }
});

/* ====================== IMAGE STUDIO ====================== */
// POST /api/image-studio
app.post("/api/image-studio", async (req, res) => {
  try {
    // Параметры и константы
    const WAIT_POLL_MS = 1200;
    const MAX_WAIT_MS  = 120000;
    const DEFAULT_AR   = "1:1";
    const DEFAULT_STRENGTH = 0.6;

    const REMOVE_BG_MODELS = [
      { slug: "recraft-ai/recraft-remove-background", makeInput: ({ image_data }) => ({ image: image_data }) },
      { slug: "851-labs/background-remover",         makeInput: ({ image_data }) => ({ image: image_data }) },
      { slug: "lucataco/remove-bg",                  makeInput: ({ image_data }) => ({ image: image_data }) },
    ];
    const UPSCALE_MODELS = [
      { slug: "stability-ai/stable-diffusion-x4-upscaler", makeInput: ({ image_data, prompt }) => ({ image: image_data, prompt: prompt || "" }) },
      { slug: "stability-ai/sd-x4-upscaler",               makeInput: ({ image_data, prompt }) => ({ image: image_data, prompt: prompt || "" }) },
      { slug: "lucataco/stable-diffusion-x4-upscaler",     makeInput: ({ image_data, prompt }) => ({ image: image_data, prompt: prompt || "" }) },
      { slug: "xinntao/real-esrgan",                       makeInput: ({ image_data }) => ({ image: image_data }) },
      { slug: "nightmareai/real-esrgan",                   makeInput: ({ image_data }) => ({ image: image_data }) },
    ];
    const ANGLE_MAP = {
      front: "front view",
      "34left": "3/4 angle, left side",
      "34right": "3/4 angle, right side",
      side: "side view",
      back: "back view",
      top: "top-down view",
      low: "low-angle cinematic shot",
    };
    const angleOrder = () => ["front","34left","side","34right","back","34right","side","34left"];
    const orbitAngles = (count)=> {
      const order = angleOrder();
      return Array.from({length:count}, (_,i)=>order[i % order.length]).map(k=>ANGLE_MAP[k]||k);
    };

    const body = readBody(req.body);
    const actionRaw  = body.action || (body.image || body.image_data ? "img2img" : "text2img");
    const action     = String(actionRaw).toLowerCase();
    const promptRaw  = (body.prompt || "").trim();
    const aspect     = body.aspect_ratio || DEFAULT_AR;
    const strength   = body.strength ?? DEFAULT_STRENGTH;
    const seed       = body.seed ?? null;
    const seed_lock  = !!body.seed_lock;
    const camera_path= String(body.camera_path || "none").toLowerCase(); // 'none'|'orbit'

    let image_data = body.image_data || null;
    let mask_data  = body.mask_data  || null;
    const imageUrl = (body.image || "").trim() || null;
    const maskUrl  = (body.mask  || "").trim() || null;

    const batch_count = Math.max(1, Math.min(8, Number(body.batch_count || 1)));
    let anglesRaw = Array.isArray(body.angles) ? body.angles : [];
    const angles = anglesRaw.map(x=>{
      const key = String(x || "").toLowerCase().trim();
      return ANGLE_MAP[key] ? ANGLE_MAP[key] : key;
    });

    if (!process.env.REPLICATE_API_TOKEN) {
      return res.status(500).json({ ok:false, success:false, error:"Missing REPLICATE_API_TOKEN" });
    }

    try {
      if (!image_data && okUrl(imageUrl)) image_data = await urlToDataURL(imageUrl);
      if (!mask_data  && okUrl(maskUrl))  mask_data  = await urlToDataURL(maskUrl);
    } catch (e) {
      return res.status(400).json({ ok:false, success:false, error:`fetch(url) failed: ${String(e)}` });
    }

    async function waitPrediction(id, model, maxMs = MAX_WAIT_MS) {
      const started = Date.now();
      while (true) {
        if (Date.now() - started > maxMs) throw new Error(`timeout (${model})`);
        const p = await fetchJson(`https://api.replicate.com/v1/predictions/${id}`, {
          headers: { Authorization: `Token ${process.env.REPLICATE_API_TOKEN}` }
        });
        if (p.status === "succeeded") {
          const image_url = pickUrl(p.output);
          if (!image_url) throw new Error(`empty output (${model})`);
          return image_url;
        }
        if (p.status === "failed" || p.status === "canceled") throw new Error(`prediction ${p.status} (${model}) ${p.error ? `: ${p.error}` : ""}`);
        await sleep(WAIT_POLL_MS);
      }
    }
    const jitterStrength = (base)=> {
      const b = isFinite(base) ? Number(base) : DEFAULT_STRENGTH;
      const j = (Math.random()*0.12 - 0.06);
      return Math.min(0.9, Math.max(0.3, b + j));
    };

    async function runSingle({ prompt, image_data, mask_data, strength, seed }) {
      let model = "", input = {};
      if (action === "text2img") {
        if (!prompt) throw new Error("prompt is required for text2img");
        model = "black-forest-labs/flux-schnell";
        input = { prompt, aspect_ratio: aspect, ...(seed!=null? { seed } : {}) };
      } else if (action === "img2img") {
        if (!image_data) throw new Error("image is required for img2img");
        model = "black-forest-labs/flux-kontext-pro";
        input = { prompt, input_image: image_data, image: image_data, strength, output_format: "jpg", ...(seed!=null? { seed } : {}) };
      } else if (action === "inpaint") {
        if (!image_data) throw new Error("image is required for inpaint");
        model = "black-forest-labs/flux-kontext-pro";
        input = { prompt, input_image: image_data, image: image_data, ...(mask_data ? { mask_image: mask_data, mask: mask_data } : {}), strength, output_format: "jpg", ...(seed!=null? { seed } : {}) };
      } else if (action === "remove_bg" || action === "upscale") {
        const MODELS = action === "remove_bg" ? REMOVE_BG_MODELS : UPSCALE_MODELS;
        if (!image_data) throw new Error(`image is required for ${action}`);
        const tried = [];
        for (const m of MODELS) {
          try {
            const inputX = m.makeInput({ image_data, prompt: prompt || "" });
            const pred = await fetchJson("https://api.replicate.com/v1/predictions", {
              method: "POST", headers: REPLICATE_HEADERS, body: JSON.stringify({ model: m.slug, input: inputX })
            });
            const image_url = await waitPrediction(pred.id, m.slug, body.max_wait_ms);
            return { image_url, model: m.slug, tried };
          } catch (e) { tried.push({ model: m.slug, error: String(e) }); }
        }
        if (action === "upscale") return { error: `${action}: no available models`, status: 502 };
        throw new Error(`${action}: all models failed`);
      } else {
        throw new Error(`unknown action: ${action}`);
      }

      const pred = await fetchJson("https://api.replicate.com/v1/predictions", {
        method: "POST", headers: REPLICATE_HEADERS, body: JSON.stringify({ model, input })
      });
      const url = await waitPrediction(pred.id, model, body.max_wait_ms);
      return { image_url: url, model };
    }

    // Single or Batch
    if (batch_count === 1) {
      const anglePhrase = (camera_path === "orbit" && batch_count > 1) ? orbitAngles(batch_count)[0] : (angles[0] || "");
      const prompt = anglePhrase ? `${promptRaw}, ${anglePhrase}` : promptRaw;
      const out = await runSingle({ prompt, image_data, mask_data, strength, seed });
      if (out && out.image_url) return res.json({ ok:true, success:true, mode: action, model: out.model, image_url: out.image_url });
      if (out && out.error && out.status === 502) return res.status(502).json({ ok:false, success:false, error: out.error });
      throw new Error("unexpected empty output");
    }

    // Batch
    const count = Math.max(1, Math.min(8, batch_count));
    const baseSeed = (seed!=null ? Number(seed) : Math.floor(Math.random()*10_000_000));
    const seqAngles = (camera_path === "orbit" && batch_count > 1) ? orbitAngles(count) : angles;

    const tasks = [];
    for (let i=0;i<count;i++){
      const anglePhrase = seqAngles.length ? seqAngles[i % seqAngles.length] : "";
      const p = anglePhrase ? `${promptRaw}, ${anglePhrase}` : promptRaw;
      const s = (action==="img2img" || action==="inpaint") ? jitterStrength(strength) : strength;
      const useSeed = seed_lock ? baseSeed : (baseSeed + i);
      tasks.push({ prompt:p, image_data, mask_data, strength:s, seed: useSeed });
    }

    const out = [];
    for (const t of tasks){
      try{ const r = await runSingle(t); out.push({ ok:true, image_url: r.image_url }); }
      catch(e){ out.push({ ok:false, error:String(e) }); }
    }
    const firstOk = out.find(x=>x.ok && x.image_url);
    const result = { ok:true, success:true, mode: action, batch: out };
    if (firstOk) result.image_url = firstOk.image_url;
    return res.json(result);

  } catch (e) {
    res.status(502).json({ ok:false, success:false, error:String(e) });
  }
});

/* ====================== VIDEO STUDIO ====================== */
// POST /api/video-studio
app.post("/api/video-studio", async (req, res) => {
  try {
    const body = readBody(req.body);

    const mapRatioToWanSize = (ratio)=>{
      const r = String(ratio||"9:16").replace("-",":");
      if (r==="16:9") return "1920*1080";
      if (r==="1:1")  return "1080*1080";
      return "1080*1920";
    };
    const fitDurationToWan = (sec)=> (parseInt(sec||5,10) <= 5 ? 5 : 10);
    const ratioToWH = (ratio)=>{
      const r = String(ratio||"9:16").replace("-",":");
      if (r==="16:9") return [1920,1080];
      if (r==="1:1")  return [1080,1080];
      return [1080,1920];
    };
    const buildVPrompt = ({idea,style,ratio,seconds,category,subcategory})=>{
      const topic = [category||"General", subcategory||""].filter(Boolean).join(" • ");
      const hints = {
        auto:        "Let AI pick style. No on-frame text. Text-safe area.",
        realistic:   "Photo-realistic, cinematic warm light, shallow depth.",
        cartoon3d:   "3D toon shading, rounded forms, glossy surfaces.",
        illustrated: "Flat vector, bold simple shapes, warm palette.",
        futuristic:  "Futuristic neon/glass, volumetric light, bokeh.",
        canva:       "Minimal beige+orange (#E87D24,#F4E6CF), clean bg."
      };
      const s = hints[(style||"auto").toLowerCase()] || hints.auto;
      return [
        `Subject: ${topic||"General"}.`,
        idea || "Cinematic short video.",
        s,
        `Aspect ratio: ${String(ratio||"9:16").replace("-",":")}.`,
        `Target duration: ${parseInt(seconds||5,10)}s.`,
        `Avoid logos, watermarks, letters, subtitles.`
      ].join(" ");
    };

    const mode   = String(body.mode || "text2video").toLowerCase();
    const idea   = String(body.idea || body.prompt || "").trim();
    const style  = String(body.style || "auto").toLowerCase();
    const ratio  = String(body.ratio || "9:16").replace("-",":");
    const secs   = fitDurationToWan(body.video_seconds ?? body.duration_seconds ?? 5);
    const size   = mapRatioToWanSize(ratio);
    const [w,h]  = ratioToWH(ratio);

    const vprompt = buildVPrompt({
      idea, style, ratio, seconds: secs,
      category: body.category, subcategory: body.subcategory
    });

    if (mode !== "image2video" && !idea) {
      return res.json({ ok:false, error:"Missing 'idea' for text2video" });
    }
    if (mode === "image2video" && !body.image_url && !body.image_data_url) {
      return res.json({ ok:false, error:"Provide 'image_url' or 'image_data_url'" });
    }

    let video_url = null;
    let image_url = null;
    let modeUsed  = mode;

    // TEXT -> VIDEO
    if (mode === "text2video") {
      const verVideo = (process.env.REPLICATE_MODEL_VERSION_VIDEO || "").trim();
      if (!verVideo) return res.json({ ok:false, error:"Set REPLICATE_MODEL_VERSION_VIDEO" });

      const inputV = {
        prompt: vprompt, size, duration: secs,
        negative_prompt: "text, logo, watermark, letters, subtitles",
        enable_prompt_expansion: true
      };
      try {
        const job = await replicatePredict(verVideo, inputV);
        video_url = typeof job.output === "string" ? job.output : (Array.isArray(job.output)? job.output[0] : null);
      } catch (e) { /* fallback ниже */ }

      if (!video_url && (process.env.REPLICATE_MODEL_VERSION_SDXL || process.env.REPLICATE_MODEL_VERSION_FLUX)) {
        const useFlux = style==="cartoon3d" || style==="illustrated";
        const verStill = useFlux ? process.env.REPLICATE_MODEL_VERSION_FLUX : process.env.REPLICATE_MODEL_VERSION_SDXL;
        if (verStill) {
          const inputI = useFlux
            ? { prompt:vprompt, go_fast:false, megapixels:"1", num_outputs:1, output_format:"png", output_quality:90 }
            : { prompt:vprompt, width:w, height:h, num_inference_steps:30, guidance_scale:7.0, num_outputs:1 };
          try {
            const jobI = await replicatePredict(verStill.trim(), inputI);
            image_url = Array.isArray(jobI.output) ? jobI.output[0] : jobI.output;
            modeUsed = "image_fallback";
          } catch (e) {}
        }
      }
    }

    // IMAGE -> VIDEO (start_image data:URL safe)
    if (mode === "image2video") {
      let versionI2V = (process.env.REPLICATE_MODEL_VERSION_I2V || "").trim();
      const fallbackSlug = (process.env.REPLICATE_MODEL_SLUG_I2V_HD || "").trim();

      // если светится только SLUG — можно просто передать slug как model в predictions.create
      // но мы используем version, поэтому оставим env REPLICATE_MODEL_VERSION_I2V по возможности
      if (!versionI2V && !fallbackSlug) {
        return res.json({ ok:false, error:"No i2v model (set REPLICATE_MODEL_VERSION_I2V or SLUG_I2V_HD)" });
      }

      let startImage = (body.image_url || "").trim();
      const dataUrlRaw = (body.image_data_url || "").trim();
      if (dataUrlRaw) {
        startImage = makeDataUrlSafe(dataUrlRaw);
        if (!startImage) return res.json({ ok:false, error:"Bad data URL" });
      } else if (!/^https?:\/\//i.test(startImage)) {
        return res.json({ ok:false, error:"image_url must be https" });
      }

      const motion = Math.max(0, Math.min(1, parseFloat(body.motion_strength) || 0.35));
      const cam = ({"push-in":"push-in","pan-left":"pan-left","pan-right":"pan-right","tilt-up":"tilt-up","tilt-down":"tilt-down"})[String(body.camera || "auto").toLowerCase()] || "auto";

      const inputI2V = {
        prompt: vprompt,
        start_image: startImage,
        camera_motion_strength: motion,
        camera_motion: cam,
        size, duration: secs,
        negative_prompt: "text, logo, watermark, letters, subtitles"
      };

      try {
        // если есть version — используем его; если нет — попробуем через slug напрямую
        if (versionI2V) {
          const job = await replicatePredict(versionI2V, inputI2V, { tries: 240, delayMs: 1500 });
          video_url = typeof job.output === "string" ? job.output : (Array.isArray(job.output)? job.output[0] : null);
        } else {
          // прямой вызов через model: slug
          const job = await fetchJson("https://api.replicate.com/v1/predictions", {
            method: "POST", headers: REPLICATE_HEADERS,
            body: JSON.stringify({ model: fallbackSlug, input: inputI2V })
          });
          const done = await pollPredictionByUrl(job?.urls?.get, { tries: 240, delayMs: 1500 });
          video_url = typeof done.output === "string" ? done.output : (Array.isArray(done.output)? done.output[0] : null);
        }
      } catch (e) {
        return res.json({ ok:false, error: `Replicate i2v error: ${e.message}` });
      }
    }

    return res.json({
      ok: true, vprompt, video_url: video_url || null, image_url: image_url || null,
      ratio, seconds: secs, size_used: size, mode: video_url ? modeUsed : (image_url ? "image_fallback" : mode)
    });

  } catch (e) {
    res.json({ ok:false, error:String(e.message||e) });
  }
});

/* ====================== VIDEO REELS (caption + video) ====================== */
// POST /api/video-reels
app.post("/api/video-reels", async (req, res) => {
  try {
    const body = readBody(req.body);

    const idea = (body.idea || body.prompt || "").toString().trim();
    if (!idea) return res.status(400).json({ ok:false, error:"Missing 'idea'" });

    const text_only  = !!body.text_only || !!body.force_text_only;
    const image_only = !!body.image_only;
    const full_mode  = !text_only && !image_only;

    const style  = (body.style || "auto").toString().toLowerCase();
    const ratio  = (body.ratio || "9:16").toString().replace("-", ":");
    const length = (body.length || "medium").toString().toLowerCase();
    const opts   = typeof body.options === "object" ? body.options : { image: true };
    const category = (body.category || "General").toString();
    const subcategory = (body.subcategory || "").toString();
    const preset = (body.preset || "neutral").toString();
    const cta    = (body.cta || "Learn more").toString();
    const imageHint = (body.image_model_hint || "auto").toString().toLowerCase();

    const requestedSeconds = body.video_seconds ?? body.duration_seconds ?? 5;
    const wanSeconds = (parseInt(requestedSeconds || 5, 10) <= 5 ? 5 : 10);
    const wanSize = ((r)=> r==="9:16"?"1080*1920": r==="16:9"?"1920*1080":"1080*1080")(ratio);

    const [w, h] = ratio === "9:16" ? [1080,1920] : ratio === "16:9" ? [1920,1080] : [1080,1080];
    const imageModelKey = chooseImageModelKey({ idea, style, hint: imageHint });

    const lengthTargets = { short: 120, medium: 220, long: 400 };
    const maxChars = lengthTargets[length] || 220;
    const wantsEmoji = !!opts.emojis;
    const wantsHash = !!opts.auto_hashtags;

    // GPT
    let caption = null, vprompt = null, gptUsed = false;
    if (!image_only) {
      const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
      try {
        const user = `
Write a short social caption AND a clean visual prompt.

Constraints:
- Tone: ${preset}
- CTA: ${cta}
- Category: ${category}${subcategory ? " / " + subcategory : ""}
- Emojis: ${wantsEmoji ? "ON (1–3)" : "OFF"}
- Hashtags: ${wantsHash ? "ON (2–4)" : "OFF"}
- No URLs.
- Max ~${maxChars} chars.
- Visual prompt: no on-frame text, text-safe area.
- Aspect ratio: ${ratio}
- Style: ${style}

Idea: "${idea}"

Return JSON:
{
  "caption": "...",
  "visual_prompt": "..."
}`.trim();

        const resp = await openai.chat.completions.create({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: user }],
          temperature: 0.9,
          max_tokens: 500,
        });
        const text = resp?.choices?.[0]?.message?.content || "";
        const s = text.indexOf("{"), e = text.lastIndexOf("}");
        const j = JSON.parse(s >= 0 && e >= 0 ? text.slice(s, e + 1) : "{}");
        caption = (j.caption || "").toString().trim();
        vprompt = (j.visual_prompt || "").toString().trim();
        if (!caption) caption = `✨ ${idea}\n🚀 Learn more and take action today.\n\n➡️ Learn more\n\nhttps://hi-ai.ai #ai #automation #creativity`.slice(0, maxChars);
        if (!vprompt) vprompt = `${idea}. Clean. AR ${ratio}.`;
        if (caption.length > Math.round(maxChars * 1.1)) {
          const cut = caption.slice(0, Math.round(maxChars * 1.1));
          const idx = Math.max(cut.lastIndexOf(". "), cut.lastIndexOf(" "), Math.floor(cut.length * 0.9));
          caption = cut.slice(0, idx).trim() + "…";
        }
        gptUsed = true;
      } catch (e) {
        caption = `✨ ${idea}\n🚀 Learn more and take action today.\n\n➡️ Learn more\n\nhttps://hi-ai.ai #ai #automation #creativity`.slice(0, maxChars);
        vprompt = `${idea}. Clean. AR ${ratio}.`;
      }
    }

    // VISUAL
    let video_url = null, image_url = null;

    if (text_only) {
      // ничего
    } else if (image_only && opts.image !== false) {
      const versionImg = imageModelKey === "flux"
        ? process.env.REPLICATE_MODEL_VERSION_FLUX
        : process.env.REPLICATE_MODEL_VERSION_SDXL;
      if (versionImg) {
        const inputI = imageModelKey === "flux"
          ? { prompt: vprompt, go_fast:false, megapixels:"1", num_outputs:1, output_format:"png", output_quality:90 }
          : { prompt: vprompt, width:w, height:h, num_inference_steps:30, guidance_scale:7.0, num_outputs:1 };
        try {
          const jobI = await replicatePredict(versionImg, inputI);
          const outI = jobI?.output;
          image_url = Array.isArray(outI) ? outI[0] : (typeof outI === "string" ? outI : null);
        } catch (e) {}
      }
    } else if (full_mode) {
      const versionVideo = process.env.REPLICATE_MODEL_VERSION_VIDEO;
      if (versionVideo) {
        const inputV = {
          prompt: vprompt, size: wanSize, duration: wanSeconds,
          negative_prompt: "text, logo, watermark, letter", enable_prompt_expansion: true
        };
        try {
          const job = await replicatePredict(versionVideo, inputV);
          const out = job?.output;
          video_url = typeof out === "string" ? out : null;
        } catch (e) {}
      }
      if (!video_url && opts.image !== false) {
        const versionImg = imageModelKey === "flux"
          ? process.env.REPLICATE_MODEL_VERSION_FLUX
          : process.env.REPLICATE_MODEL_VERSION_SDXL;
        if (versionImg) {
          const inputI = imageModelKey === "flux"
            ? { prompt: vprompt, go_fast:false, megapixels:"1", num_outputs:1, output_format:"png", output_quality:90 }
            : { prompt: vprompt, width:w, height:h, num_inference_steps:30, guidance_scale:7.0, num_outputs:1 };
          try {
            const jobI = await replicatePredict(versionImg, inputI);
            const outI = jobI?.output;
            image_url = Array.isArray(outI) ? outI[0] : (typeof outI === "string" ? outI : null);
          } catch (e) {}
        }
      }
    }

    return res.json({
      ok: true, caption, vprompt, video_url: video_url || null, image_url: image_url || null,
      gpt_used: !!gptUsed, ratio, seconds: wanSeconds, size_used: wanSize,
      mode: text_only ? "text_only" : image_only ? "image_only" : video_url ? "video" : image_url ? "image_fallback" : "text_only"
    });
  } catch (e) {
    res.status(200).json({ ok:false, error:String(e.message||e) });
  }
});

/* ====================== START ====================== */
const port = process.env.PORT || 8080;
app.listen(port, () => console.log(`HI-AI backend on :${port}`));
