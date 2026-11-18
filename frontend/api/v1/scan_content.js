export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ detail: "Method not allowed." });
  }

  const backend = process.env.SENTINEL_BACKEND_URL?.replace(/\/+$/, "");
  const apiKey = process.env.SENTINEL_API_KEY;

  if (!backend || !apiKey) {
    return res.status(503).json({ detail: "Backend configuration missing." });
  }

  try {
    const response = await fetch(`${backend}/v1/scan_content`, {
      method: "POST",
      headers: {
        "X-Sentinel-API-Key": apiKey,
        "Content-Type": "application/json",
        "Accept": "application/json"
      },
      body: JSON.stringify(req.body),
      cache: "no-store",
      signal: AbortSignal.timeout(120000)
    });

    const body = await response.text();
    res.setHeader("Cache-Control", "no-store");
    res.setHeader("Content-Type", "application/json");
    return res.status(response.status).send(body);
  } catch {
    return res.status(502).json({ detail: "Backend unavailable." });
  }
}