
export const config = {
  api: { bodyParser: false }
};

const MAX_BYTES = 4 * 1024 * 1024;

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ detail: "Method not allowed." });
  }

  const backend = process.env.SENTINEL_BACKEND_URL?.replace(/\/+$/, "");
  const apiKey = process.env.SENTINEL_API_KEY;
  const contentType = req.headers["content-type"] || "";

  if (!backend || !apiKey) {
    return res.status(503).json({ detail: "Backend configuration missing." });
  }

  if (!contentType.toLowerCase().startsWith("multipart/form-data;")) {
    return res.status(415).json({ detail: "Multipart PDF upload required." });
  }

  const contentLength = Number(req.headers["content-length"] || 0);
  if (contentLength > MAX_BYTES) {
    return res.status(413).json({ detail: "PDF upload must be under 4 MB." });
  }

  try {
    const chunks = [];
    let totalBytes = 0;

    for await (const chunk of req) {
      totalBytes += chunk.length;

      if (totalBytes > MAX_BYTES) {
        return res.status(413).json({ detail: "PDF upload must be under 4 MB." });
      }

      chunks.push(chunk);
    }

    const response = await fetch(`${backend}/v1/scan_pdf`, {
      method: "POST",
      headers: {
        "X-Sentinel-API-Key": apiKey,
        "Content-Type": contentType,
        "Accept": "application/json"
      },
      body: Buffer.concat(chunks),
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
