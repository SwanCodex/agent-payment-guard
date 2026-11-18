export default async function handler(req, res) {
  if (req.method !== "GET") {
    return res.status(405).json({ detail: "Method not allowed." });
  }

  const segments = Array.isArray(req.query.path)
    ? req.query.path
    : [req.query.path].filter(Boolean);

  const route = `/api/${segments.join("/")}`;

  const allowedRoutes = new Set([
    "/api/health",
    "/api/dashboard/summary"
  ]);

  if (!allowedRoutes.has(route)) {
    return res.status(403).json({ detail: "Endpoint not publicly available." });
  }

  const backend = process.env.SENTINEL_BACKEND_URL?.replace(/\/+$/, "");
  const apiKey = process.env.SENTINEL_API_KEY;

  if (!backend || !apiKey) {
    return res.status(503).json({ detail: "Backend configuration missing." });
  }

  try {
    const response = await fetch(`${backend}${route}`, {
      method: "GET",
      headers: {
        "X-Sentinel-API-Key": apiKey,
        "Accept": "application/json"
      },
      cache: "no-store",
      signal: AbortSignal.timeout(15000)
    });

    const body = await response.text();

    res.setHeader("Cache-Control", "no-store");
    res.setHeader("Content-Type", "application/json");

    return res.status(response.status).send(body);
  } catch {
    return res.status(502).json({ detail: "Backend unavailable." });
  }
}