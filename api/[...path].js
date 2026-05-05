export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req, res) {
  const baseUrl =
    "http://formcheck-ai-api.eba-pvfk7qtv.us-west-2.elasticbeanstalk.com";

  const path = Array.isArray(req.query.path)
    ? req.query.path.join("/")
    : "";

  try {
    const upstream = await fetch(`${baseUrl}/${path}`, {
      method: req.method,
      headers: {
        ...req.headers,
        host: undefined,
      },
      body: req.method === "GET" || req.method === "HEAD" ? undefined : req,
      duplex: "half",
    });

    const buffer = Buffer.from(await upstream.arrayBuffer());

    res.status(upstream.status);
    upstream.headers.forEach((value, key) => {
      if (!["content-encoding", "transfer-encoding"].includes(key)) {
        res.setHeader(key, value);
      }
    });

    res.send(buffer);
  } catch (err) {
    res.status(500).json({
      error: "Proxy failed",
      message: err.message,
    });
  }
}