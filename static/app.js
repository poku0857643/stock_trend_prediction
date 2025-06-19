async function predictTrend() {
  const ticker = document.getElementById("trendTicker").value;
  const output = document.getElementById("trendResult");

  if (!ticker) return alert("Please enter a ticker");

  try {
    const res = await fetch(`/trend_predict?ticker=${ticker}`);
    const data = await res.json();
    output.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    output.textContent = "Error: " + err.message;
  }
}

async function generateStrategy() {
  const prompt = document.getElementById("strategyPrompt").value;
  const tickers = document.getElementById("strategyTickers").value.split(",").map(t => t.trim());
  const useInternal = document.getElementById("useInternal").checked;
  const output = document.getElementById("strategyResult");

  const payload = {
    strategy_prompt: prompt,
    tickers: tickers,
    use_internal_trend_api: useInternal,
    local_folder: null,
    outsource_folder: null,
    cloud_folder: null,
    api_headers: null,
    trend_prediction: null
  };

  try {
    const res = await fetch("/generate_strategies", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    output.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    output.textContent = "Error: " + err.message;
  }
}