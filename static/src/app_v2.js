
// Chart configurations
Chart.defaults.color = 'rgba(255, 255, 255, 0.8)';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';
Chart.defaults.backgroundColor = 'rgba(255, 255, 255, 0.1)';

// Trend Analysis Chart
const trendCtx = document.getElementById('trendChart').getContext('2d');
const trendChart = new Chart(trendCtx, {
    type: 'line',
    data: {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        datasets: [{
            label: 'Current Trends',
            data: [65, 78, 85, 92, 88, 95],
            borderColor: '#4ade80',
            backgroundColor: 'rgba(74, 222, 128, 0.1)',
            borderWidth: 3,
            fill: true,
            tension: 0.4
        }, {
            label: 'Predicted Trends',
            data: [70, 82, 87, 89, 94, 98],
            borderColor: '#60a5fa',
            backgroundColor: 'rgba(96, 165, 250, 0.1)',
            borderWidth: 3,
            fill: true,
            tension: 0.4,
            borderDash: [5, 5]
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    usePointStyle: true,
                    color: 'rgba(255, 255, 255, 0.8)'
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: 'rgba(255, 255, 255, 0.8)'
                }
            },
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: 'rgba(255, 255, 255, 0.8)'
                }
            }
        }
    }
});

// Prediction Chart
const predictionCtx = document.getElementById('predictionChart').getContext('2d');
const predictionChart = new Chart(predictionCtx, {
    type: 'bar',
    data: {
        labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
        datasets: [{
            label: 'Predicted Growth',
            data: [12, 18, 25, 32],
            backgroundColor: [
                'rgba(74, 222, 128, 0.8)',
                'rgba(96, 165, 250, 0.8)',
                'rgba(251, 191, 36, 0.8)',
                'rgba(248, 113, 113, 0.8)'
            ],
            borderColor: [
                '#4ade80',
                '#60a5fa',
                '#fbbf24',
                '#f87171'
            ],
            borderWidth: 2,
            borderRadius: 8,
            borderSkipped: false
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: 'rgba(255, 255, 255, 0.8)'
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: 'rgba(255, 255, 255, 0.8)'
                }
            },
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: 'rgba(255, 255, 255, 0.8)'
                }
            }
        }
    }
});

// Sentiment Chart
const sentimentCtx = document.getElementById('sentimentChart').getContext('2d');
const sentimentChart = new Chart(sentimentCtx, {
    type: 'doughnut',
    data: {
        labels: ['Positive', 'Neutral', 'Negative'],
        datasets: [{
            data: [65, 25, 10],
            backgroundColor: [
                'rgba(74, 222, 128, 0.8)',
                'rgba(251, 191, 36, 0.8)',
                'rgba(248, 113, 113, 0.8)'
            ],
            borderColor: [
                '#4ade80',
                '#fbbf24',
                '#f87171'
            ],
            borderWidth: 2
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    color: 'rgba(255, 255, 255, 0.8)',
                    padding: 20
                }
            }
        }
    }
});

// Interactive functions
function setTimeframe(period) {
    // Remove active class from all buttons
    document.querySelectorAll('.btn').forEach(btn => btn.classList.remove('active'));
    // Add active class to clicked button
    event.target.classList.add('active');

    // Simulate data update
    updateCharts(period);
}

function refreshData() {
    // Simulate data refresh
    const accuracyEl = document.getElementById('accuracy');
    const trendsEl = document.getElementById('trends');
    const strategiesEl = document.getElementById('strategies');
    const roiEl = document.getElementById('roi');

    // Animate values
    animateValue(accuracyEl, 94.7, (Math.random() * 5 + 92).toFixed(1), '%');
    animateValue(trendsEl, 127, Math.floor(Math.random() * 50 + 100), '', '+');
    animateValue(strategiesEl, 23, Math.floor(Math.random() * 20 + 15), '');
    animateValue(roiEl, 342, Math.floor(Math.random() * 100 + 300), '%', '+');
}

function generateStrategy() {
    // Simulate strategy generation
    alert('🤖 AI Agent is generating new strategies based on current trends...\n\nNew strategy will be available in 30 seconds!');
}

function updateCharts(period) {
    // Simulate chart updates based on timeframe
    const newData = generateRandomData(period);

    trendChart.data.datasets[0].data = newData.trend;
    trendChart.update('active');

    predictionChart.data.datasets[0].data = newData.prediction;
    predictionChart.update('active');
}

function generateRandomData(period) {
    const multiplier = period === '1h' ? 0.5 : period === '24h' ? 1 : period === '7d' ? 2 : 4;

    return {
        trend: Array.from({length: 6}, () => Math.floor(Math.random() * 30 + 70) * multiplier),
        prediction: Array.from({length: 4}, () => Math.floor(Math.random() * 20 + 10) * multiplier)
    };
}

function animateValue(element, start, end, suffix = '', prefix = '') {
    const duration = 1000;
    const startTime = performance.now();
    const isFloat = end.toString().includes('.');

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        const current = start + (end - start) * progress;
        const displayValue = isFloat ? current.toFixed(1) : Math.floor(current);

        element.textContent = prefix + displayValue + suffix;

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// Auto-refresh data every 30 seconds
setInterval(refreshData, 30000);

// Initialize with random data updates
setTimeout(() => {
    refreshData();
}, 2000);

// repond cells
// trend prediction and generate strategies
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