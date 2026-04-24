let dashboardData = [];
let currentLambdaIndex = 0;

// Chart Instances
let tradeoffChart = null;
let histogramChart = null;

// DOM Elements
const slider = document.getElementById('lambda-slider');
const lambdaValueDisplay = document.getElementById('lambda-value');
const accValueDisplay = document.getElementById('acc-value');
const sparsityValueDisplay = document.getElementById('sparsity-value');
const canvas = document.getElementById('networkCanvas');
const ctx = canvas.getContext('2d');

// Helper to format numbers
const formatNum = (num) => Number(num).toFixed(2);

// Network Config
const layers = [6, 5, 4, 3, 2]; // Number of nodes per layer (abstract representation)

async function loadData() {
    try {
        const response = await fetch('dashboard_data.json');
        dashboardData = await response.json();
        
        // Setup slider
        slider.max = dashboardData.length - 1;
        slider.value = 0;
        
        // Init visualization
        initCharts();
        updateDashboard(0);
        
        // Listeners
        slider.addEventListener('input', (e) => {
            updateDashboard(parseInt(e.target.value));
        });
        
        // Handle window resize for canvas
        window.addEventListener('resize', drawNetwork);
    } catch (err) {
        console.error("Failed to load dashboard data:", err);
    }
}

function updateDashboard(index) {
    currentLambdaIndex = index;
    const data = dashboardData[index];
    
    // Update Stats
    lambdaValueDisplay.innerText = data.lam.toExponential(1);
    accValueDisplay.innerText = formatNum(data.accuracy) + "%";
    sparsityValueDisplay.innerText = formatNum(data.sparsity) + "%";
    
    // Update Charts & Network
    updateTradeoffChart(index);
    updateHistogramChart(data);
    drawNetwork();
}

function initCharts() {
    const lams = dashboardData.map(d => d.lam.toExponential(1));
    const accs = dashboardData.map(d => d.accuracy);
    const sparsities = dashboardData.map(d => d.sparsity);
    
    // 1. Trade-off Chart (Accuracy vs Sparsity)
    const ctxTradeoff = document.getElementById('tradeoffChart').getContext('2d');
    tradeoffChart = new Chart(ctxTradeoff, {
        type: 'line',
        data: {
            labels: lams,
            datasets: [
                {
                    label: 'Accuracy (%)',
                    data: accs,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    yAxisID: 'y',
                    tension: 0.3,
                    borderWidth: 2,
                    pointRadius: 4,
                },
                {
                    label: 'Sparsity (%)',
                    data: sparsities,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.3,
                    borderWidth: 2,
                    pointRadius: 4,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: { type: 'linear', display: true, position: 'left', title: {display: true, text: 'Accuracy', color: '#94a3b8'} },
                y1: { type: 'linear', display: true, position: 'right', grid: { drawOnChartArea: false }, title: {display: true, text: 'Sparsity', color: '#94a3b8'} },
                x: { ticks: { color: '#94a3b8' } }
            },
            plugins: {
                legend: { labels: { color: '#f0f2f5' } }
            }
        }
    });

    // 2. Histogram Chart
    const ctxHist = document.getElementById('histogramChart').getContext('2d');
    histogramChart = new Chart(ctxHist, {
        type: 'bar',
        data: {
            labels: dashboardData[0].hist_bins.slice(0, -1).map(b => b.toFixed(2)),
            datasets: [{
                label: 'Gate Value Counts',
                data: dashboardData[0].hist_counts,
                backgroundColor: 'rgba(59, 130, 246, 0.7)',
                borderColor: '#3b82f6',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, title: {display: true, text: 'Count', color: '#94a3b8'} },
                x: { title: {display: true, text: 'Gate Value (0 = Pruned, 1 = Kept)', color: '#94a3b8'} }
            },
            plugins: {
                legend: { display: false }
            },
            animation: {
                duration: 400
            }
        }
    });
}

function updateTradeoffChart(index) {
    // Highlight the current point
    const accPoints = tradeoffChart.data.datasets[0].pointRadius;
    const sparsityPoints = tradeoffChart.data.datasets[1].pointRadius;
    
    // Reset all to 4, set current to 10
    const newAccRadii = dashboardData.map((_, i) => i === index ? 10 : 4);
    const newSparsityRadii = dashboardData.map((_, i) => i === index ? 10 : 4);
    
    tradeoffChart.data.datasets[0].pointRadius = newAccRadii;
    tradeoffChart.data.datasets[0].pointBackgroundColor = dashboardData.map((_, i) => i === index ? '#fff' : '#10b981');
    
    tradeoffChart.data.datasets[1].pointRadius = newSparsityRadii;
    tradeoffChart.data.datasets[1].pointBackgroundColor = dashboardData.map((_, i) => i === index ? '#fff' : '#f59e0b');
    
    tradeoffChart.update();
}

function updateHistogramChart(data) {
    histogramChart.data.datasets[0].data = data.hist_counts;
    histogramChart.update();
}

function drawNetwork() {
    // Resize canvas
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height - 60; // minus title height
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const data = dashboardData[currentLambdaIndex];
    // layer_sparsities has 4 values for the 4 prunable linear layers
    const layerSparsities = data.layer_sparsities || [0, 0, 0, 0];
    
    const w = canvas.width;
    const h = canvas.height;
    
    const nodeRadius = Math.min(w * 0.015, 8);
    const xGap = w / (layers.length + 1);
    
    // Calculate node positions
    const nodePositions = [];
    
    layers.forEach((nodesInLayer, layerIdx) => {
        const x = xGap * (layerIdx + 1);
        const yGap = h / (nodesInLayer + 1);
        
        const currentLayerNodes = [];
        for (let i = 0; i < nodesInLayer; i++) {
            currentLayerNodes.push({
                x: x,
                y: yGap * (i + 1)
            });
        }
        nodePositions.push(currentLayerNodes);
    });
    
    // Draw edges
    for (let l = 0; l < nodePositions.length - 1; l++) {
        const currentNodes = nodePositions[l];
        const nextNodes = nodePositions[l+1];
        
        // Opacity inversely proportional to sparsity (0 sparsity = 1 opacity, 100 sparsity = 0.05 opacity)
        const sparsityPct = layerSparsities[l] || 0;
        let opacity = 1 - (sparsityPct / 100);
        opacity = Math.max(0.05, opacity); // keep a tiny bit visible
        
        ctx.strokeStyle = `rgba(59, 130, 246, ${opacity})`;
        ctx.lineWidth = opacity > 0.5 ? 1.5 : 0.5;
        
        currentNodes.forEach(startNode => {
            nextNodes.forEach(endNode => {
                ctx.beginPath();
                ctx.moveTo(startNode.x, startNode.y);
                ctx.lineTo(endNode.x, endNode.y);
                ctx.stroke();
            });
        });
    }
    
    // Draw nodes
    ctx.fillStyle = '#fff';
    nodePositions.forEach((layerNodes, l) => {
        layerNodes.forEach(node => {
            ctx.beginPath();
            ctx.arc(node.x, node.y, nodeRadius, 0, Math.PI * 2);
            ctx.fill();
            
            // Glow
            ctx.shadowColor = '#3b82f6';
            ctx.shadowBlur = 10;
        });
    });
    ctx.shadowBlur = 0; // reset
}

// Start
document.addEventListener('DOMContentLoaded', loadData);
