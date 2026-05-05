// Protocol detection for WebSockets
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const host = window.location.host;
const telemetryUrl = `${protocol}//${host}/ws/telemetry`;
const lidarUrl = `${protocol}//${host}/ws/lidar`;

let teleWs, lidarWs;
let isConnected = false;

function connectWebsockets() {
    // --- Telemetry WebSocket ---
    teleWs = new WebSocket(telemetryUrl);
    
    teleWs.onopen = () => {
        isConnected = true;
        console.log("✅ Telemetry connected");
    };
    
    teleWs.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateUI(data);
    };
    
    teleWs.onclose = () => {
        isConnected = false;
        console.log("❌ Telemetry disconnected. Retrying...");
        setTimeout(connectWebsockets, 3000);
    };

    // --- LiDAR WebSocket ---
    lidarWs = new WebSocket(lidarUrl);
    lidarWs.onmessage = (event) => {
        try {
            // Clean valid json if backend still sends Infinity
            const cleanData = event.data.replace(/NaN/g, "null").replace(/Infinity/g, "null");
            const data = JSON.parse(cleanData);
            updateLidar(data);
        } catch (e) {
            console.error("LiDAR JSON Parse Error:", e);
            console.log("Raw bad JSON:", event.data.substring(0, 100)); // Sample bad data
        }
    };
}

// --- UI Element Selectors ---
const batteryEl = document.getElementById('battery-val');
const steeringEl = document.getElementById('steering-val');
const speedEl = document.getElementById('speed-val');

function updateUI(state) {
    if (state.battery !== undefined) {
        batteryEl.textContent = `${state.battery.toFixed(1)}%`;
        batteryEl.style.color = state.battery < 20 ? '#ef4444' : '';
    }
    
    if (state.steering_angle !== undefined) {
        steeringEl.textContent = `${state.steering_angle.toFixed(1)}°`;
    }
    
    if (state.speed !== undefined) {
        speedEl.textContent = `${state.speed.toFixed(2)} m/s`;
    }
}

// --- LiDAR 2D Radar Visualization (Canvas) ---
const canvas = document.getElementById('lidar-canvas');
const ctx = canvas.getContext('2d');
let lidarData = { angles: [], distances: [] };

function resizeCanvas() {
    const container = document.querySelector('.lidar-view');
    canvas.width = container.clientWidth || 400;
    canvas.height = container.clientHeight || 400;
    drawRadar(); // Redraw immediately on resize
}
window.addEventListener('resize', resizeCanvas);

// Radar visual settings matching lidar_radar.py
const maxRadiusMeters = 10.0;
const ringStep = 1.0; // 1m intervals

function drawRadar() {
    const w = canvas.width;
    const h = canvas.height;
    const centerX = w / 2;
    const centerY = h / 2;
    // Leave some padding
    const scale = Math.min(w, h) / 2 / (maxRadiusMeters * 1.05);

    // Clear canvas
    ctx.clearRect(0, 0, w, h);

    // 1. Draw Concentric Circles and Crosshairs
    ctx.strokeStyle = 'rgba(40, 60, 80, 0.6)'; // Subtle blue-grey grid
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, h);
    ctx.moveTo(0, centerY);
    ctx.lineTo(w, centerY);
    ctx.stroke();

    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillStyle = 'rgba(130, 150, 170, 0.8)';
    ctx.font = '10px monospace';

    for (let r = ringStep; r <= maxRadiusMeters; r += ringStep) {
        ctx.beginPath();
        ctx.arc(centerX, centerY, r * scale, 0, 2 * Math.PI);
        ctx.stroke();
        
        // Labels
        ctx.fillText(`${r}m`, centerX + r * scale + 2, centerY - 2);
    }

    // 2. Plot Points
    const angles = lidarData.angles || [];
    const distances = lidarData.distances || [];
    
    if (distances.length === 0) return;

    let closestDist = Infinity;
    let closestX = 0, closestY = 0;

    ctx.fillStyle = 'rgba(0, 255, 255, 0.8)'; // Cyan points
    for (let i = 0; i < distances.length; i++) {
        const dist = distances[i];
        if (dist < 0.05 || dist > maxRadiusMeters) continue;

        const angle = angles[i]; // in radians
        const x = centerX + (dist * Math.cos(angle)) * scale;
        const y = centerY - (dist * Math.sin(angle)) * scale; // Invert Y for canvas

        // Draw point
        ctx.fillRect(x - 1.5, y - 1.5, 3, 3);

        if (dist < closestDist) {
            closestDist = dist;
            closestX = x;
            closestY = y;
        }
    }

    // 3. Highlight closest point
    if (closestDist !== Infinity) {
        ctx.fillStyle = 'rgba(255, 50, 50, 1)'; // Red star/highlight
        ctx.beginPath();
        ctx.arc(closestX, closestY, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.stroke();
        
        ctx.fillStyle = 'rgba(255, 50, 50, 1)';
        ctx.textAlign = 'right';
        ctx.fillText(`⚠️ ${closestDist.toFixed(2)}m`, closestX - 8, closestY - 8);
    }
}

function updateLidar(data) {
    if (data && data.angles && data.distances) {
        lidarData = data;
    } else if (Array.isArray(data)) {
        // Fallback if array
        lidarData = {
            distances: data,
            angles: data.map((_, i) => i * (Math.PI * 2 / data.length))
        };
    }
    requestAnimationFrame(drawRadar);
}

// --- Offline Fallback Logic ---
setInterval(() => {
    if (!isConnected) {
        const t = performance.now() / 1000;
        updateUI({
            battery: 100 - (t % 100) / 10,
            steering_angle: Math.sin(t) * 25,
            speed: 1.2 + Math.sin(t/2) * 0.5
        });
        
        // Local mock lidar animation in canvas
        const numPoints = 720;
        const mockAngles = [];
        const mockDists = [];
        for (let i = 0; i < numPoints; i++) {
            const a = i * (Math.PI * 2 / numPoints);
            mockAngles.push(a);
            mockDists.push(6.0 + 1.5 * Math.sin(a * 4 + t*2) + Math.random() * 0.1);
        }
        updateLidar({ angles: mockAngles, distances: mockDists });
    }
}, 100);

// Initialize
setTimeout(resizeCanvas, 100);
connectWebsockets();

