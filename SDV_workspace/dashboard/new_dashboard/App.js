import { useState, useEffect, useCallback, useRef } from "react";
import "@/App.css";
import Sidebar from "@/components/Sidebar";
import TopHeader from "@/components/TopHeader";
import RealSensePanel from "@/components/RealSensePanel";
import LidarPanel from "@/components/LidarPanel";
import QCarPanel from "@/components/QCarPanel";
import CsiCameraGrid from "@/components/CsiCameraGrid";
import FullScreenModal from "@/components/FullScreenModal";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "";

function App() {
  const [telemetry, setTelemetry] = useState({
    battery: 100.0,
    steering_angle: 0.0,
    speed: 0.0,
    mode: "SIMULATION",
  });
  const [lidarData, setLidarData] = useState([]);
  const [connected, setConnected] = useState(false);
  const [fullscreen, setFullscreen] = useState(null);
  const [rsView, setRsView] = useState("rgb");
  const [activeSection, setActiveSection] = useState("realsense");
  const teleWsRef = useRef(null);
  const lidarWsRef = useRef(null);

  useEffect(() => {
    let wsBase;
    if (BACKEND_URL) {
      wsBase = BACKEND_URL.replace(/^http/, "ws");
    } else {
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      wsBase = `${proto}//${window.location.host}`;
    }

    function connectTelemetry() {
      const ws = new WebSocket(`${wsBase}/api/ws/telemetry`);
      ws.onopen = () => setConnected(true);
      ws.onmessage = (e) => {
        try { setTelemetry(JSON.parse(e.data)); } catch {}
      };
      ws.onclose = () => {
        setConnected(false);
        setTimeout(connectTelemetry, 3000);
      };
      teleWsRef.current = ws;
    }

    function connectLidar() {
      const ws = new WebSocket(`${wsBase}/api/ws/lidar`);
      ws.onmessage = (e) => {
        try { setLidarData(JSON.parse(e.data)); } catch {}
      };
      ws.onclose = () => setTimeout(connectLidar, 3000);
      lidarWsRef.current = ws;
    }

    connectTelemetry();
    connectLidar();
    return () => {
      if (teleWsRef.current) teleWsRef.current.close();
      if (lidarWsRef.current) lidarWsRef.current.close();
    };
  }, []);

  useEffect(() => {
    if (connected) return;
    const iv = setInterval(() => {
      const t = performance.now() / 1000;
      setTelemetry({
        battery: 100 - (t % 100) / 10,
        steering_angle: Math.sin(t) * 25,
        speed: 1.2 + Math.sin(t / 2) * 0.5,
        mode: "SIMULATION",
      });
      setLidarData(
        Array.from({ length: 40 }, (_, i) =>
          1.5 + Math.sin(i * 0.1 + t) * 0.5 + Math.random() * 0.2
        )
      );
    }, 100);
    return () => clearInterval(iv);
  }, [connected]);

  useEffect(() => {
    const handler = (e) => { if (e.key === "Escape") setFullscreen(null); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const openFs = useCallback((type, extra) => {
    setFullscreen({ type, ...extra });
  }, []);

  const sidebarNav = useCallback((section) => {
    setActiveSection(section);
    openFs(section, { label: section });
  }, [openFs]);

  return (
    <div className="app-root" data-testid="dashboard-layout">
      <Sidebar active={activeSection} onNavigate={sidebarNav} />
      <div className="app-right">
        <TopHeader connected={connected} />
        <div className="main-area">
          <div className="dash-grid" data-testid="dashboard-grid">
            <RealSensePanel
              backendUrl={BACKEND_URL}
              view={rsView}
              onViewChange={setRsView}
              onFullscreen={() => openFs("realsense", { view: rsView })}
            />
            <LidarPanel
              lidarData={lidarData}
              onFullscreen={() => openFs("lidar")}
            />
            <QCarPanel telemetry={telemetry} />
            <CsiCameraGrid
              backendUrl={BACKEND_URL}
              onFullscreen={(camId, label) => openFs("csi", { camId, label })}
            />
          </div>
        </div>
      </div>

      {fullscreen && (
        <FullScreenModal
          type={fullscreen.type}
          backendUrl={BACKEND_URL}
          lidarData={lidarData}
          rsView={rsView}
          onRsViewChange={setRsView}
          camId={fullscreen.camId}
          label={fullscreen.label}
          onClose={() => setFullscreen(null)}
        />
      )}
    </div>
  );
}

export default App;



