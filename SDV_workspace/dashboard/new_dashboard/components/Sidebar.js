import { Camera, Radar, LayoutGrid, MonitorPlay, Home, Settings, BarChart3, Bluetooth } from "lucide-react";

const NAV = [
  { id: "home", icon: Home },
  { id: "realsense", icon: Camera, label: "RealSense" },
  { id: "lidar", icon: Radar, label: "LiDAR" },
  { id: "csi", icon: LayoutGrid, label: "CSI Cameras" },
  { id: "settings", icon: Settings, label: "Settings" },
  { id: "stats", icon: BarChart3, label: "Stats" },
  { id: "bluetooth", icon: Bluetooth, label: "Bluetooth" },
];

export default function Sidebar({ active, onNavigate }) {
  return (
    <nav className="sidebar" data-testid="sidebar">
      <div className="sidebar-logo">
        <MonitorPlay />
      </div>

      {NAV.map((item) => (
        <button
          key={item.id}
          className={`sidebar-btn ${active === item.id ? "active" : ""}`}
          data-testid={`sidebar-${item.id}`}
          onClick={() => {
            if (item.label) onNavigate(item.id);
          }}
        >
          <item.icon />
          {item.label && <span className="sidebar-tip">{item.label}</span>}
        </button>
      ))}

      <div className="sidebar-space" />
    </nav>
  );
}



