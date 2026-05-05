import { Mic, Activity } from "lucide-react";

export default function TopHeader({ connected }) {
  return (
    <header className="top-header" data-testid="dashboard-header">
      <div className="header-brand">NTT</div>

      <div className="header-center">
        <div className="header-search">
          <Mic />
          <span>Give a voice command</span>
        </div>
        <span className="header-link">Control panel</span>
        <span className="header-link">Add a panel</span>
      </div>

      <div className="header-right">
        <div className="theme-toggle">
          <button className="theme-btn active">Light</button>
          <button className="theme-btn">Dark</button>
        </div>

        <div className="notif-dot" />

        <div className="user-avatar">
          <Activity size={14} />
        </div>

        <span className="header-text">
          Hello, <strong>User</strong>
        </span>

        <div className="status-indicator" data-testid="connection-status">
          <div className={`status-led ${connected ? "on" : "off"}`} />
          <span className="status-label">
            {connected ? "Live" : "Sim"}
          </span>
        </div>
      </div>
    </header>
  );
}



