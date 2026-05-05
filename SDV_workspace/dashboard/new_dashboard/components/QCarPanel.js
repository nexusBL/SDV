import { Battery, Gauge, RotateCcw, Activity } from "lucide-react";

export default function QCarPanel({ telemetry }) {
  const { battery, speed, steering_angle, mode } = telemetry;

  return (
    <div className="card qcar-card" data-testid="qcar-panel">
      <div className="card-head">
        <div>
          <div className="card-title">Vehicle State</div>
          <div className="card-sub">Real-time Telemetry</div>
        </div>
        <div className={`mode-badge ${mode.toLowerCase()}`}>{mode}</div>
      </div>
      <div className="card-body">
        <div className="tele-grid">
          <div className="tele-item">
            <div className="tele-label"><Battery /> Battery</div>
            <div className="tele-val">{battery.toFixed(1)}%</div>
            <div className="bar-wrap">
              <div className="bar-fill" style={{ width: `${battery}%` }} />
            </div>
          </div>
          <div className="tele-item">
            <div className="tele-label"><Gauge /> Speed</div>
            <div className="tele-val">{speed.toFixed(2)} m/s</div>
          </div>
          <div className="tele-item">
            <div className="tele-label"><RotateCcw /> Steering</div>
            <div className="tele-val">{steering_angle.toFixed(1)}°</div>
          </div>
          <div className="tele-item">
            <div className="tele-label"><Activity /> Health</div>
            <div className="tele-val">STABLE</div>
          </div>
        </div>
      </div>
    </div>
  );
}
