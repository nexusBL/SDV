import { Maximize2, Camera } from "lucide-react";

const CAMS = [
  { id: 0, label: "Front CSI" },
  { id: 1, label: "Left CSI" },
  { id: 2, label: "Right CSI" },
  { id: 3, label: "Rear CSI" },
];

export default function CsiCameraGrid({ backendUrl, onFullscreen }) {
  return (
    <div className="card csi-card" data-testid="csi-panel">
      <div className="card-head">
        <div>
          <div className="card-title">CSI Camera Array</div>
          <div className="card-sub">360° Surround View</div>
        </div>
      </div>
      <div className="card-body">
        <div className="csi-grid">
          {CAMS.map((cam) => (
            <div key={cam.id} className="csi-item">
              <img 
                src={`${backendUrl}/api/video/csi/${cam.id}`} 
                alt={cam.label}
                className="csi-feed"
                onDoubleClick={() => onFullscreen(cam.id, cam.label)}
              />
              <div className="csi-meta">
                <span>{cam.label}</span>
                <button 
                  className="mini-btn" 
                  onClick={() => onFullscreen(cam.id, cam.label)}
                >
                  <Maximize2 size={14} />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
