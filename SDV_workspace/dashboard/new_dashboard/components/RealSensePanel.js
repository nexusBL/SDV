import { useState, useRef, useEffect } from "react";
import { ChevronDown, Maximize2, Eye } from "lucide-react";

const VIEWS = [
  { value: "rgb", label: "RGB View" },
  { value: "depth", label: "Depth View" },
  { value: "infrared", label: "Infrared View" },
];

export default function RealSensePanel({ backendUrl, view, onViewChange, onFullscreen }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const h = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  const current = VIEWS.find((v) => v.value === view)?.label || "RGB View";
  const url = `${backendUrl}/api/video/realsense?view=${view}`;

  return (
    <div className="card realsense-card" data-testid="realsense-panel">
      <div className="card-head">
        <div>
          <div className="card-title">RealSense Camera</div>
          <div className="card-sub">Live feed &middot; {current}</div>
        </div>
        <div className="card-acts">
          <div className="rs-dropdown" ref={ref}>
            <button
              className="rs-trigger"
              data-testid="realsense-dropdown-btn"
              onClick={() => setOpen((p) => !p)}
            >
              <Eye />
              {current}
              <ChevronDown />
            </button>
            {open && (
              <div className="rs-menu" data-testid="realsense-dropdown-menu">
                {VIEWS.map((v) => (
                  <button
                    key={v.value}
                    className={`rs-item ${view === v.value ? "on" : ""}`}
                    data-testid={`realsense-view-${v.value}`}
                    onClick={() => { onViewChange(v.value); setOpen(false); }}
                  >
                    {v.label}
                  </button>
                ))}
              </div>
            )}
          </div>
          <button
            className="icon-btn"
            data-testid="realsense-fullscreen-btn"
            onClick={onFullscreen}
          >
            <Maximize2 />
          </button>
        </div>
      </div>
      <div className="card-body">
        <div
          className="feed-wrap"
          data-testid="realsense-video-feed"
          onDoubleClick={onFullscreen}
        >
          <img className="feed-img" src={url} alt="RealSense" />
          <span className="feed-badge">RealSense</span>
          <div className="feed-rec" />
          <span className="feed-hint">Double-click to expand</span>
        </div>
      </div>
    </div>
  );
}
