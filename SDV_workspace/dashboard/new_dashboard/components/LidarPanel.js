import { Maximize2, Compass } from "lucide-react";

export default function LidarPanel({ lidarData, onFullscreen }) {
  // lidarData is expected to be an array of distances
  const maxDist = 6.0;
  
  return (
    <div className="card lidar-card" data-testid="lidar-panel">
      <div className="card-head">
        <div>
          <div className="card-title">LiDAR Scan</div>
          <div className="card-sub">2D Point Cloud &middot; 360°</div>
        </div>
        <button className="icon-btn" onClick={onFullscreen}>
          <Maximize2 />
        </button>
      </div>
      <div className="card-body">
        <div className="lidar-viz">
          <svg viewBox="-100 -100 200 200" className="lidar-svg">
            <circle cx="0" cy="0" r="100" className="grid-outer" />
            <circle cx="0" cy="0" r="66" className="grid-inner" />
            <circle cx="0" cy="0" r="33" className="grid-inner" />
            <line x1="-100" y1="0" x2="100" y2="0" className="grid-axis" />
            <line x1="0" y1="-100" x2="0" y2="100" className="grid-axis" />
            
            {lidarData?.map((dist, i) => {
              if (dist <= 0) return null;
              const angle = (i * (360 / lidarData.length) - 90) * (Math.PI / 180);
              const r = (dist / maxDist) * 100;
              const x = Math.cos(angle) * r;
              const y = Math.sin(angle) * r;
              return <circle key={i} cx={x} cy={y} r="1.5" className="lidar-pt" />;
            })}
            
            <circle cx="0" cy="0" r="5" className="car-center" />
          </svg>
          <div className="lidar-overlay">
            <Compass className="comp-icon" />
            <span>FRONT</span>
          </div>
        </div>
      </div>
    </div>
  );
}
