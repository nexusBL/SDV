import { X } from "lucide-react";
import LidarPanel from "@/components/LidarPanel"; // Reuse

export default function FullScreenModal({ 
  type, backendUrl, lidarData, rsView, camId, label, onClose 
}) {
  return (
    <div className="fs-overlay" onClick={onClose}>
      <div className="fs-content" onClick={e => e.stopPropagation()}>
        <button className="fs-close" onClick={onClose}><X size={32} /></button>
        
        {type === "realsense" && (
          <div className="fs-view-wrap">
            <img 
              src={`${backendUrl}/api/video/realsense?view=${rsView}`} 
              className="fs-full-img" 
              alt="RealSense Full"
            />
            <div className="fs-label">RealSense Stream &middot; {rsView}</div>
          </div>
        )}

        {type === "csi" && (
          <div className="fs-view-wrap">
            <img 
              src={`${backendUrl}/api/video/csi/${camId}`} 
              className="fs-full-img" 
              alt={label}
            />
            <div className="fs-label">{label} &middot; Live</div>
          </div>
        )}

        {type === "lidar" && (
          <div className="fs-lidar-wrap">
             <LidarPanel lidarData={lidarData} onFullscreen={() => {}} />
          </div>
        )}
      </div>
    </div>
  );
}
