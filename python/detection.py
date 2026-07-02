import sys
from turtle import width

from PIL import Image
from PIL.ExifTags import TAGS
import re
import torchvision.transforms as T
import torch
from util_2 import get_model_mask, read_metadata_exiftool, write_metadata_csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import math, csv, os

import csv
import os

def export_all_geobox_to_csv(detections, output_file="data.csv", folder_name="results"):
    os.makedirs(folder_name, exist_ok=True)
    csv_path = os.path.join(folder_name, output_file)

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Object Index", "Latitude", "Longitude", "Area (m^2)"])

        for det in detections:
            image_name = os.path.basename(det.path)

            for idx, ((lat, lon), area) in enumerate(zip(det.geobox, det.areas), start=1):
                writer.writerow([image_name, idx, float(lat), float(lon), float(area)])

    print(f"All geobox coordinates saved to: {csv_path}")
    return csv_path

from simplekml import Kml

def export_mask_gps_to_kml(mask_gps, output_folder="results", output_filename="coverage.kml"):

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    kml = Kml()

    for img_idx, img_polygons in enumerate(mask_gps):
        for poly_idx, polygon in enumerate(img_polygons):
            if not polygon:
                continue  # skip empty polygons
            # simplekml expects (lon, lat)
            coords = [(pt[1], pt[0]) for pt in polygon]
            kml_polygon = kml.newpolygon(name=f"Image {img_idx+1} - Poly {poly_idx+1}")
            kml_polygon.outerboundaryis = coords
            # KML colours are AABBGGRR (alpha, blue, green, red)
            kml_polygon.style.polystyle.color = '7d0000ff'  # semi-transparent red fill
            kml_polygon.style.linestyle.width = 2
            kml_polygon.style.linestyle.color = 'ff0000ff'  # red border

    kml.save(output_path)
    print(f"KML file saved to: {os.path.abspath(output_path)}")
    return os.path.abspath(output_path)

# # checkpoint = torch.load(model_path, map_location=device)
# # model.load_state_dict(checkpoint["model_state_dict"])

# # Image_detection("images/test5/test_set/5.jpg", model,device,0.6)
class Image_detection:
    def __init__(self, image_path, model, device, threshold):
        self.path=image_path
        self.device=device
        self.score_threshold=threshold
        self.model=model
        # self.sensor_width = 0.0064   # DJI Mavic (6.4 mm)
        self.sensor_width = 0.0173     # DJI Zenmuse L2 (17.3 mm)
        self.width=None
        self.boxes = []
        self.labels = []
        self.scores = []
        self.masks = []
        self.gimbal = None
        self.flight_orientation = None
        self.gps = []
        self.geobox = []
        self.geopolygon = []
        self.areas = []
        self.gsd = None
        self.altitude = None
        # gps, gsd, orientation = self.parse_metadata()
        #gps = [latitude, N or S , longitude, W or E], langitude and longitude = (degree, min, second)

        self.parse_metadata()
        self.predict()
        self.visualise()
        print(f"GPS: {self.gps}")
        
    def _dd_to_dms(self, dd):
        """Convert (signed) decimal degrees into a (deg, min, sec) tuple of absolute values."""
        dd = abs(float(dd))
        d = int(dd)
        m = int((dd - d) * 60)
        s = (dd - d - m / 60.0) * 3600.0
        return (d, m, s)

    def parse_metadata(self):
        # -------------------------------------------------------------------
        # OLD: metadata read with PIL EXIF (_getexif). Commented out and
        # replaced by the exiftool-based parsing below.
        # -------------------------------------------------------------------
        # image = Image.open(self.path)
        # exif_data = image._getexif()
        # print("here")
        # if exif_data:
        #     for i in range(1,5):
        #         self.gps.append(exif_data[34853][i]) #GPS
        #     self.gps.append(exif_data[34853][6])
        #     self.focal_length=exif_data[37386]
        #     self.image_width, self.image_height = image.size
        #     # self.image_width=exif_data[256]
        #     # self.image_height=exif_data[257]
        #     altitude = float(self.gps[4])       # meters
        #     # altitude = float(self.gps[4])-float(53)         # meters
        #     focal_m = float(self.focal_length) / 1000  # mm → meters
        #
        #     self.gsd = (altitude * self.sensor_width) / (focal_m * self.image_width)
        #     print(f"focal length: {focal_m}")
        #     print(f"gsd: {self.gsd} m/pixel")
        #
        #     # gsd = gps[4] * sensor_width/focal_length * width
        #     # self.gsd = (self.gps[4] * self.sensor_width) / (self.focal_length * self.image_width)
        #     makernote=exif_data[37500]
        #     makernote_str = makernote.decode('utf-8', errors='ignore')
        #
        #     gimbal_match = re.search(r'GimbalDegree\(Y,P,R\):(-?\d+),(-?\d+),(-?\d+)', makernote_str)
        #     if gimbal_match:
        #         gimbal_y, gimbal_p, gimbal_r = map(int, gimbal_match.groups())
        #         self.gimbal=(gimbal_y, gimbal_p, gimbal_r)
        #         # print("Gimbal Degrees:", gimbal_y, gimbal_p, gimbal_r)
        #         #Yaw, Pitch, Roll
        #     else:
        #         print("No gimbal")
        #     # Extract Flight
        #     # flight_match = re.search(r'FlightDegree\(Y,P,R\):-([\d\-]+),([\d\-]+),([\d\-]+)', makernote_str)
        #     flight_match = re.search(r'FlightDegree\(Y,P,R\):(-?\d+),(-?\d+),(-?\d+)', makernote_str)
        #     if flight_match:
        #         flight_y, flight_p, flight_r = map(int, flight_match.groups())
        #         self.flight_orientation=(flight_y, flight_p, flight_r)
        #         # print("Flight Degrees:", flight_y, flight_p, flight_r)
        #     else:
        #         print("No flight degrees")
        # else:
        #     print("No metadata available")
        # -------------------------------------------------------------------

        # NEW: metadata read with exiftool (subprocess)
        meta = read_metadata_exiftool(self.path)

        # GSD / geolocation need GPS plus camera intrinsics; bail out if missing
        required = ("GPSLatitude", "GPSLongitude", "GPSAltitude",
                    "FocalLength", "ImageWidth", "ImageHeight")
        if not all(meta.get(k) is not None for k in required):
            print("No metadata available")
            return

        # Keep the same gps shape used downstream:
        # [LatRef, (deg, min, sec), LonRef, (deg, min, sec), Altitude]
        self.gps.append(meta.get("GPSLatitudeRef", "N"))
        self.gps.append(self._dd_to_dms(meta["GPSLatitude"]))
        self.gps.append(meta.get("GPSLongitudeRef", "E"))
        self.gps.append(self._dd_to_dms(meta["GPSLongitude"]))
        self.gps.append(meta["GPSAltitude"])

        self.focal_length = float(meta["FocalLength"])   # mm
        print(f"Image width: {meta['ImageWidth']}")
        self.image_width = int(meta["ImageWidth"])
        self.image_height = int(meta["ImageHeight"])

        # use DJI Relative Altitude (height above take-off point) for GSD;
        # exiftool returns it as a signed string like "+50.00", so coerce to float
        self.altitude = float(meta.get("RelativeAltitude", self.gps[4]))   # meters
        focal_m = float(self.focal_length) / 1000        # mm → meters
        self.gsd = (self.altitude * self.sensor_width) / (focal_m * self.image_width)
        print(f"focal length: {focal_m}")
        print(f"gsd: {self.gsd} m/pixel")

        # Gimbal (Yaw, Pitch, Roll). exiftool reports degrees (sometimes as
        # signed strings like "+0.00"), so coerce to float. Stored as
        # tenths-of-degrees to match the old maker-note convention
        # (pixel_to_latlon divides gimbal[0] by 10).
        gy, gp, gr = (meta.get("GimbalYawDegree"),
                      meta.get("GimbalPitchDegree"),
                      meta.get("GimbalRollDegree"))
        if None not in (gy, gp, gr):
            self.gimbal = (int(round(float(gy) * 10)),
                           int(round(float(gp) * 10)),
                           int(round(float(gr) * 10)))
        else:
            print("No gimbal")

        # Flight orientation (Yaw, Pitch, Roll), same tenths-of-degrees convention
        fy, fp, fr = (meta.get("FlightYawDegree"),
                      meta.get("FlightPitchDegree"),
                      meta.get("FlightRollDegree"))
        if None not in (fy, fp, fr):
            self.flight_orientation = (int(round(float(fy) * 10)),
                                       int(round(float(fp) * 10)),
                                       int(round(float(fr) * 10)))
        else:
            print("No flight degrees")
    def predict(self):
        image = Image.open(self.path).convert("RGB")
                # Resize while keeping aspect ratio
        self.original_width, self.original_height = image.size
        w, h = image.size
        self.max_size = 1024
        self.scale = self.max_size / max(w, h) if max(w, h) > self.max_size else 1.0
        new_w, new_h = int(w * self.scale), int(h * self.scale)
        image = image.resize((new_w, new_h), resample=Image.BILINEAR)
        transform = T.ToTensor()
        img_tensor = transform(image).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model([img_tensor])[0]

        keep = output['scores'] >= self.score_threshold
        self.tensor_boxes=output['boxes'][keep]
        self.tensor_labels=output['labels'][keep]
        self.tensor_scores=output['scores'][keep]
        self.tensor_masks=output['masks'][keep]
        self.boxes = self.convert_prediction_to_numpy(self.tensor_boxes)
        self.geobox = self.geoboxing(self.boxes)
        self.labels = self.convert_prediction_to_numpy(self.tensor_labels)
        self.scores = self.convert_prediction_to_numpy(self.tensor_scores)
        self.masks = self.convert_to_binary_mask(self.tensor_masks)
        self.areas = self.compute_mask_areas()
        self.geopolygon = self.geopoly(self.convert_binary_mask_to_coordinates(self.masks))
        # print(self.boxes)

    def convert_prediction_to_numpy(self,tensor):
        return tensor.detach().cpu().numpy()
    def convert_binary_mask_to_coordinates(self, binary_mask):
        all_mask_points=[]
        for mask in binary_mask:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue            
            biggest_contour = max(contours, key=cv2.contourArea)
            # Make sure it's 2D and convert to list of [x, y]
            points = biggest_contour.reshape(-1, 2).tolist()
            all_mask_points.append(points)
        return all_mask_points
    def convert_to_binary_mask(self, tensor_masks):
        binary_masks = []
        for mask in tensor_masks:
            mask_np = mask[0].detach().cpu().numpy()
            binary = (mask_np > 0.9).astype(np.uint8)
            binary_resized = cv2.resize(binary, (self.original_width, self.original_height), interpolation=cv2.INTER_NEAREST)
            binary_masks.append(binary_resized)
        return binary_masks
    def geopoly(self, coordinate_masks):
        geo_poly=[]
        for coords in coordinate_masks:
            c = []
            for point in coords:
                cx = point[0]
                cy = point[1]
                latlon_x, latlon_y = self.pixel_to_latlon(cx, cy)
                c.append([latlon_x, latlon_y])
            geo_poly.append(c)
        return geo_poly

    def pixel_to_latlon(self, x, y):

        # Convert GPS center to decimal degrees
        lat_center = self.dms_to_dd(self.gps[1], self.gps[0])
        lon_center = self.dms_to_dd(self.gps[3], self.gps[2])

        img_center_x = self.image_width / 2
        img_center_y = self.image_height / 2
        # Approximate meters per degree
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat_center))

        if self.gimbal is not None:
            g_yaw = self.gimbal[0] / 10.0
            if g_yaw < 0:
                n_yaw = abs(g_yaw)
            else:
                n_yaw = 360 - g_yaw
        else:
            raise ValueError("Gimbal data not available for yaw calculation")

        dx = x - img_center_x
        dy = y - img_center_y

        yaw_rad = math.radians(n_yaw)

        dx_rot = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
        dy_rot = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
        delta_x = dx_rot * self.gsd
        delta_y = -dy_rot * self.gsd   
        new_lat = lat_center + (delta_y / meters_per_deg_lat)
        new_lon = lon_center + (delta_x / meters_per_deg_lon)


        return new_lat, new_lon
    def geoboxing(self, boxes):
        geobox=[]
        for box in boxes:
            b=[]
            x1, y1, x2, y2 = box
            # Upscale box coordinates if image was resized for prediction
            if hasattr(self, 'scale') and self.scale < 1.0:
                x1 = int(x1 / self.scale)
                x2 = int(x2 / self.scale)
                y1 = int(y1 / self.scale)
                y2 = int(y2 / self.scale)

            # Compute box center
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            latlon_x, latlon_y = self.pixel_to_latlon(cx, cy)
            b.append(latlon_x)
            b.append(latlon_y)
            geobox.append(b)
        return geobox
    
    def dms_to_dd(self, dms, ref):
        # Convert IFDRational to float
        degrees = float(dms[0])
        minutes = float(dms[1])
        seconds = float(dms[2])

        dd = degrees + minutes / 60 + seconds / 3600

        if ref in ['S', 'W']:
            dd *= -1

        return dd
    def dd_to_dms(dd):
        """
        Convert decimal degrees to degrees, minutes, seconds.
        
        Args:
            dd (float): Decimal degrees (positive or negative)
        
        Returns:
            tuple: (degrees, minutes, seconds, hemisphere)
        """
        if dd >= 0:
            hemisphere = 'N' if dd <= 90 else 'E'  # latitude vs longitude logic optional
        else:
            hemisphere = 'S' if dd >= -90 else 'W'
        
        dd_abs = abs(dd)
        degrees = int(dd_abs)
        minutes_full = (dd_abs - degrees) * 60
        minutes = int(minutes_full)
        seconds = (minutes_full - minutes) * 60  # keep high precision
        
        return degrees, minutes, seconds, hemisphere
    def compute_mask_areas(self):
        areas = []
        for mask in self.masks:
            pixel_count = np.sum(mask)  # count white pixels (1s)
            area_m2 = pixel_count * (self.gsd ** 2)
            areas.append(area_m2)
        return areas

    def visualise(self):
        # Load original image
        image = cv2.imread(self.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Overlay contours with index
        for idx, (mask, box) in enumerate(zip(self.masks, self.boxes), start=1):
            # Draw contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (255, 0, 0), 8)

            # Scale bounding box coordinates back to original image size
            x1, y1, x2, y2 = box
            if self.scale < 1.0:  # Only scale if prediction image was resized
                x1 = int(x1 / self.scale)
                x2 = int(x2 / self.scale)
                y1 = int(y1 / self.scale)
                y2 = int(y2 / self.scale)

            # Draw index at center of bounding box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.putText(image, str(idx), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)

        # Display only the image
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image)
        ax.axis('off')
        plt.tight_layout()
        return fig

    def export_to_csv(self, folder_name="coords"):
        # Create folder if it doesn't exist
        os.makedirs(folder_name, exist_ok=True)

        # Create filename based on image name
        image_name = os.path.splitext(os.path.basename(self.path))[0]
        csv_path = os.path.join(folder_name, f"{image_name}_yaw_coordinates.csv")

        # Write CSV
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Index", "Latitude", "Longitude"])

            for idx, (lat, lon) in enumerate(self.geobox, start=1):
                writer.writerow([idx, float(lat), float(lon)])

        print(f"CSV saved to: {csv_path}")
import json
def tracker(batch_folder, batch_id, total_images, processed_images, error=None):
    """
    Save batch status into JSON file.
    If `error` is provided, the batch is marked as failed and the message is stored.
    """

    status_file = os.path.join(batch_folder, "status.json")

    progress = int((processed_images / total_images) * 100) if total_images > 0 else 0

    if error is not None:
        status = "error"
    elif processed_images < total_images:
        status = "processing"
    else:
        status = "completed"

    data = {
        "batchId": batch_id,
        "status": status,
        "total": total_images,
        "processed": processed_images,
        "progress": progress,
        "error": str(error) if error is not None else None
    }

    with open(status_file, "w") as f:
        json.dump(data, f, indent=4)

    if error is not None:
        print(f"[TRACKER] ERROR: {error}")
    else:
        print(f"[TRACKER] {processed_images}/{total_images} ({progress}%)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model path relative to this script (project_root/python/model/...)
model_A = os.path.join(os.path.dirname(__file__), "model", "checkpoint_epoch_20_nogamma.pth")
checkpoint = torch.load(model_A, map_location=device)
model=get_model_mask(2,True)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
if __name__ == "__main__":
    batch_id = sys.argv[1] if len(sys.argv) > 1 else None
    if batch_id is None:
        raise ValueError("Missing batchId argument")

    print("Running inference for batch:", batch_id)
    mask_gps = []
    all_detections = []
    # uploads dir relative to this script (project_root/uploads/<batch_id>)
    batch_folder = os.path.join(os.path.dirname(__file__), "..", "uploads", batch_id)
    batch_folder = os.path.normpath(batch_folder)
    import glob
    images = glob.glob(batch_folder + "/*")
    visual_folder = os.path.join(batch_folder, "visuals")
    os.makedirs(visual_folder, exist_ok=True)

    # dump exiftool metadata for every image in this batch into one csv file
    image_files = [
        p for p in images
        if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png", ".webp")
    ]
    metadata_path = write_metadata_csv(
        image_files,
        os.path.join(batch_folder, "metadata.csv")
    )
    print(f"Metadata saved to: {metadata_path}")

    total_images = len(images)
    processed_images = 0
    tracker(batch_folder, batch_id, total_images, processed_images)
    try:
        for img_path in images:
            print("Processing:", img_path)
            det = Image_detection(img_path, model, device, 0.6)
            all_detections.append(det)
            mask_gps.append(det.geopolygon)
            fig = det.visualise()
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            fig.savefig(
                    os.path.join(visual_folder, f"{base_name}.png"),
                    bbox_inches="tight",
                    pad_inches=0
            )
            plt.close(fig)
            processed_images += 1
            tracker(
                batch_folder,
                batch_id,
                total_images,
                processed_images
            )
        kml_file_path = export_mask_gps_to_kml(
                mask_gps,
                output_folder=batch_folder,
                output_filename="coverage.kml"
        )

        all_csv_path = export_all_geobox_to_csv(
                all_detections,
                output_file="data.csv",
                folder_name=batch_folder
        )
    except Exception:
        import traceback
        traceback.print_exc()
        tracker(
            batch_folder,
            batch_id,
            total_images,
            processed_images,
            error=traceback.format_exc()
        )
        sys.exit(1)
