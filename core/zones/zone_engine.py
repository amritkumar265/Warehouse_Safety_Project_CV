import cv2
import numpy as np

class ZoneEngine:

    """
    Polygon base spatial association engine
    """

    def __init__(self , camera_profile):

        self.camera_id = camera_profile["camera_id"]
        self.zones = camera_profile.get("zones" , {})

    def get_pixel_polygons(self, frame_shape):
        height , width = frame_shape[:2]

        pixel_zones ={}

        for zone_key , zone_data in self.zones.items():
            polygons = []

            for polygon in zone_data.get("polygons" , []):
                points = np.array(
                    [
                        [
                            round(x * (width - 1)),
                            round(y * (height - 1))
                        ]
                        for x , y in polygon
                    ],
                    dtype = np.int32
                )

                if len(points) >= 3:
                    polygons.append(points)

            pixel_zones[zone_key] = polygons

        return pixel_zones

    @staticmethod
    def get_bottom_center(bbox):
        x1 , y1 , x2 , y2 = bbox

        return (
            (x1 + x2) / 2,
            y2
        )

    @staticmethod
    def point_in_zone(point , polygons):
        x , y = point

        for polygon in polygons:

            result = cv2.pointPolygonTest(
                polygon,
                (float(x) , float(y)),
                False
            )

            if result >= 0:
                return True

        return False

    @staticmethod
    def calculate_zone_coverage(bbox , polygons , frame_shape):

        height , width = frame_shape[:2]

        x1 , y1 , x2 , y2 = bbox

        x1 = max(0 , min(width , int(np.floor(x1))))
        y1 = max(0, min(height, int(np.floor(y1))))
        x2 = max(0, min(width, int(np.ceil(x2))))
        y2 = max(0, min(height, int(np.ceil(y2))))

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            return 0.0

        bbox_area = box_width * box_height

        zone_mask = np.zeros(
            (box_height , box_width),
            dtype = np.uint8
        )

        for polygon in polygons:
            local_polygon = polygon.copy()

            local_polygon[ :  , 0] -= x1
            local_polygon[:, 1] -= y1

            cv2.fillPoly(
                zone_mask,
                [local_polygon],
                1
            )

        intersection_area = cv2.countNonZero(zone_mask)

        return intersection_area / bbox_area

    def associate_detection(self , bbox , class_name , frame_shape):

        pixel_zones = self.get_pixel_polygons(frame_shape)
        bottom_center = self.get_bottom_center(bbox)

        zone_results = {}

        containing_zones = []

        for zone_key , polygons in pixel_zones.items():

            is_inside = self.point_in_zone(
                bottom_center,
                polygons
            )

            coverage = self.calculate_zone_coverage(
                bbox,
                polygons,
                frame_shape
            )

            zone_results[zone_key] = {
                "contains_bottom_center": is_inside,
                "bbox_coverage": round(coverage, 4)
            }

            if is_inside:
                containing_zones.append(zone_key)

        primary_zone = (
            containing_zones[0]
            if containing_zones
            else None
        )

        return {
            "camera_id" : self.camera_id,
            "class_name" : class_name,
            "bbox" : [float(value) for value in bbox],
            "bottom_center" : [
                round(bottom_center[0] , 2),
                round(bottom_center[1] , 2)
            ],
            "primary_zone" : primary_zone,
            "containing_zones" : containing_zones,
            "zone_results" : zone_results
        }    
    