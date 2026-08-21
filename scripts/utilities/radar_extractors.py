"""Earth Engine radar feature extractors with a NISAR-ready interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class RadarExtractorSpec:
    dataset: str
    scale_m: int
    year_match: str = "exact"
    extraction_method: str = "central_pixel"


class RadarFeatureExtractor(ABC):
    spec: RadarExtractorSpec

    @abstractmethod
    def image_for_year(self, year: int):
        """Return an Earth Engine image in the frozen predictor schema."""


class Sentinel1FeatureExtractor(RadarFeatureExtractor):
    spec = RadarExtractorSpec("COPERNICUS/S1_GRD", 10)

    def image_for_year(self, year: int):
        import ee
        bands = ["VV", "VH"]
        image = (ee.ImageCollection(self.spec.dataset)
                 .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
                 .filter(ee.Filter.eq("instrumentMode", "IW"))
                 .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                 .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                 .select(bands).median())
        return image.addBands(image.select("VV").subtract(image.select("VH"))
                              .rename("VV_minus_VH")).toFloat()


class Palsar2FeatureExtractor(RadarFeatureExtractor):
    spec = RadarExtractorSpec("JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH", 25)
    raw_bands = ["HH", "HV", "angle", "epoch", "qa"]

    def raw_image_for_year(self, year: int):
        import ee
        collection = (ee.ImageCollection(self.spec.dataset)
                      .filterDate(f"{year}-01-01", f"{year + 1}-01-01"))
        return collection.mosaic().select(self.raw_bands)

    def image_for_year(self, year: int):
        import ee
        raw = self.raw_image_for_year(year)
        land = raw.select("qa").eq(255)
        hh = raw.select("HH").updateMask(raw.select("HH").gt(0)).toDouble()
        hv = raw.select("HV").updateMask(raw.select("HV").gt(0)).toDouble()
        hh_db = hh.log10().multiply(20).subtract(83).rename("palsar_hh_db")
        hv_db = hv.log10().multiply(20).subtract(83).rename("palsar_hv_db")
        difference = hh_db.subtract(hv_db).rename("palsar_hh_minus_hv_db")
        hh_power, hv_power = hh.pow(2), hv.pow(2)
        denominator = hh_power.add(hv_power)
        rfdi = hh_power.subtract(hv_power).divide(denominator).rename("palsar_rfdi")
        primary = ee.Image.cat([hh_db, hv_db, difference, rfdi]).updateMask(
            land.And(denominator.gt(0)))
        metadata = raw.select(["angle", "epoch", "qa"],
                              ["palsar_angle", "palsar_epoch", "palsar_qa"])
        # Keep epoch in its native integer precision; casting millisecond timestamps
        # to float32 would corrupt acquisition dates.
        return primary.toFloat().addBands(metadata)


class FutureNisarLFeatureExtractor(RadarFeatureExtractor):
    """Interface placeholder only; historical 2019-2024 NISAR data do not exist."""
    spec = RadarExtractorSpec("UNIMPLEMENTED_2026_PROSPECTIVE_NISAR_L", 25)

    def image_for_year(self, year: int):
        raise NotImplementedError("NISAR is prospective and excluded from this benchmark")
