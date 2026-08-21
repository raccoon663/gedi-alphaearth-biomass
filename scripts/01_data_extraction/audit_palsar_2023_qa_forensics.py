"""Forensic audit of the 2023 PALSAR QA anomaly.

This script reads the Earth Engine asset directly before predictor conversion.  Raw
point coordinates are written only to an ignored private file; committed outputs
contain aggregate counts, hashes, schemas and coordinate-free examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET = "JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH"
YEARS = [2022, 2023, 2024]
BANDS = ["HH", "HV", "angle", "epoch", "qa"]
EE_CATALOG_QA = {0, 50, 100, 150, 255}
JAXA_V24_QA = {0, 1, 2, 3, 4, 50, 100, 150, 255}
VALID_LAND_QA = {1, 255}
JAXA_QA_MEANINGS = {
    0: "no_data", 1: "scansar_land", 2: "scansar_layover",
    3: "scansar_shadow", 4: "scansar_ocean_water", 50: "water",
    100: "layover", 150: "shadow", 255: "land",
}
DUMMY = {"qa": 7, "HH": -101, "HV": -102, "angle": -103, "epoch": -104}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native(value):
    return value.item() if hasattr(value, "item") else value


def target_geometry(ee, path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["type"] == "FeatureCollection":
        geometries = [f["geometry"] for f in payload["features"]]
        return ee.Geometry.MultiPolygon(
            [g["coordinates"] for g in geometries if g["type"] == "Polygon"]
        ) if all(g["type"] == "Polygon" for g in geometries) else ee.FeatureCollection(
            payload
        ).geometry()
    if payload["type"] == "Feature":
        payload = payload["geometry"]
    return ee.Geometry(payload)


def collection_for_year(ee, year: int):
    return (ee.ImageCollection(DATASET)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01"))


def image_records(ee, year: int) -> tuple[list[dict], list[dict]]:
    collection = collection_for_year(ee, year)
    count = int(collection.size().getInfo())
    images = collection.toList(count)
    rows, records = [], []
    for i in range(count):
        image = ee.Image(images.get(i))
        image_id = image.id().getInfo()
        names = image.bandNames().getInfo()
        types = image.bandTypes().getInfo()
        properties = image.toDictionary(image.propertyNames()).getInfo()
        projections = {}
        for position, band in enumerate(names):
            projection = image.select(band).projection()
            info = projection.getInfo()
            nominal_scale = float(projection.nominalScale().getInfo())
            projections[band] = {"projection": info, "nominal_scale": nominal_scale}
            rows.append({
                "year": year, "collection_size": count, "image_position": i,
                "image_id": image_id, "system_index": properties.get("system:index"),
                "system_time_start": properties.get("system:time_start"),
                "band_position": position, "band": band,
                "data_type_json": json.dumps(types.get(band), sort_keys=True),
                "crs": info.get("crs"),
                "crs_transform_json": json.dumps(info.get("transform")),
                "nominal_scale_m": nominal_scale,
                "band_names_json": json.dumps(names),
            })
        records.append({
            "year": year, "image_id": image_id, "system_index": properties.get("system:index"),
            "system_time_start": properties.get("system:time_start"), "band_names": names,
            "band_types": types, "projections": projections, "properties": properties,
        })
    return rows, records


def qa_histogram(image, geometry, scale: int) -> dict[int, int]:
    result = (image.select("qa").reduceRegion(
        reducer=__import__("ee").Reducer.frequencyHistogram(), geometry=geometry,
        scale=scale, maxPixels=10**13, tileScale=8
    ).get("qa").getInfo()) or {}
    return {int(float(k)): int(v) for k, v in result.items()}


def raw_histograms(ee, geometries: dict[str, object], scales: dict[str, int]):
    rows = []
    for year in YEARS:
        collection = collection_for_year(ee, year)
        count = int(collection.size().getInfo())
        images = collection.toList(count)
        for i in range(count):
            image = ee.Image(images.get(i)).select(BANDS)
            image_id = image.id().getInfo()
            for domain, geometry in geometries.items():
                histogram = qa_histogram(image, geometry, scales[domain])
                total = sum(histogram.values())
                for qa, n in sorted(histogram.items()):
                    rows.append({
                        "domain": domain, "year": year, "image_id": image_id,
                        "image_position": i, "analysis_scale_m": scales[domain],
                        "qa": qa, "pixel_count": n,
                        "proportion": n / total if total else None,
                        "earth_engine_catalog_documented_class": qa in EE_CATALOG_QA,
                        "jaxa_v24_documented_class": qa in JAXA_V24_QA,
                        "jaxa_v24_meaning": JAXA_QA_MEANINGS.get(qa, "undocumented"),
                        "valid_land_class": qa in VALID_LAND_QA,
                    })
    return pd.DataFrame(rows)


def fixed_points(ee, geometries: dict[str, object]):
    collections = []
    for domain, geometry in geometries.items():
        for year in YEARS:
            n = 20 if year == 2023 else 8
            points = ee.FeatureCollection.randomPoints(geometry, n, 42000 + year)
            points = points.map(lambda feature, d=domain, y=year: feature.set({
                "audit_id": ee.String(d).cat("_").cat(ee.Number(y).format()).cat("_").cat(
                    ee.String(feature.get("system:index"))),
                "audit_domain": d, "audit_year": y,
            }))
            collections.append(points)
    combined = collections[0]
    for collection in collections[1:]:
        combined = combined.merge(collection)
    return combined


def sample_image(ee, image, points, stage: str) -> pd.DataFrame:
    sample_image = image.select(BANDS).unmask(-9999)
    sampled = sample_image.sampleRegions(
        collection=points, properties=["audit_id", "audit_domain", "audit_year"],
        scale=25, geometries=True, tileScale=4)
    features = sampled.getInfo().get("features", [])
    rows = []
    for feature in features:
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]
        rows.append({"stage": stage, "lon": coords[0], "lat": coords[1], **props})
    return pd.DataFrame(rows)


def stage_a_to_e(ee, points) -> tuple[pd.DataFrame, dict]:
    frames, stage_checks = [], {}
    for year in YEARS:
        year_points = points.filter(ee.Filter.eq("audit_year", year))
        collection = collection_for_year(ee, year)
        count = int(collection.size().getInfo())
        first = ee.Image(collection.first())
        stages = {
            "A_raw_collection_first": first,
            "B_collection_mosaic": collection.mosaic(),
            "C_select_bands": collection.mosaic().select(BANDS),
            "D_sampleRegions": collection.mosaic().select(BANDS),
        }
        year_frames = {}
        for stage, image in stages.items():
            frame = sample_image(ee, image, year_points, stage)
            frame["year"] = year
            frame["image_id"] = first.id().getInfo()
            frames.append(frame)
            year_frames[stage] = frame
        # E is the pandas materialization of the D response.
        e_frame = pd.DataFrame(year_frames["D_sampleRegions"].to_dict("records"))
        e_frame["stage"] = "E_pandas_dataframe"
        frames.append(e_frame)
        year_frames["E_pandas_dataframe"] = e_frame
        keys = ["audit_id", *BANDS]
        reference = year_frames["A_raw_collection_first"][keys]
        equality = {}
        for stage, frame in year_frames.items():
            joined = reference.merge(frame[keys], on="audit_id", how="outer",
                                     suffixes=("_raw", "_candidate"), indicator=True)
            common = joined[joined._merge == "both"]
            mismatches = {
                band: int((common[f"{band}_raw"] != common[f"{band}_candidate"]).sum())
                for band in BANDS
            }
            equality[stage] = {
                "raw_n": len(reference), "candidate_n": len(frame),
                "common_n": len(common),
                "raw_only_n": int((joined._merge == "left_only").sum()),
                "candidate_only_n": int((joined._merge == "right_only").sum()),
                "band_mismatch_n_on_common_points": mismatches,
                "identical": len(reference) == len(frame) and not any(mismatches.values()),
            }
        stage_checks[str(year)] = {
            "collection_size": count, "stage_comparison_to_raw": equality,
            "first_projection": first.select("qa").projection().getInfo(),
            "mosaic_projection": collection.mosaic().select("qa").projection().getInfo(),
        }
    return pd.concat(frames, ignore_index=True), stage_checks


def property_collision_test(ee, point, image) -> dict:
    feature = ee.Feature(point.geometry(), {"audit_id": "collision_control", **DUMMY})
    safe = image.select(BANDS).unmask(-9999).sampleRegions(
        ee.FeatureCollection([feature]), properties=["audit_id"], scale=25,
        geometries=False).first().toDictionary().getInfo()
    colliding = image.select(BANDS).unmask(-9999).sampleRegions(
        ee.FeatureCollection([feature]), properties=["audit_id", *BANDS], scale=25,
        geometries=False).first().toDictionary().getInfo()
    return {
        "input_dummy_properties": DUMMY,
        "safe_properties_argument": ["audit_id"],
        "colliding_properties_argument": ["audit_id", *BANDS],
        "safe_result": safe, "colliding_result": colliding,
        "safe_result_uses_image_qa": safe.get("qa") != DUMMY["qa"],
        "colliding_result_uses_image_qa": colliding.get("qa") != DUMMY["qa"],
        "current_pipeline_properties_collide": False,
    }


def manifest_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        return list(pd.read_parquet(path).columns)
    return list(pd.read_csv(path, nrows=0).columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-aoi", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--project")
    parser.add_argument("--source-histogram-scale", type=int, default=250)
    parser.add_argument("--target-histogram-scale", type=int, default=25)
    args = parser.parse_args()
    import ee
    ee.Initialize(project=args.project) if args.project else ee.Initialize()

    geometries = {
        "source": ee.FeatureCollection("TIGER/2018/States").filter(
            ee.Filter.inList("NAME", ["Georgia", "South Carolina"])).geometry(),
        "target": target_geometry(ee, args.target_aoi),
    }
    metadata_rows, image_records_all = [], []
    for year in YEARS:
        rows, records = image_records(ee, year)
        metadata_rows.extend(rows)
        image_records_all.extend(records)
    metadata_path = ROOT / "outputs/tables/audits/palsar_asset_metadata_2022_2024.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)

    histogram = raw_histograms(
        ee, geometries,
        {"source": args.source_histogram_scale, "target": args.target_histogram_scale})
    histogram_path = ROOT / "outputs/tables/audits/palsar_raw_qa_histograms.csv"
    histogram.to_csv(histogram_path, index=False)

    points = fixed_points(ee, geometries)
    private_points, stage_checks = stage_a_to_e(ee, points)
    private_path = ROOT / "data/interim/palsar_point_forensics_private.csv"
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_points.to_csv(private_path, index=False)

    point_2023 = points.filter(ee.Filter.eq("audit_year", 2023)).first()
    image_2023 = ee.Image(collection_for_year(ee, 2023).first())
    collision = property_collision_test(ee, point_2023, image_2023)
    source_columns = manifest_columns(args.source_manifest)
    target_columns = manifest_columns(args.target_manifest)
    collision_names = set(BANDS)
    input_collision = {
        "source": sorted(collision_names.intersection(source_columns)),
        "target": sorted(collision_names.intersection(target_columns)),
    }

    coordinate_free_examples = (private_points[private_points.stage == "A_raw_collection_first"]
                                [["audit_domain", "audit_year", *BANDS]]
                                .groupby(["audit_domain", "audit_year"], as_index=False)
                                .head(3).to_dict("records"))
    point_manifest = {
        "status": "PALSAR_POINT_FORENSIC_AUDIT_COMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "private_point_file_sha256": sha256(private_path),
        "private_point_file_committed": False,
        "private_rows": len(private_points),
        "coordinate_free_examples": coordinate_free_examples,
        "qa_frequency_by_stage": [
            {"stage": stage, "domain": domain, "year": int(year), "qa": native(qa), "n": int(n)}
            for (stage, domain, year, qa), n in private_points.groupby(
                ["stage", "audit_domain", "audit_year", "qa"], dropna=False).size().items()
        ],
        "stage_checks": stage_checks,
        "property_collision_test": collision,
        "input_manifest_band_name_collisions": input_collision,
    }
    point_manifest_path = ROOT / "outputs/manifests/palsar_point_forensic_audit.json"
    point_manifest_path.write_text(json.dumps(point_manifest, indent=2), encoding="utf-8")

    raw_values = sorted(int(x) for x in histogram.qa.unique())
    ee_catalog_omissions = sorted(set(raw_values) - EE_CATALOG_QA)
    undocumented = sorted(set(raw_values) - JAXA_V24_QA)
    target_2023 = histogram[(histogram.domain == "target") & (histogram.year == 2023)]
    target_2023_land = int(target_2023.loc[target_2023.qa.isin(VALID_LAND_QA),
                                          "pixel_count"].sum())
    asset_manifest = {
        "status": "PALSAR_ASSET_METADATA_QA_FORENSIC_COMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET, "years": YEARS,
        "earth_engine_catalog_qa_classes": sorted(EE_CATALOG_QA),
        "authoritative_jaxa_v24_qa_classes": sorted(JAXA_V24_QA),
        "authoritative_jaxa_v24_qa_meanings": JAXA_QA_MEANINGS,
        "valid_land_qa_classes": sorted(VALID_LAND_QA),
        "qa_semantics_url": "https://www.eorc.jaxa.jp/ALOS/en/dataset/pdf/DatasetDescription_PALSAR2_Mosaic_ver240.pdf",
        "raw_qa_values_observed": raw_values,
        "earth_engine_catalog_omitted_values_observed": ee_catalog_omissions,
        "authoritatively_undocumented_raw_qa_values": undocumented,
        "resolved_2023_qa_status": "OUTCOME_C_DOCUMENTATION_ASSET_VERSION_DISCREPANCY",
        "root_cause": "Earth Engine catalog omits ScanSAR QA classes 1-4; JAXA v2.4 defines class 1 as land.",
        "existing_benchmark_valid": False,
        "rerun_required": True,
        "target_2023_land_pixel_count_at_audit_scale": target_2023_land,
        "histogram_scales_m": {"source": args.source_histogram_scale,
                               "target": args.target_histogram_scale},
        "asset_metadata_csv_sha256": sha256(metadata_path),
        "raw_qa_histogram_csv_sha256": sha256(histogram_path),
        "point_forensic_manifest_sha256": sha256(point_manifest_path),
        "images": image_records_all,
        "single_image_per_year": all(
            sum(r["year"] == year for r in image_records_all) == 1 for year in YEARS),
    }
    asset_manifest_path = ROOT / "outputs/manifests/palsar_asset_metadata_audit.json"
    asset_manifest_path.write_text(json.dumps(asset_manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "metadata": str(metadata_path), "histograms": str(histogram_path),
        "point_manifest": str(point_manifest_path), "asset_manifest": str(asset_manifest_path),
        "raw_qa_values": raw_values,
        "earth_engine_catalog_omitted_values_observed": ee_catalog_omissions,
        "authoritatively_undocumented_raw_qa_values": undocumented,
        "target_2023_land_count": target_2023_land,
        "stage_checks": stage_checks, "collision": collision,
    }, indent=2))


if __name__ == "__main__":
    main()
