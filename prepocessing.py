from __future__ import annotations

from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


SEX_MAP = {"0": "Unknown", "1": "Male", "2": "Female"}
REPORT_TYPE_MAP = {"1": "Spontaneous", "2": "Study", "3": "Other", "4": "Unknown"}
QUALIFICATION_MAP = {
	"1": "Physician",
	"2": "Pharmacist",
	"3": "Other HP",
	"4": "Lawyer",
	"5": "Consumer",
}
AGE_GROUP_MAP = {
	"1": "Neonate",
	"2": "Infant",
	"3": "Child",
	"4": "Adolescent",
	"5": "Adult",
	"6": "Elderly",
}
DRUG_ROLE_MAP = {
	"1": "Suspect",
	"2": "Concomitant",
	"3": "Interacting",
	"4": "Not Administered",
}
OUTCOME_MAP = {
	"1": "Recovered",
	"2": "Recovering",
	"3": "Not Recovered",
	"4": "Recovered w/ Sequelae",
	"5": "Fatal",
	"6": "Unknown",
}
ACTION_MAP = {
	"1": "Withdrawn",
	"2": "Dose Reduced",
	"3": "Dose Increased",
	"4": "Dose Not Changed",
	"5": "Unknown",
	"6": "Not Applicable",
}
ROUTE_MAP = {
	"048": "Oral",
	"061": "Topical",
	"065": "Subcutaneous",
	"042": "Intramuscular",
	"040": "Intravenous",
	"064": "Transdermal",
	"054": "Nasal",
	"059": "Rectal",
	"066": "Sublingual",
	"041": "Intra-articular",
	"058": "Percutaneous",
	"050": "Ophthalmic",
	"062": "Inhalation",
	"070": "Vaginal",
	"067": "Unknown",
}


def normalize_code(value: object) -> str:
	if pd.isna(value):
		return ""
	text = str(value).strip().replace(",", ".")
	if text.endswith(".0"):
		text = text[:-2]
	return text


def map_column(df: pd.DataFrame, source_col: str, target_col: str, mapping: dict[str, str], default: str = "Unknown") -> None:
	if source_col not in df.columns:
		return
	normalized = df[source_col].map(normalize_code)
	df[target_col] = normalized.map(mapping).fillna(default)


def standardize_age_years(age: float, age_unit_code: object) -> float:
	if pd.isna(age):
		return np.nan

	unit = normalize_code(age_unit_code)
	conversions = {
		"800": age * 10,
		"801": age,
		"802": age / 12.0,
		"803": age / 52.18,
		"804": age / 365.25,
		"805": age / 8766.0,
	}
	return conversions.get(unit, age)


def clean_demo(demo: pd.DataFrame, drug: pd.DataFrame, reac: pd.DataFrame) -> pd.DataFrame:
	demo = demo.copy()

	map_column(demo, "patientsex", "sex_label", SEX_MAP)
	map_column(demo, "reporttype", "reporttype_label", REPORT_TYPE_MAP, default="Missing")
	map_column(demo, "qualification", "qualification_label", QUALIFICATION_MAP, default="Missing")
	map_column(demo, "patientagegroup", "agegrp_label", AGE_GROUP_MAP)

	if "patientonsetage" in demo.columns:
		demo["patientonsetage"] = pd.to_numeric(demo["patientonsetage"], errors="coerce")
		unit_series = demo["patientonsetageunit"] if "patientonsetageunit" in demo.columns else ""
		demo["age_years"] = [
			standardize_age_years(age, unit)
			for age, unit in zip(demo["patientonsetage"], unit_series)
		]
		demo.loc[(demo["age_years"] < 0) | (demo["age_years"] > 120), "age_years"] = np.nan

	if "patientweight" in demo.columns:
		demo["patientweight"] = pd.to_numeric(demo["patientweight"], errors="coerce")
		demo.loc[(demo["patientweight"] < 3) | (demo["patientweight"] > 300), "patientweight"] = np.nan

	if "age_years" in demo.columns:
		if "agegrp_label" in demo.columns:
			demo["age_years"] = demo.groupby("agegrp_label")["age_years"].transform(lambda s: s.fillna(s.median()))
		demo["age_years"] = demo["age_years"].fillna(demo["age_years"].median())

	if "patientweight" in demo.columns:
		group_cols = [col for col in ["sex_label", "agegrp_label"] if col in demo.columns]
		if group_cols:
			demo["patientweight"] = demo.groupby(group_cols)["patientweight"].transform(lambda s: s.fillna(s.median()))
		demo["patientweight"] = demo["patientweight"].fillna(demo["patientweight"].median())

	seriousness_base = [
		"seriousnessdeath",
		"seriousnesslifethreatening",
		"seriousnesshospitalization",
		"seriousnessdisabling",
		"seriousnesscongenitalanomali",
		"seriousnessother",
	]
	seriousness_flags: list[str] = []
	for col in seriousness_base:
		if col in demo.columns:
			flag_col = f"{col}_flag"
			demo[flag_col] = demo[col].map(normalize_code).eq("1").astype(int)
			seriousness_flags.append(flag_col)

	if seriousness_flags:
		demo["seriousness_score"] = demo[seriousness_flags].sum(axis=1)
	else:
		demo["seriousness_score"] = 0

	if "serious" in demo.columns:
		demo["is_serious"] = demo["serious"].map(normalize_code).eq("1").astype(int)
	else:
		demo["is_serious"] = (demo["seriousness_score"] > 0).astype(int)

	if "seriousnessdeath_flag" in demo.columns:
		demo["is_fatal"] = demo["seriousnessdeath_flag"]
	elif "seriousnessdeath" in demo.columns:
		demo["is_fatal"] = demo["seriousnessdeath"].map(normalize_code).eq("1").astype(int)
	else:
		demo["is_fatal"] = 0

	if "safetyreportid" in demo.columns:
		if "safetyreportid" in drug.columns:
			drug_counts = drug.groupby("safetyreportid").size().rename("num_drugs")
			demo = demo.merge(drug_counts, on="safetyreportid", how="left")
		else:
			demo["num_drugs"] = 0

		if "safetyreportid" in reac.columns:
			reac_counts = reac.groupby("safetyreportid").size().rename("num_reactions")
			demo = demo.merge(reac_counts, on="safetyreportid", how="left")
		else:
			demo["num_reactions"] = 0

	for col in ["num_drugs", "num_reactions"]:
		if col not in demo.columns:
			demo[col] = 0
		demo[col] = demo[col].fillna(0).astype(int)

	if "receivedate" in demo.columns:
		demo["receive_dt"] = pd.to_datetime(demo["receivedate"].astype(str), format="%Y%m%d", errors="coerce")
		missing_date = demo["receive_dt"].isna()
		if missing_date.any():
			demo.loc[missing_date, "receive_dt"] = pd.to_datetime(demo.loc[missing_date, "receivedate"], errors="coerce")
		demo["report_month"] = demo["receive_dt"].dt.to_period("M").astype(str)
		demo["report_dayofweek"] = demo["receive_dt"].dt.day_name()

	demo["serious_label"] = demo["is_serious"].map({0: "Non-Serious", 1: "Serious"})
	demo["fatal_label"] = demo["is_fatal"].map({0: "Non-Fatal", 1: "Fatal"})
	return demo


def clean_drug(drug: pd.DataFrame) -> pd.DataFrame:
	drug = drug.copy()
	map_column(drug, "drugcharacterization", "role_label", DRUG_ROLE_MAP)
	map_column(drug, "actiondrug", "action_label", ACTION_MAP, default="Missing")

	if "drugadministrationroute" in drug.columns:
		route_codes = drug["drugadministrationroute"].map(normalize_code)
		drug["route_label"] = route_codes.map(ROUTE_MAP).fillna(route_codes.replace("", "Unknown"))

	numeric_cols = [
		"drugstructuredosagenumb",
		"drugseparatedosagenumb",
		"drugintervaldosageunitnumb",
		"drugcumulativedosagenumb",
		"drugtreatmentduration",
	]
	for col in numeric_cols:
		if col in drug.columns:
			drug[col] = pd.to_numeric(drug[col], errors="coerce")
	return drug


def clean_reac(reac: pd.DataFrame) -> pd.DataFrame:
	reac = reac.copy()
	map_column(reac, "reactionoutcome", "outcome_label", OUTCOME_MAP, default="Missing")
	return reac


def first_existing(paths: list[Path]) -> Path | None:
	for path in paths:
		if path.exists() and path.is_file():
			return path
	return None


def resolve_input_paths(input_dir: Path) -> tuple[Path, Path, Path]:
	demo = first_existing([input_dir / "demo.csv", input_dir / "demo_cleaned.csv"])
	drug = first_existing([input_dir / "drug.csv", input_dir / "drug_cleaned.csv"])
	reac = first_existing([input_dir / "reac.csv", input_dir / "reaction.csv", input_dir / "reac_cleaned.csv"])

	if demo and drug and reac:
		return demo, drug, reac

	csvs = sorted(input_dir.glob("*.csv"))
	if len(csvs) < 3:
		raise FileNotFoundError(
			f"Could not find required FAERS CSV files in {input_dir}. "
			"Expected demo/drug/reac style files."
		)
	raise FileNotFoundError(
		f"Found CSV files in {input_dir} but could not map them to demo/drug/reac: "
		f"{', '.join(p.name for p in csvs)}"
	)


def get_text(parent: ET.Element | None, tag: str, default: str = "") -> str:
	if parent is None:
		return default
	value = parent.findtext(tag)
	if value is None:
		return default
	return value.strip()


def parse_faers_xml(xml_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	demo_rows: list[dict[str, object]] = []
	drug_rows: list[dict[str, object]] = []
	reac_rows: list[dict[str, object]] = []

	for _, report in ET.iterparse(xml_path, events=("end",)):
		if report.tag != "safetyreport":
			continue

		safetyreportid = get_text(report, "safetyreportid")
		primarysource = report.find("primarysource")
		patient = report.find("patient")

		demo_rows.append(
			{
				"safetyreportid": safetyreportid,
				"primarysourcecountry": get_text(report, "primarysourcecountry"),
				"occurcountry": get_text(report, "occurcountry"),
				"reporttype": get_text(report, "reporttype"),
				"serious": get_text(report, "serious"),
				"seriousnessdeath": get_text(report, "seriousnessdeath"),
				"seriousnesslifethreatening": get_text(report, "seriousnesslifethreatening"),
				"seriousnesshospitalization": get_text(report, "seriousnesshospitalization"),
				"seriousnessdisabling": get_text(report, "seriousnessdisabling"),
				"seriousnesscongenitalanomali": get_text(report, "seriousnesscongenitalanomali"),
				"seriousnessother": get_text(report, "seriousnessother"),
				"receivedate": get_text(report, "receivedate"),
				"receiptdate": get_text(report, "receiptdate"),
				"fulfillexpeditecriteria": get_text(report, "fulfillexpeditecriteria"),
				"qualification": get_text(primarysource, "qualification"),
				"patientsex": get_text(patient, "patientsex"),
				"patientonsetage": get_text(patient, "patientonsetage"),
				"patientonsetageunit": get_text(patient, "patientonsetageunit"),
				"patientweight": get_text(patient, "patientweight"),
				"patientagegroup": get_text(patient, "patientagegroup"),
			}
		)

		if patient is not None:
			for drug in patient.findall("drug"):
				activesubstance = drug.find("activesubstance")
				drug_rows.append(
					{
						"safetyreportid": safetyreportid,
						"drugcharacterization": get_text(drug, "drugcharacterization"),
						"medicinalproduct": get_text(drug, "medicinalproduct"),
						"drugauthorizationnumb": get_text(drug, "drugauthorizationnumb"),
						"drugdosagetext": get_text(drug, "drugdosagetext"),
						"drugdosageform": get_text(drug, "drugdosageform"),
						"drugadministrationroute": get_text(drug, "drugadministrationroute"),
						"drugindication": get_text(drug, "drugindication"),
						"drugstartdate": get_text(drug, "drugstartdate"),
						"drugenddate": get_text(drug, "drugenddate"),
						"drugtreatmentduration": get_text(drug, "drugtreatmentduration"),
						"drugtreatmentdurationunit": get_text(drug, "drugtreatmentdurationunit"),
						"actiondrug": get_text(drug, "actiondrug"),
						"drugadditional": get_text(drug, "drugadditional"),
						"activesubstancename": get_text(activesubstance, "activesubstancename"),
					}
				)

			for reaction in patient.findall("reaction"):
				reac_rows.append(
					{
						"safetyreportid": safetyreportid,
						"reactionmeddraversionpt": get_text(reaction, "reactionmeddraversionpt"),
						"reactionmeddrapt": get_text(reaction, "reactionmeddrapt"),
						"reactionoutcome": get_text(reaction, "reactionoutcome"),
					}
				)

		report.clear()

	demo = pd.DataFrame(demo_rows)
	drug = pd.DataFrame(
		drug_rows,
		columns=[
			"safetyreportid",
			"drugcharacterization",
			"medicinalproduct",
			"drugauthorizationnumb",
			"drugdosagetext",
			"drugdosageform",
			"drugadministrationroute",
			"drugindication",
			"drugstartdate",
			"drugenddate",
			"drugtreatmentduration",
			"drugtreatmentdurationunit",
			"actiondrug",
			"drugadditional",
			"activesubstancename",
		],
	)
	reac = pd.DataFrame(
		reac_rows,
		columns=["safetyreportid", "reactionmeddraversionpt", "reactionmeddrapt", "reactionoutcome"],
	)
	return demo, drug, reac


def parse_multiple_faers_xml(xml_files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	if not xml_files:
		raise ValueError("No XML files provided.")

	demo_frames: list[pd.DataFrame] = []
	drug_frames: list[pd.DataFrame] = []
	reac_frames: list[pd.DataFrame] = []

	for xml_path in xml_files:
		demo_df, drug_df, reac_df = parse_faers_xml(xml_path)
		demo_frames.append(demo_df)
		drug_frames.append(drug_df)
		reac_frames.append(reac_df)

	demo = pd.concat(demo_frames, ignore_index=True) if demo_frames else pd.DataFrame()
	drug = pd.concat(drug_frames, ignore_index=True) if drug_frames else pd.DataFrame()
	reac = pd.concat(reac_frames, ignore_index=True) if reac_frames else pd.DataFrame()
	return demo, drug, reac


def load_raw_tables(input_dir: Path, xml_files: list[Path] | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
	if xml_files:
		missing = [path for path in xml_files if not path.exists()]
		if missing:
			raise FileNotFoundError(f"Missing XML files: {', '.join(path.as_posix() for path in missing)}")
		demo, drug, reac = parse_multiple_faers_xml(xml_files)
		label = f"XML files ({len(xml_files)}): {', '.join(path.name for path in xml_files)}"
		return demo, drug, reac, label

	try:
		demo_path, drug_path, reac_path = resolve_input_paths(input_dir)
		demo = pd.read_csv(demo_path, low_memory=False)
		drug = pd.read_csv(drug_path, low_memory=False)
		reac = pd.read_csv(reac_path, low_memory=False)
		return demo, drug, reac, f"CSV files: {demo_path.name}, {drug_path.name}, {reac_path.name}"
	except FileNotFoundError as error:
		xml_files = sorted(input_dir.glob("*.xml"))
		if len(xml_files) == 1:
			demo, drug, reac = parse_faers_xml(xml_files[0])
			return demo, drug, reac, f"XML file: {xml_files[0].name}"
		if len(xml_files) > 1:
			demo, drug, reac = parse_multiple_faers_xml(xml_files)
			label = f"XML files ({len(xml_files)}): {', '.join(path.name for path in xml_files)}"
			return demo, drug, reac, label
		raise


def build_dashboard_outputs(demo: pd.DataFrame, drug: pd.DataFrame, reac: pd.DataFrame) -> dict[str, pd.DataFrame]:
	outputs: dict[str, pd.DataFrame] = {}

	report_cols = [
		"safetyreportid",
		"receive_dt",
		"report_month",
		"sex_label",
		"agegrp_label",
		"age_years",
		"patientweight",
		"serious_label",
		"fatal_label",
		"num_drugs",
		"num_reactions",
		"seriousness_score",
	]
	keep_cols = [col for col in report_cols if col in demo.columns]
	outputs["reports_clean"] = demo[keep_cols].copy()

	if "report_month" in demo.columns:
		monthly = (
			demo.dropna(subset=["report_month"])
			.groupby("report_month", as_index=False)
			.agg(
				report_count=("report_month", "size"),
				serious_count=("is_serious", "sum"),
				fatal_count=("is_fatal", "sum"),
			)
			.sort_values("report_month")
		)
		outputs["monthly_reports"] = monthly

	if "medicinalproduct" in drug.columns:
		top_drugs = (
			drug["medicinalproduct"]
			.dropna()
			.astype(str)
			.str.strip()
			.replace("", np.nan)
			.dropna()
			.value_counts()
			.head(25)
			.rename_axis("medicinalproduct")
			.reset_index(name="report_mentions")
		)
		outputs["top_drugs"] = top_drugs

	if "reactionmeddrapt" in reac.columns:
		top_reactions = (
			reac["reactionmeddrapt"]
			.dropna()
			.astype(str)
			.str.strip()
			.replace("", np.nan)
			.dropna()
			.value_counts()
			.head(25)
			.rename_axis("reactionmeddrapt")
			.reset_index(name="count")
		)
		outputs["top_reactions"] = top_reactions

	if "sex_label" in demo.columns:
		outputs["sex_distribution"] = demo.groupby("sex_label", as_index=False).size().rename(columns={"size": "count"})

	if "agegrp_label" in demo.columns:
		outputs["age_group_distribution"] = demo.groupby("agegrp_label", as_index=False).size().rename(columns={"size": "count"})

	return outputs


def write_outputs(demo: pd.DataFrame, drug: pd.DataFrame, reac: pd.DataFrame, output_dir: Path) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	demo.to_csv(output_dir / "demo_cleaned.csv", index=False)
	drug.to_csv(output_dir / "drug_cleaned.csv", index=False)
	reac.to_csv(output_dir / "reac_cleaned.csv", index=False)

	dashboards = build_dashboard_outputs(demo, drug, reac)
	for name, frame in dashboards.items():
		frame.to_csv(output_dir / f"{name}.csv", index=False)

	summary = {
		"n_reports": int(len(demo)),
		"n_drug_rows": int(len(drug)),
		"n_reaction_rows": int(len(reac)),
		"serious_rate": float(demo["is_serious"].mean()) if "is_serious" in demo.columns and len(demo) else 0.0,
		"fatal_rate": float(demo["is_fatal"].mean()) if "is_fatal" in demo.columns and len(demo) else 0.0,
	}
	with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
		json.dump(summary, fp, indent=2)


def run_pipeline(input_dir: Path, output_dir: Path, xml_files: list[Path] | None = None) -> None:
	demo, drug, reac, source_label = load_raw_tables(input_dir, xml_files)

	cleaned_drug = clean_drug(drug)
	cleaned_reac = clean_reac(reac)
	cleaned_demo = clean_demo(demo, cleaned_drug, cleaned_reac)

	write_outputs(cleaned_demo, cleaned_drug, cleaned_reac, output_dir)

	print(f"Input source: {source_label}")
	print(f"Saved cleaned outputs to: {output_dir}")
	print(f"Reports: {len(cleaned_demo):,} | Drug rows: {len(cleaned_drug):,} | Reaction rows: {len(cleaned_reac):,}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Preprocess FAERS XML/CSV files into dashboard-ready datasets.")
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=Path("data"),
		help="Fallback directory to auto-discover CSV/XML files when explicit XML files are not provided.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("data/processed"),
		help="Directory to write cleaned and dashboard-ready outputs.",
	)
	parser.add_argument(
		"--xml-file",
		type=Path,
		default=None,
		help="Optional path to one FAERS XML file (legacy alias for --xml-files).",
	)
	parser.add_argument(
		"--xml-files",
		type=Path,
		nargs="+",
		default=None,
		help="Optional list of FAERS XML files to combine (recommended for quarterly 3-file loads).",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	xml_files: list[Path] | None = None
	if args.xml_files:
		xml_files = args.xml_files
	elif args.xml_file:
		xml_files = [args.xml_file]
	run_pipeline(args.input_dir, args.output_dir, xml_files)
