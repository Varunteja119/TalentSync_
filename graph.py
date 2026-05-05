import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
TEST_RESULTS_PATH = ROOT / "test_results.json"
SENSITIVITY_PATH = ROOT / "sensitivity_report.json"
SPOT_CHECK_PATH = ROOT / "spot_check_report.json"


def load_json(path: Path):
	# Read required JSON reports and fail fast if a file is missing.
	if not path.exists():
		raise FileNotFoundError(f"Missing required report file: {path.name}")
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def autopct_nonzero(values):
	# Show pie percentage labels only for non-zero slices to avoid clutter.
	total = sum(values)

	def _fmt(pct):
		absolute = round(pct * total / 100.0)
		return f"{pct:.1f}%" if absolute > 0 else ""

	return _fmt


def format_timestamp(value):
	# Convert ISO-style timestamps from report JSON into dashboard-friendly text.
	if not value:
		return "n/a"
	try:
		dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
		return dt.strftime("%Y-%m-%d %H:%M")
	except ValueError:
		return str(value)


def main():
	# Load official output artifacts from the evaluation pipeline.
	test_results = load_json(TEST_RESULTS_PATH)
	sensitivity = load_json(SENSITIVITY_PATH)
	spot_check = load_json(SPOT_CHECK_PATH)

	summary = test_results["summary"]
	# Validation summary values drive the donut chart and KPI center text.
	passed = summary.get("passed", 0)
	partial = summary.get("partial", 0)
	failed = summary.get("failed", 0)
	skipped = summary.get("skipped", 0)
	effectiveness = summary.get("effectiveness", 0)
	total_cases = passed + partial + failed + skipped
	test_ts = format_timestamp(test_results.get("timestamp"))
	sensitivity_ts = format_timestamp(sensitivity.get("timestamp"))
	spot_ts = format_timestamp(spot_check.get("timestamp"))

	configs = sensitivity.get("results", [])
	# Rank configurations by effectiveness and display the top 5 on the bar chart.
	top_configs = sorted(
		configs,
		key=lambda x: x.get("stats", {}).get("effectiveness", 0),
		reverse=True,
	)[:5]
	config_labels = [c.get("config", {}).get("name", "unknown") for c in top_configs]
	config_scores = [c.get("stats", {}).get("effectiveness", 0) for c in top_configs]

	spot_summary = spot_check.get("summary", {})
	# Spot-check metrics compare strict Top-1 performance against Top-3 coverage.
	top1_rate = float(spot_summary.get("top1_rate", 0))
	top3_rate = float(spot_summary.get("top3_rate", 0))
	failure_reasons = spot_summary.get("failure_reason_counts", {})

	plt.rcParams.update(
		# Global visual theme for a clean dashboard-like appearance.
		{
			"figure.facecolor": "#f7f9fc",
			"axes.facecolor": "#ffffff",
			"axes.edgecolor": "#d9dee8",
			"axes.labelcolor": "#2b2f36",
			"xtick.color": "#2b2f36",
			"ytick.color": "#2b2f36",
			"font.size": 10,
		}
	)

	subtitle = (
		f"Validation Cases: {total_cases} | "
		f"Validation: {test_ts} | Sensitivity: {sensitivity_ts} | Spot Check: {spot_ts}"
	)

	# 1) Validation outcome donut chart (Figure 1).
	fig1, ax = plt.subplots(figsize=(8.5, 6.5))
	fig1.suptitle("Validation Outcome Breakdown", fontsize=15, fontweight="bold", color="#1f2a44")
	fig1.text(0.5, 0.93, subtitle, ha="center", fontsize=9, color="#5b6270")
	labels = ["Passed", "Partial", "Failed", "Skipped"]
	values = [passed, partial, failed, skipped]
	colors = ["#2ca58d", "#f2b134", "#e04f5f", "#8d99ae"]
	ax.pie(
		values,
		labels=labels,
		colors=colors,
		startangle=120,
		autopct=autopct_nonzero(values),
		wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
		textprops={"color": "#1f2a44", "fontsize": 9},
	)
	center = plt.Circle((0, 0), 0.62, color="#ffffff")
	ax.add_artist(center)
	# Center annotation highlights overall effectiveness at a glance.
	ax.text(0, 0.05, f"{effectiveness:.1f}%", ha="center", va="center", fontsize=16, fontweight="bold", color="#1f2a44")
	ax.text(0, -0.12, "Effectiveness", ha="center", va="center", fontsize=9, color="#5b6270")
	ax.set_title(f"Validation Test Outcomes (n={total_cases})", fontsize=12, fontweight="bold")
	fig1.tight_layout(rect=[0, 0.01, 1, 0.92])

	# 2) Sensitivity analysis ranking (Figure 2).
	fig2, ax = plt.subplots(figsize=(9.5, 6.5))
	fig2.suptitle("Sensitivity Analysis Ranking", fontsize=15, fontweight="bold", color="#1f2a44")
	fig2.text(0.5, 0.93, subtitle, ha="center", fontsize=9, color="#5b6270")
	bars = ax.barh(config_labels[::-1], config_scores[::-1], color="#4f7cac")
	ax.set_xlim(0, 100)
	ax.set_xlabel("Effectiveness (%)")
	ax.set_title("Top Weight Configurations", fontsize=12, fontweight="bold")
	for bar in bars:
		# Print effectiveness values at bar ends for quick reading.
		width = bar.get_width()
		ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", fontsize=9, color="#1f2a44")
	ax.grid(axis="x", linestyle="--", alpha=0.2)
	fig2.tight_layout(rect=[0, 0.01, 1, 0.92])

	# 3) Real-data spot-check performance (Figure 3).
	fig3, ax = plt.subplots(figsize=(8.5, 6.5))
	fig3.suptitle("Real-Data Spot Check Performance", fontsize=15, fontweight="bold", color="#1f2a44")
	fig3.text(0.5, 0.93, subtitle, ha="center", fontsize=9, color="#5b6270")
	perf_labels = ["Top-1 Hit Rate", "Top-3 Hit Rate"]
	perf_values = [top1_rate, top3_rate]
	x_pos = [0, 1]
	ax.plot(x_pos, perf_values, color="#3a86ff", linewidth=2.5, marker="o", markersize=8)
	ax.fill_between(x_pos, perf_values, color="#3a86ff", alpha=0.12)
	ax.set_xticks(x_pos)
	ax.set_xticklabels(perf_labels)
	ax.set_ylim(0, 100)
	ax.set_ylabel("Rate (%)")
	ax.set_title("Real-Data Spot Check (Line)", fontsize=12, fontweight="bold")
	for x, y in zip(x_pos, perf_values):
		ax.text(x, y + 1.5, f"{y:.1f}%", ha="center", va="bottom", fontsize=10, color="#1f2a44")
	ax.grid(axis="y", linestyle="--", alpha=0.2)
	fig3.tight_layout(rect=[0, 0.01, 1, 0.92])

	# 4) Failure reason distribution (Figure 4).
	fig4, ax = plt.subplots(figsize=(10, 6.5))
	fig4.suptitle("Failure Reason Distribution", fontsize=15, fontweight="bold", color="#1f2a44")
	fig4.text(0.5, 0.93, subtitle, ha="center", fontsize=9, color="#5b6270")
	if failure_reasons:
		# Failure-reason bars surface remaining bottlenecks in the pipeline.
		reason_labels = list(failure_reasons.keys())
		reason_values = list(failure_reasons.values())
		bars = ax.bar(reason_labels, reason_values, color="#9b5de5")
		ax.set_title("Failure Reason Counts", fontsize=12, fontweight="bold")
		ax.set_ylabel("Count")
		ax.tick_params(axis="x", rotation=20)
		for bar in bars:
			height = bar.get_height()
			ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05, f"{int(height)}", ha="center", va="bottom", fontsize=9, color="#1f2a44")
		ax.grid(axis="y", linestyle="--", alpha=0.2)
	else:
		ax.text(0.5, 0.5, "No failure data available", ha="center", va="center", fontsize=11, color="#5b6270")
		ax.set_title("Failure Reason Counts", fontsize=12, fontweight="bold")
		ax.set_xticks([])
		ax.set_yticks([])
	fig4.tight_layout(rect=[0, 0.01, 1, 0.92])

	# Show four separate windows/figures.
	plt.show()


if __name__ == "__main__":
	main()