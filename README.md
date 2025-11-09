Advanced RTK Farm Rover Traversal System (v12.9)

This repository holds the path planning logic for an **RTK GPS-enabled farm rover**, designed for optimized field coverage and autonomous sowing. The system generates a fixed, pre-calculated path (waypoints and operational flags) for a rectangular field, prioritizing a high-efficiency **boustrophedon (serpentine) inner sweep** before completing the perimeter.

The logic guarantees full field coverage while offering precise control over the mission's termination, with specialized handling for corner and custom boundary exit points, including automatic management of unsown buffer gaps.

Problem Statement

Modern precision agriculture demands efficient, repeatable, and complete field coverage to maximize yield and minimize operational costs (fuel, time, wear). Traditional manual or basic automated path planning often results in **redundant travel, inefficient turning, and incomplete coverage** in the headlands or around field boundaries.

The core challenge addressed by this project is the need for a **deterministic, optimized traversal plan** that:

1.  **Prioritizes maximum productive time** by using the quickest path for the inner field area (boustrophedon).
2.  **Guarantees complete sowing** of the crucial, irregularly shaped boundary lanes (headlands) after the main sweep.
3.  **Facilitates safe mission termination** at a designated exit point without leaving any unintended unsown area near the boundary, accomplished by managing turn-out gaps.
4.  **Is parametrically scalable** based on variable field size and fixed rover dimensions.

Core Functionality

| Feature | Description |
| :--- | :--- |
| **Boustrophedon Sweep** | Implements efficient vertical-pass serpentine coverage across all inner lanes (excluding the boundary lanes 0 and $X_{max}$). |
| **Headland Sowing** | Ensures the entire perimeter (headlands) is sown after the main inner sweep, using an optimal retracing path that minimizes non-productive distance. |
| **Exit Management** | Supports two mission end modes: **Fixed Corner Exit** (e.g., Top-Left, Bottom-Right) and **Custom Boundary Exit** (any non-corner boundary lane). |
| **Gap Buffer** | Automatically introduces unsown segments (`gap_size` units) immediately adjacent to the chosen exit point to facilitate safe rover retrieval and turn-out maneuvers. |
| **Telemetry & Visualization** | Includes a `matplotlib` animation for visual path verification and a `LiveTelemetryLogger` that outputs runtime data to the console and a persistent `navigation_log.csv` file. |


Usage

Requirements

Requires the standard Python data science and visualization libraries:

```bash
pip install numpy matplotlib
```

Execution

1.  Run the main script from your terminal:
    ```bash
    python your_script_name.py
    ```
2.  The program will prompt for core system parameters:
      * **Farm Dimensions (m):** Width (X-axis) and Breadth (Y-axis).
      * **Rover Dimensions (m):** Rover Width (determines lane spacing) and Rover Length (determines coverage depth).
      * **Exit Strategy:** Choose a corner or a custom boundary lane for mission completion.

The script validates inputs, calculates the optimal grid, generates the path, and initiates the real-time simulation and logging sequence.

Grid and Indexing

The path planning operates on a 0-indexed grid derived from the field and rover dimensions:

  * **Vertical Passes (X-lanes):** Range from $0$ to $X_{max}$. Lanes 0 and $X_{max}$ are the headland passes.
  * **Horizontal Rows (Y-lanes):** Range from $0$ to $Y_{max}$. Rows 0 and $Y_{max}$ are the headland rows.
  * **Waypoints:** Defined by $ (lane\_x, lane\_y) $, corresponding to the center of the lane coverage area.

Telemetry Data Structure

The system logs every movement segment to `navigation_log.csv`. This data is essential for post-mission analysis of efficiency, coverage, and time usage.

| Column | Data Format | Purpose |
| :--- | :--- | :--- |
| `Timestamp` | Datetime (ISO format) | Time of segment completion. |
| `Step` | Integer | Segment index in the overall path sequence. |
| `Label` | String | Automated identifier (e.g., VRow1, HRow2, H-Turn). |
| `From (m)`, `To (m)` | Tuple String $(X, Y)$ | Start and end center coordinates in meters. |
| `SegDist (m)` | Float (1 d.p.) | Distance of the current segment. |
| `SownDist (m)` | Float (1 d.p.) | Cumulative distance where sowing was active. |
| `Action`, `FarmType` | String | Categorization of movement (e.g., `INNER_VERTICAL_FARMING`, `NAVIGATION_UNSOWN`). |
