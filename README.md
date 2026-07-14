# DeepLabCut Behavioral Analysis Scripts

Python scripts developed during a Master's thesis for automated analysis of rodent behavior from DeepLabCut pose estimation data.

## Included analyses
- Open Field Test
- Elevated Plus Maze
- Novel Object Recognition
- Y-Maze
- Y-Maze spontaneous alternation
- Social interaction test
- Stress test
- Distance and speed analysis

## Repository structure
- `scripts/` : Python analysis scripts
- `result/` : Example output figures
- `requirements.txt` : Python dependencies

## Usage
1. Run the desired script with DeepLabCut CSV outputs. with the comand
python "name of the script".py

## Outputs
The scripts generate trajectories, heatmaps, behavioral metrics, speed, distance, zone occupancy and other figures used during the Master's project.


## Important

Before running any script, **modify the input file name and path** in the script so that they match your own DeepLabCut output files (CSV) and any other required input documents.
