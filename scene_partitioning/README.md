# Scene partitioning

## Run the partition pipeline (notebook)

Use the notebook to run the partition on a Radiant Foam model and save results:  

[**scene_partitioning.ipynb**](./scene_partitioning.ipynb)

## Interactive viewer (KD-tree partition colours)

View the point cloud(Voronoi cell represented by its primal point) coloured by KD-tree partitions (cycle through different depths):

```bash
python3 kdtree_partition_viewer.py ./data/<scene_name>.ply
