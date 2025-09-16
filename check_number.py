
def count_ply_points_binary(ply_path):
    with open(ply_path, 'rb') as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.startswith(b"element vertex"):
                num_points = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
        return num_points

ply_file = "/Downloads/3dgs/output/72cd46c3-f/point_cloud/iteration_30000/point_cloud.ply"
points = count_ply_points_binary(ply_file)
print(f"Number of points: {points}")

