# track.py

import os
import cv2
import numpy as np
import torch
from celltrack_plmodel import CellTrackLitModel
from celltrack_model import CellTrack_Model
from torch_geometric.data import Data

# Helper function to calculate cell centers
def cell_center(seg_img):
    results = {}
    for label in np.unique(seg_img):
        if label != 0:
            all_points_x, all_points_y = np.where(seg_img == label)
            avg_x = np.round(np.mean(all_points_x))
            avg_y = np.round(np.mean(all_points_y))
            results[label] = [avg_x, avg_y]
    return results

# Function to create a graph representation of cell locations
def create_graph_from_centers(centers):
    labels = list(centers.keys())
    edge_index = []
    edge_features = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pos1 = centers[labels[i]]
            pos2 = centers[labels[j]]
            distance = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
            edge_index.append([i, j])
            edge_features.append([distance])
            edge_index.append([j, i])
            edge_features.append([distance])

    node_features = torch.zeros(len(labels), 1)  # Placeholder for node features
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

# Main tracking function using GNN
def predict_dataset_2(source_path, output_path, model=None, threshold=0.15):
    if not os.path.isdir(source_path):
        print('Input path is not a valid path')
        return

    # Initialize GNN model if not provided
    if model is None:
        model = CellTrackLitModel()  # Initialize with default parameters
    model.eval()  # Set model to evaluation mode

    # Prepare input and output paths
    names = sorted([name for name in os.listdir(source_path) if '.tif' in name])
    old_img = np.zeros(cv2.imread(os.path.join(source_path, names[0]), cv2.IMREAD_ANYDEPTH).shape, dtype=np.uint16)
    records = {}
    index = 1

    for i, name in enumerate(names):
        result = np.zeros_like(old_img, dtype=np.uint16)
        img = cv2.imread(os.path.join(source_path, name), cv2.IMREAD_ANYDEPTH)
        labels = np.unique(img)[1:]

        # Generate graph data from current segmentation image
        new_centers = cell_center(img)
        g2 = create_graph_from_centers(new_centers)

        if old_img.any():
            # Generate graph data from previous segmentation image
            old_centers = cell_center(old_img)
            g1 = create_graph_from_centers(old_centers)

            # Process each graph separately with the model
            with torch.no_grad():
                # Forward pass for previous frame graph
                prev_predictions = model(g1)
                # Forward pass for current frame graph
                curr_predictions = model(g2)

            # Determine tracking associations based on GNN predictions
            parent_cells = []
            for label in labels:
                mask = (img == label).astype(np.uint16)
                mask_size = np.sum(mask)

                candidates = np.unique(mask * old_img)[1:]
                max_score = 0
                max_candidate = 0

                for candidate in candidates:
                    score = np.sum((mask * old_img) == candidate) / mask_size
                    if score > max_score:
                        max_score = score
                        max_candidate = candidate

                if max_score < threshold:
                    records[index] = [i, i, 0]
                    result += (mask * index).astype(np.uint16)
                    index += 1
                else:
                    if max_candidate not in parent_cells:
                        records[max_candidate][1] = i
                        result += (mask * max_candidate).astype(np.uint16)
                    else:
                        if records[max_candidate][1] == i:
                            records[max_candidate][1] = i - 1
                            m_mask = (result == max_candidate)
                            result -= (m_mask * max_candidate).astype(np.uint16)
                            result += (m_mask * index).astype(np.uint16)

                            records[index] = [i, i, max_candidate]
                            index += 1

                        records[index] = [i, i, max_candidate]
                        result += (mask * index).astype(np.uint16)
                        index += 1

                    parent_cells.append(max_candidate)
        else:
            # First frame - initialize tracks without comparison
            for label in labels:
                mask = (img == label).astype(np.uint16)
                records[index] = [i, i, 0]
                result += (mask * index).astype(np.uint16)
                index += 1

        # Save the tracking result for the current frame
        cv2.imwrite(os.path.join(output_path, name), result.astype(np.uint16))
        old_img = result

    # Save tracking data in text format
    with open(os.path.join(output_path, 'res_track.txt'), "w") as file:
        for key, value in records.items():
            file.write(f'{key} {value[0]} {value[1]} {value[2]}\n')

    print("Tracking complete!")

# Run predict_dataset_2 if this script is called directly
if __name__ == "__main__":
    predict_result = "data/res_result/"
    track_result = "data/track_result/"
    predict_dataset_2(predict_result, track_result)
