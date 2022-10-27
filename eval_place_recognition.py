"""
Evaluate Place Recognition results

Given:
    - Descriptors for query and database.
    - Poses information for both query and database.
Results:
    - Recall@N metrics, where N in {1, 3, 5, 10}. The match is considered "true positive"
      if the distance to the GT pose of the query is less than dist_threshold.
    - Draws trajectory with "good" and "bad" matches
    - TODO: visualize input images and pcds (optionally)
"""
import argparse
import os
import os.path as osp
from datetime import datetime

import numpy as np
import faiss
import cv2
from tqdm import tqdm

from matplotlib.backends.backend_agg import FigureCanvas
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # using non-GUI backend solves OOM issue and fasten the processing


# define fonts
black_text_font = (cv2.FONT_HERSHEY_COMPLEX,  # font
                   1.5,  # font scale
                   (50, 50, 50),  # color
                   2,  # thickness
                   cv2.LINE_AA)
red_text_font = (cv2.FONT_HERSHEY_COMPLEX,  # font
                 1.5,  # font scale
                 (0, 0, 128),  # color
                 2,  # thickness
                 cv2.LINE_AA)
green_text_font = (cv2.FONT_HERSHEY_COMPLEX,  # font
                   1.5,  # font scale
                   (0, 128, 0),  # color
                   2,  # thickness
                   cv2.LINE_AA)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # TODO: write help for each arg
    parser.add_argument('--query_descriptors', type=str, required=True)
    parser.add_argument('--db_descriptors', type=str, required=True)
    parser.add_argument('--query_poses', type=str, required=True)
    parser.add_argument('--db_poses', type=str, required=True)
    parser.add_argument('--dist_threshold', type=float, default=25.0)
    parser.add_argument('--out_dir', type=str, default='./output')
    parser.add_argument('--method', type=str, default='MinkLoc3Dv2')
    args = parser.parse_args()

    # check if args are correct
    assert osp.isfile(args.query_descriptors) and args.query_descriptors.endswith('.npy'), \
        "The query_descriptors should be existing '.npy' file"
    assert osp.isfile(args.db_descriptors) and args.db_descriptors.endswith('.npy'), \
        "The db_descriptors should be existing '.npy' file"
    assert osp.isfile(args.query_poses) and args.query_poses.endswith('.txt'), \
        "The query_poses should be existing '.txt' file"
    assert osp.isfile(args.db_poses) and args.db_poses.endswith('.txt'), \
        "The db_poses should be existing '.txt' file"
    assert args.dist_threshold > 0, "dist_threshold should be positive float"

    return args


def parse_poses_txt(filename, timestamps_first=True):
    with open(filename) as txt:
        lines = txt.readlines()
    lines = [l[:-1].split(' ') for l in lines]
    positions_list = []
    for l in lines:
        if timestamps_first:
            position = np.array([float(l[1]), float(l[2]), float(l[3])])
        else:
            position = np.array([float(l[0]), float(l[1]), float(l[2])])
        positions_list.append(position)
    return np.array(positions_list)


def dist(a, b):
    return np.linalg.norm(a - b)


def plot_trajectory(db_poses, q_poses, gt_id, matched_id, threshold):
    db_x_coords = db_poses[:,0]
    db_y_coords = db_poses[:,1]

    q_x_coords = q_poses[:,0]
    q_y_coords = q_poses[:,1]

    fig = plt.figure(figsize=(4,4), dpi=270)  # dpi 270 result in 1080x1080 output dim
    ax = fig.add_subplot(111)
    fig.set_tight_layout(True)
    ax.axis('off')
    ax.set_xlim(db_x_coords.min() - 10, db_x_coords.max() + 10)
    ax.set_ylim(db_y_coords.min() - 10, db_y_coords.max() + 10)
    
    ax.plot(db_x_coords, db_y_coords, 'g', label='database trajectory', linewidth=0.2)
    ax.plot(q_x_coords, q_y_coords, 'b', label='query gt trajectory', linewidth=0.2)

    ax.scatter(db_x_coords[matched_id], db_y_coords[matched_id],
               color='red', label='matched pose', s=2, alpha=1)

    # gt pose dot
    ax.scatter(q_x_coords[gt_id], q_y_coords[gt_id],
               color='cyan', label='gt pose', s=2, alpha=1)
    # gt pose threshold area
    threshold_area = np.pi * threshold**2
    ax.scatter(q_x_coords[gt_id], q_y_coords[gt_id],
               color='cyan', label=f'correct match threshold ({int(threshold)} m)', s=threshold_area, alpha=0.1, linewidth=0)

    ax.legend()

    canvas = FigureCanvas(fig)
    canvas.draw()

    # convert canvas to image
    graph_image = np.array(fig.canvas.get_renderer()._renderer)

    # it still is rgb, convert to opencv's default bgr
    graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)

    plt.close(fig)
    plt.close()
    plt.clf()
    del canvas
    del fig

    return graph_image


def draw_frame(db_poses, q_poses, gt_id, matched_id, threshold):
    trajectory_plot = plot_trajectory(db_poses, q_poses, gt_id, matched_id,
                                      threshold)
    frame = np.full((1080, 1920, 3), fill_value=255, dtype=np.uint8)
    frame[:, :1080, :] = trajectory_plot
    frame = cv2.putText(frame, f'query id: {query_index}', (1080, 850),
                        *black_text_font)
    frame = cv2.putText(frame, f'top-1 matched db id: {db_indices[0]}', (1080, 910),
                        *black_text_font)
    frame = cv2.putText(frame, f'top-1 distance: {top_1_distance:.2f} m', (1080, 970),
                        *black_text_font)
    if top_1_distance < opt.dist_threshold:
        frame = cv2.putText(frame, 'success', (1080, 1030),
                            *green_text_font)
    else:
        frame = cv2.putText(frame, 'failure', (1080, 1030),
                            *red_text_font)

    return frame


def make_description_str(opt):
    description = f"Method name: {opt.method}"
    description += f"\nDistance threshold: {opt.dist_threshold:.1f} m"
    description += "\nQuery:"
    description += f"\n\tdescriptors file: {opt.query_descriptors}"
    description += f"\n\tposes file: {opt.query_poses}"
    description += "\nDatabase:"
    description += f"\n\tdescriptors file: {opt.db_descriptors}"
    description += f"\n\tposes file: {opt.db_poses}"
    return description


def make_results_str(recalls):
    results = "Results:"
    results += f"\n\tR@1  = {recalls[1]*100:.2f} %"
    results += f"\n\tR@3  = {recalls[3]*100:.2f} %"
    results += f"\n\tR@5  = {recalls[5]*100:.2f} %"
    results += f"\n\tR@10 = {recalls[10]*100:.2f} %"
    
    return results


if __name__ == '__main__':
    opt = parse_arguments()

    # create output directory
    cur_datetime = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    out_dir = osp.join(opt.out_dir, f'{opt.method}_{cur_datetime}')
    print(f"Creating output directory: {out_dir}")
    os.makedirs(osp.join(out_dir, 'frames'))

    query_descriptors = np.load(opt.query_descriptors)
    db_descriptors = np.load(opt.db_descriptors)

    query_poses = parse_poses_txt(opt.query_poses)
    db_poses = parse_poses_txt(opt.db_poses)

    assert len(query_descriptors) == len(query_poses)
    assert len(db_descriptors) == len(db_poses)

    # if everything loaded successfully - print out and save description
    description = make_description_str(opt)
    print(description)
    with open(osp.join(out_dir, 'description.txt'), 'w') as f:
        f.write(description)

    index = faiss.IndexFlatL2(db_descriptors.shape[1])
    index.add(db_descriptors)

    k = 10  # top-10 matches
    D, I = index.search(query_descriptors, k)  # D - top-k similarities, I - top-k indexes of matches

    query_len = len(query_descriptors)
    recalls = {1: 0.0,
               3: 0.0,
               5: 0.0,
               10: 0.0}

    for query_index, db_indices in tqdm(enumerate(I), desc='Calculating recalls', total=len(I)):
        top_1_distance = dist(query_poses[query_index], db_poses[db_indices[0]])

        # draw and save pretty frame
        frame = draw_frame(db_poses, query_poses, query_index, db_indices[0],
                           opt.dist_threshold)
        frame_name = f'frame_{str(query_index).zfill(6)}.png'
        cv2.imwrite(osp.join(out_dir, 'frames', frame_name), frame)

        for p, i in enumerate(db_indices):
            if dist(query_poses[query_index], db_poses[i]) < opt.dist_threshold:
                if p < 1:
                    recalls[1] += 1 / query_len
                if p < 3:
                    recalls[3] += 1 / query_len
                if p < 5:
                    recalls[5] += 1 / query_len
                if p < 10:
                    recalls[10] += 1 / query_len
                break

    results_str = make_results_str(recalls)
    print(results_str)
    results_filename = osp.join(out_dir, 'results.txt')
    with open(results_filename, 'w') as f:
        f.write(results_str)
    print(f"Saved results to: {results_filename}")
