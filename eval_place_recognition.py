"""
Evaluate Place Recognition results

Given:
    - Descriptors for query and database.
    - Poses information for both query and database.
Results:
    - Recall@N metrics, where N in {1, 3, 5, 10}. The match is considered "true positive"
      if the distance to the GT pose of the query is less than dist_threshold.
    - TODO: draws trajectory with "good" and "bad" matches (optionally)
    - TODO: visualize input images and pcds (optionally)
"""
import argparse
import os.path as osp
import numpy as np
import faiss


def parse_arguments():
    parser = argparse.ArgumentParser()
    # TODO: write help for each arg
    parser.add_argument('--query_descriptors', type=str, required=True)
    parser.add_argument('--db_descriptors', type=str, required=True)
    parser.add_argument('--query_poses', type=str, required=True)
    parser.add_argument('--db_poses', type=str, required=True)
    parser.add_argument('--dist_threshold', type=float, default=25.0)
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


if __name__ == '__main__':
    opt = parse_arguments()

    query_descriptors = np.load(opt.query_descriptors)
    db_descriptors = np.load(opt.db_descriptors)

    query_poses = parse_poses_txt(opt.query_poses)
    db_poses = parse_poses_txt(opt.db_poses)

    assert len(query_descriptors) == len(query_poses)
    assert len(db_descriptors) == len(db_poses)

    index = faiss.IndexFlatL2(db_descriptors.shape[1])
    index.add(db_descriptors)

    k = 10  # top-10 matches
    D, I = index.search(query_descriptors, k)  # D - top-k similarities, I - top-k indexes of matches

    query_len = len(query_descriptors)
    recalls = {1: 0.0,
               3: 0.0,
               5: 0.0,
               10: 0.0}

    for query_index, db_indices in enumerate(I):
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

    print(recalls)
