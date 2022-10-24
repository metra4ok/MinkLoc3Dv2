import argparse
import os
from time import time

import tqdm
import torch
import numpy as np
import MinkowskiEngine as ME

from models.model_factory import model_factory
from misc.utils import TrainingParams
from datasets.base_datasets import read_pc, normalize_pc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on PointNetVLAD (Oxford) dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--input_dir', type=str, required=True, help="Input directory")
    parser.add_argument('--ext', type=str, default='pcd')
    parser.add_argument('--out_dir', type=str, required=True)

    args = parser.parse_args()

    params = TrainingParams(args.config, args.model_config, debug=False)
    params.print()

    out_dir = args.out_dir
    if os.path.exists(out_dir):
        print(f"saving descriprors into directory: {out_dir}")
    else:
        print(f"creating output directory: {out_dir}")
        os.makedirs(out_dir)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(params.model_params)
    assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
    print('Loading weights: {}'.format(args.weights))
    model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)

    input_dir = args.input_dir
    assert os.path.exists(input_dir)

    ext = args.ext
    assert ext in ['pcd', 'bin']

    input_dir_list = os.listdir(input_dir)
    input_list = []
    for fname in input_dir_list:
        if fname.endswith(ext):
            input_list.append(fname)

    input_list = sorted(input_list)
    input_list = [os.path.join(input_dir, f) for f in input_list]

    des_list = np.zeros((len(input_list), 256))
    times = []
    for i, f in tqdm.tqdm(enumerate(input_list), total=len(input_list)):
        pc = read_pc(f, ext=ext)
        
        start_time = time()
        pc = normalize_pc(pc)
        quantized_pc, ndx = ME.utils.sparse_quantize(pc, quantization_size=0.01, return_index=True)
        coords = ME.utils.batched_coordinates([quantized_pc])
        feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
        batch = {'coords': coords, 'features': feats}
        batch = {e: batch[e].to(device) for e in batch}
        out = model(batch)
        des_list[i, :] = out['global'].detach().cpu().numpy()
        time_taken_ms = int((time() - start_time) * 1000)
        times.append(time_taken_ms)

    des_list = des_list.astype('float32')
    des_list_filename = os.path.join(out_dir, 'minkloc3d_descriptors.npy')
    np.save(des_list_filename, des_list)
    print(f"Saved descriptors to file: {des_list_filename}")

    mean_time = np.mean(times)
    max_time = np.max(times)
    min_time = np.min(times)
    std_time = np.std(times)
    time_performance_filename = os.path.join(out_dir, 'minkloc3d_time_performance.txt')
    with open(time_performance_filename, 'w') as time_performance_txt:
        report_str = "time performance results:" + \
                     f"\n\tmean: {mean_time:.3f} ms" + \
                     f"\n\tmax:  {max_time:.3f} ms" + \
                     f"\n\tmin:  {min_time:.3f} ms" + \
                     f"\n\tstd:  {std_time:.3f} ms"
        print('\n'+report_str+'\n')
        time_performance_txt.write(report_str)
