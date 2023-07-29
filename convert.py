import argparse
import importlib
from modules.preprocess import prepare_cfg
from modules.model import load_model
import torch
import os
import subprocess

from openvino.tools import mo
from openvino.runtime import serialize

def make_parser():
    parser = argparse.ArgumentParser(description='BirdCLEF2023')
    parser.add_argument('--model_name', choices=["sed_v2s",'sed_b3ns','sed_seresnext26t','cnn_v2s','cnn_resnet34d','cnn_b3ns','cnn_b0ns'])
    return parser

def main():
    stage = 'train_ce'
    parser = make_parser()
    args = parser.parse_args()
    model_name = args.model_name
    cfg = importlib.import_module(f'configs.{model_name}').basic_cfg
    cfg = prepare_cfg(cfg,stage)

    model = load_model(cfg,stage,train=False) # train=False
    
    onxx_model_path = os.path.join(cfg.onnx_path,f"{model_name}.onnx")
    input_dummy = torch.randn(*cfg.input_shape)
    torch.onnx.export(model, input_dummy, onxx_model_path, verbose=True, input_names=cfg.input_names, output_names=cfg.output_names,opset_version=cfg.opset_version)

    # openvino command line interface - run
    #proc = subprocess.run(['mo','--input_model', onxx_model_path, '--output_dir', cfg.openvino_path,'--compress_to_fp16'], shell=True,)
    
    print("Exporting ONNX model to IR... This may take a few minutes.")
    cfg.openvino_path = cfg.openvino_path + "/sed.xml"
    ov_model = mo.convert_model(onxx_model_path, compress_to_fp16=True)
    serialize(ov_model, cfg.openvino_path)

if __name__=='__main__':
    main()