import subprocess

def main():
    subprocess.run(
        [
            "python",
            "mmdeploy-main/tools/deploy.py",
            "mmdeploy-main/configs/mmseg/segmentation_onnxruntime_dynamic.py",
            "mmsegmentation-main/configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py",
            "pretrained_models/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth",
            "data/ade/ADEChallengeData2016/images/validation/ADE_val_00000001.jpg",
            "--device", "cpu",
            "--work-dir", "temp",
            "--show"
        ]
    )


if __name__ == '__main__':
    main()
