import matplotlib.pyplot as plt

from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, inference_model, show_result_pyplot

def inference_ade():
    img_file = 'data/ade/ADEChallengeData2016/images/validation/ADE_val_00000001.jpg'
    config_file = 'mmsegmentation-main/configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py'
    checkpoint_file = 'pretrained_models/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth'

    model = init_model(config=config_file, checkpoint=checkpoint_file, device='cpu')
    model = revert_sync_batchnorm(model)
    
    result = inference_model(model, img_file)

    # Show the results
    vis_result = show_result_pyplot(model, img_file, result, show=False, save_dir='temp')
    
    plt.imshow(vis_result)
    plt.show()


def inference_cityscapes():
    img_file = 'data/cityscapes/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png'
    config_file = 'mmsegmentation-main/configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py'
    checkpoint_file = 'pretrained_models/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth'

    model = init_model(config=config_file, checkpoint=checkpoint_file, device='cpu')
    model = revert_sync_batchnorm(model)
    
    result = inference_model(model, img_file)

    # Show the results
    vis_result = show_result_pyplot(model, img_file, result, show=False, save_dir='temp')
    
    plt.imshow(vis_result)
    plt.show()


if __name__ == '__main__':
    # inference_ade()
    inference_cityscapes()

