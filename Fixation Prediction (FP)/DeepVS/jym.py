'''
    1. DeepVS得到第一帧的眼动图
    2. 眼动图转成tensor
    3. tensor送入ASNet最后一层
    4. 生成显著图
'''

from jym_FPPredictor import main_DeepVS
from jym_SODPredictor import main_ASNet
import jym_gol


if __name__ == '__main__':
    print("###########################################")

    print("In DeepVS, predicting FP...")
    fp_from_DeepVS = main_DeepVS()
    print("FP done, tensor returned.")
    jym_gol._init()
    jym_gol.set_value('fp0', fp_from_DeepVS)
    print("###########################################")

    print("In ASNet, getting FP tensor.")
    main_ASNet()
    print("SOD done, image returned.")
    print("###########################################")
