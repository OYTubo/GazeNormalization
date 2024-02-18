import numpy as np

# 可靠性校验模块（不需要）
def vModule(devSt, preds, pogs, w, h):
    blink = []
    eps = 3*devSt
    vpreds = []
    vpogs = []
    for pred,pog in zip(preds,pogs):
        print(pred, pog)
        # condition 1: in the screen
        if 0 <= pred[0] <= w and 0 <= pred[1] <= h:
            l2 = np.linalg.norm(pred - pog)
            print(l2)
            if l2 < eps:
                blink.append(1)
                vpreds.append(pred)
                vpogs.append(pog)
                continue
        blink.append(0)
    vpreds = np.asarray(vpreds)
    vpogs = np.asarray(vpogs)
    blink = np.asarray(blink)
    return blink,vpreds,vpogs
    

# 离线自校准
def scModule_offline(preds,history_pogs,history_preds):
    refine_pred = []
    tmp = np.mean(history_preds, axis=0)
    gtr = np.mean(history_pogs, axis=0)
    refine_pred = preds - (tmp - gtr)
    refine_pred = np.asarray(refine_pred)
    return refine_pred

