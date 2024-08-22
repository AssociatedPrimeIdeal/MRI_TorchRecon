from utils import *

def ReconLLR(A, K, method, llr, it, reg, blk, L, regfactor=1, device='cuda', gt=None, save_loss=False, mask=None):
    '''
    :param A: Operator consists of under-sampling mask, fourier transform and coil sensitivity maps
    :param K: Acquired under-sampled k-space
    :param method: Iteration method (ISTA, FISTA, POGM)
    :param llr: Use locally low-rank or globally low-rank
    :param it: Iteration number
    :param reg: regularization of the nuclear norm term
    :param blk: block size for locally low-rank
    :param L: Lipschitz constant
    :param regfactor: scale factor for denormalization
    :param device: cuda or cpu
    :param gt: ground truth image, if it is not None, PSNR, SSIM, and nRMSE will be calculated and stored in metrics
    :param save_loss: if save_loss is True, loss will be calculated and stored in metrics
    :param mask: mask of flow for calculating art_nrmse and vein_nrmse
    :return: X:reconstructed image, metrics:dictionary
    '''
    Nt, Nc, FE, PE, SPE = K.shape
    loop = tqdm.tqdm(range(1, it + 1), total=it)
    metrics = {}
    if save_loss:
        metrics['loss'] = []
        metrics['loss1'] = []
        metrics['loss2'] = []
    if gt is not None:
        metrics['psnr'] = []
        metrics['ssim'] = []
        metrics['nrmse'] = []
        metrics['art_nrmse'] = []
        metrics['vein_nrmse'] = []
    if llr:
        stepx = np.ceil(FE / blk)
        stepy = np.ceil(PE / blk)
        stepz = np.ceil(SPE / blk)
        padx = (stepx * blk).astype('int64')
        pady = (stepy * blk).astype('int64')
        padz = (stepz * blk).astype('int64')
        M = blk ** 3
        N = Nt
        B = padx * pady * padz / M
        RF = GETWIDTH(M, N, B)
        reg *= RF
    else:
        reg *= (np.sqrt(np.prod(K.shape[-3:])) + 1)
    if method == 'ISTA':
        X = A.mtimes(K, 1)
        for i in loop:
            axb = A.mtimes(X, 0) - K
            X = X - 1 / L * A.mtimes(axb, 1)
            if llr:
                X, loss2 = SVT_LLR(X, reg / L, blk)
            else:
                X, loss2 = SVT(X, reg / L)
            if save_loss:
                loss1 = torch.sum(torch.abs(A.mtimes(X, 0) - K) ** 2).item()
                metrics['loss1'].append(loss1)
                metrics['loss2'].append(loss2)
                metrics['loss'].append(loss1 * 0.5 + loss2 * reg)
            if gt is not None:
                metrics['art_nrmse'].append(
                    nRMSE(torch.angle(X)[mask == 1], torch.angle(gt)[mask== 1], use_torch=True).item())
                metrics['vein_nrmse'].append(
                    nRMSE(torch.angle(X)[mask == 2], torch.angle(gt)[mask == 2], use_torch=True).item())
                metrics['psnr'].append(PSNR(torch.abs(X) * regfactor, torch.abs(gt), use_torch=True).item())
                metrics['ssim'].append(SSIM(torch.abs(X.unsqueeze(0).unsqueeze(0)) * regfactor, torch.abs(gt.unsqueeze(0).unsqueeze(0)), use_torch=True))
                metrics['nrmse'].append(nRMSE(torch.abs(X) * regfactor, torch.abs(gt), use_torch=True).item())
    elif method == 'FISTA':
        tp = 1
        Xp = A.mtimes(K, 1)
        Y = Xp.clone()
        for i in loop:
            t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
            axb = A.mtimes(Y, 0) - K
            Y = Y - 1 / L * A.mtimes(axb, 1)
            if llr:
                X, loss2 = SVT_LLR(Y, reg / L, blk)
            else:
                X, loss2 = SVT(Y, reg / L)
            Y = X + (tp - 1) / t * (X - Xp)
            Xp = X
            tp = t
            if save_loss:
                loss1 = torch.sum(torch.abs(A.mtimes(X, 0) - K) ** 2).item()
                metrics['loss1'].append(loss1)
                metrics['loss2'].append(loss2)
                metrics['loss'].append(loss1 * 0.5 + loss2 * reg)
            if gt is not None:
                metrics['art_nrmse'].append(
                    nRMSE(torch.angle(X)[mask == 1], torch.angle(gt)[mask== 1], use_torch=True).item())
                metrics['vein_nrmse'].append(
                    nRMSE(torch.angle(X)[mask == 2], torch.angle(gt)[mask == 2], use_torch=True).item())
                metrics['psnr'].append(PSNR(torch.abs(X) * regfactor, torch.abs(gt), use_torch=True).item())
                metrics['ssim'].append(SSIM(torch.abs(X.unsqueeze(0).unsqueeze(0)) * regfactor, torch.abs(gt.unsqueeze(0).unsqueeze(0)), use_torch=True))
                metrics['nrmse'].append(nRMSE(torch.abs(X) * regfactor, torch.abs(gt), use_torch=True).item())
    elif method == 'POGM':
        tp = 1
        gp = 1
        Xp = A.mtimes(K, 1)
        X, Y, Z, Yp, Zp = Xp.clone(), Xp.clone(), Xp.clone(), Xp.clone(), Xp.clone()
        for i in loop:
            t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
            g = 1 / L * (2 * tp + t - 1) / t
            axb = A.mtimes(X, 0) - K
            Y = X - 1 / L * A.mtimes(axb, 1)
            Z = Y + (tp - 1) / t * (Y - Yp) + tp / t * (Y - Xp) + (tp - 1) / (L * gp * t) * (Zp - Xp)
            if llr:
                X, loss2 = SVT_LLR(Z, reg * g, blk)
            else:
                X, loss2 = SVT(Z, reg * g)
            Xp = X
            Yp = Y
            Zp = Z
            tp = t
            gp = g
            if save_loss:
                loss1 = torch.sum(torch.abs(A.mtimes(X, 0) - K) ** 2).item()
                metrics['loss1'].append(loss1)
                metrics['loss2'].append(loss2)
                metrics['loss'].append(loss1 * 0.5 + loss2 * reg)
            if gt is not None:
                metrics['art_nrmse'].append(
                    nRMSE(torch.angle(X)[mask == 1], torch.angle(gt)[mask== 1], use_torch=True).item())
                metrics['vein_nrmse'].append(
                    nRMSE(torch.angle(X)[mask == 2], torch.angle(gt)[mask == 2], use_torch=True).item())
                metrics['psnr'].append(PSNR(torch.abs(X) * regfactor, torch.abs(gt), use_torch=True).item())
                metrics['ssim'].append(SSIM(torch.abs(X.unsqueeze(0).unsqueeze(0)) * regfactor, torch.abs(gt.unsqueeze(0).unsqueeze(0)), use_torch=True))
                metrics['nrmse'].append(nRMSE(torch.abs(X) * regfactor, torch.abs(gt), use_torch=True).item())
    return X * regfactor, metrics


def ReconLplusS(A, K, method, llr, it, regL, regS, blk, Lc, regfactor=None, device='cuda', gt=None, save_loss=False, mask=None):
    '''
    :param A: Operator consists of under-sampling mask, fourier transform and coil sensitivity maps
    :param K: Acquired under-sampled k-space
    :param method: Iteration method (ISTA, FISTA, POGM)
    :param llr: Use locally low-rank or globally low-rank
    :param it: Iteration number
    :param regL: regularization of the low-rank term
    :param regS: regularization of the sparse term
    :param blk: block size for locally low-rank
    :param Lc: Lipschitz constant
    :param regfactor: scale factor for denormalization
    :param device: cuda or cpu
    :param gt: ground truth image, if it is not None, PSNR, SSIM, and nRMSE will be calculated and stored in metrics
    :param save_loss: if save_loss is True, loss will be calculated and stored in metrics
    :param mask: mask of flow for calculating art_nrmse and vein_nrmse
    :return: X:reconstructed image, metrics:dictionary
    '''
    Nt, Nc, FE, PE, SPE = K.shape
    loop = tqdm.tqdm(range(1, it + 1), total=it)
    metrics = {}
    if save_loss:
        metrics['loss'] = []
        metrics['loss1'] = []
        metrics['loss2'] = []
        metrics['loss3'] = []
    if gt is not None:
        metrics['psnr'] = []
        metrics['ssim'] = []
        metrics['nrmse'] = []
        metrics['art_nrmse'] = []
        metrics['vein_nrmse'] = []
    if llr:
        stepx = np.ceil(FE / blk)
        stepy = np.ceil(PE / blk)
        stepz = np.ceil(SPE / blk)
        padx = (stepx * blk).astype('int64')
        pady = (stepy * blk).astype('int64')
        padz = (stepz * blk).astype('int64')
        M = blk ** 3
        N = Nt
        B = padx * pady * padz / M
        RF = GETWIDTH(M, N, B)
        regL *= RF
    else:
        regL *= (np.sqrt(np.prod(K.shape[-3:])) + 1)

    if method == 'ISTA':
        X = A.mtimes(K, 1)
        L, Lp = X.clone(), X.clone()
        S = torch.zeros_like(X).to(device)
        for i in loop:
            if llr:
                L, loss2 = SVT_LLR(X - S, regL / Lc, blk)
            else:
                L, loss2 = SVT(X - S, regL / Lc)
            S, loss3 = Sparse(X - Lp, regS / Lc)
            axb = A.mtimes(L + S, 0) - K
            X = L + S - 1 / Lc * A.mtimes(axb, 1)
            Lp = L
            if save_loss:
                loss1 = torch.sum(torch.abs(A.mtimes(X, 0) - K) ** 2).item()
                metrics['loss1'].append(loss1)
                metrics['loss2'].append(loss2)
                metrics['loss3'].append(loss3)
                metrics['loss'].append(loss1 * 0.5 + loss2 * regL + loss3 * regS)
            if gt is not None:
                metrics['art_nrmse'].append(
                    nRMSE(torch.angle(X)[mask == 1], torch.angle(gt)[mask== 1], use_torch=True).item())
                metrics['vein_nrmse'].append(
                    nRMSE(torch.angle(X)[mask == 2], torch.angle(gt)[mask == 2], use_torch=True).item())
                metrics['psnr'].append(PSNR(torch.abs(X) * regfactor, torch.abs(gt), use_torch=True).item())
                metrics['ssim'].append(SSIM(torch.abs(X.unsqueeze(0).unsqueeze(0)) * regfactor, torch.abs(gt.unsqueeze(0).unsqueeze(0)), use_torch=True))
                metrics['nrmse'].append(nRMSE(torch.abs(X) * regfactor, torch.abs(gt), use_torch=True).item())
    elif method == 'FISTA':
        tp = 1
        X = A.mtimes(K, 1)
        L, Lp, Lh = X.clone(), X.clone(), X.clone()
        S, Sp, Sh = torch.zeros_like(X).to(device), torch.zeros_like(X).to(device), torch.zeros_like(X).to(device)
        for i in loop:
            t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
            if llr:
                L, loss2 = SVT_LLR(X - Sh, regL / Lc, blk)
            else:
                L, loss2 = SVT(X - Sh, regL / Lc)
            S, loss3 = Sparse(X - Lh, regS / Lc)
            Lh = L + (tp - 1) / t * (L - Lp)
            Sh = S + (tp - 1) / t * (S - Sp)
            axb = A.mtimes(Lh + Sh, 0) - K
            X = Lh + Sh - 1 / Lc * A.mtimes(axb, 1)
            tp = t
            Lp = L
            Sp = S
            if save_loss:
                loss1 = torch.sum(torch.abs(A.mtimes(X, 0) - K) ** 2).item()
                metrics['loss1'].append(loss1)
                metrics['loss2'].append(loss2)
                metrics['loss3'].append(loss3)
                metrics['loss'].append(loss1 * 0.5 + loss2 * regL + loss3 * regS)
            if gt is not None:
                metrics['art_nrmse'].append(
                    nRMSE(torch.angle(X)[mask == 1], torch.angle(gt)[mask== 1], use_torch=True).item())
                metrics['vein_nrmse'].append(
                    nRMSE(torch.angle(X)[mask == 2], torch.angle(gt)[mask == 2], use_torch=True).item())
                metrics['psnr'].append(PSNR(torch.abs(X) * regfactor, torch.abs(gt), use_torch=True).item())
                metrics['ssim'].append(SSIM(torch.abs(X.unsqueeze(0).unsqueeze(0)) * regfactor, torch.abs(gt.unsqueeze(0).unsqueeze(0)), use_torch=True))
                metrics['nrmse'].append(nRMSE(torch.abs(X) * regfactor, torch.abs(gt), use_torch=True).item())
    elif method == 'POGM':
        tp = 1
        gp = 1
        X = A.mtimes(K, 1)
        L, L_, L_p, Lh, Lhp = X.clone(), X.clone(), X.clone(), X.clone(), X.clone()
        S, S_, S_p, Sh, Shp = torch.zeros_like(X).to(device), torch.zeros_like(X).to(device), torch.zeros_like(X).to(
            device), torch.zeros_like(X).to(device), torch.zeros_like(X).to(device)
        for i in loop:
            Lh = X - S
            Sh = X - L
            t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
            L_ = Lh + (tp - 1) / t * (Lh - Lhp) + tp / t * (Lh - L) + (tp - 1) / (gp * t) * 1 / Lc * (L_p - L)
            S_ = Sh + (tp - 1) / t * (Sh - Shp) + tp / t * (Sh - S) + (tp - 1) / (gp * t) * 1 / Lc * (S_p - S)
            g = 1 / Lc * (1 + (tp - 1) / t + tp / t)
            if llr:
                L, loss2 = SVT_LLR(L_, regL * g, blk)
            else:
                L, loss2 = SVT(L_, regL * g)
            S, loss3 = Sparse(S_, regS * g)
            axb = A.mtimes(L + S, 0) - K
            X = L + S - 1 / Lc * A.mtimes(axb, 1)
            tp = t
            gp = g
            L_p = L_
            S_p = S_
            Lhp = Lh
            Shp = Sh
            if save_loss:
                loss1 = torch.sum(torch.abs(A.mtimes(X, 0) - K) ** 2).item()
                metrics['loss1'].append(loss1)
                metrics['loss2'].append(loss2)
                metrics['loss3'].append(loss3)
                metrics['loss'].append(loss1 * 0.5 + loss2 * regL + loss3 * regS)
            if gt is not None:
                metrics['art_nrmse'].append(
                    nRMSE(torch.angle(X)[mask == 1], torch.angle(gt)[mask== 1], use_torch=True).item())
                metrics['vein_nrmse'].append(
                    nRMSE(torch.angle(X)[mask == 2], torch.angle(gt)[mask == 2], use_torch=True).item())
                metrics['psnr'].append(PSNR(torch.abs(X) * regfactor, torch.abs(gt), use_torch=True).item())
                metrics['ssim'].append(SSIM(torch.abs(X.unsqueeze(0).unsqueeze(0)) * regfactor, torch.abs(gt.unsqueeze(0).unsqueeze(0)), use_torch=True))
                metrics['nrmse'].append(nRMSE(torch.abs(X) * regfactor, torch.abs(gt), use_torch=True).item())
    return X * regfactor, metrics

def ReconHAAR(A, K, method, it, reg_list, L, regfactor=1, device='cuda', gt=None, save_loss=False, mask=None):
    '''
    :param A: Operator consists of under-sampling mask, fourier transform and coil sensitivity maps
    :param K: Acquired under-sampled k-space with shape of Nt, Nc, FE, PE, SPE
    :param method: Iteration method (ISTA, FISTA, POGM)
    :param it: Iteration number
    :param reg_list: regularization for each dimension
    :param L: Lipschitz constant
    :param regfactor: scale factor for denormalization
    :param device: cuda or cpu
    :param gt: ground truth image, if it is not None, PSNR, SSIM, and nRMSE will be calculated and stored in metrics
    :param save_loss: if save_loss is True, loss will be calculated and stored in metrics
    :param mask: mask of flow for calculating art_nrmse and vein_nrmse
    :return: X:reconstructed image, metrics:dictionary
    '''
    Nt, Nc, FE, PE, SPE = K.shape
    loop = tqdm.tqdm(range(1, it + 1), total=it)
    metrics = {}
    if save_loss:
        metrics['loss'] = []
        metrics['loss1'] = []
        metrics['loss2'] = []
    if gt is not None:
        metrics['psnr'] = []
        metrics['ssim'] = []
        metrics['nrmse'] = []
        metrics['art_nrmse'] = []
        metrics['vein_nrmse'] = []
    if method == 'ISTA':
        X = A.mtimes(K, 1)
        for i in loop:
            axb = A.mtimes(X, 0) - K
            X = X - 1 / L * A.mtimes(axb, 1)
            X, loss2 = ST_HAAR(X, reg_list / L, device)
            if save_loss:
                loss1 = torch.sum(torch.abs(A.mtimes(X, 0) - K) ** 2).item()
                metrics['loss1'].append(loss1)
                metrics['loss2'].append(loss2)
                metrics['loss'].append(loss1 * 0.5 + loss2)
            if gt is not None:
                metrics['art_nrmse'].append(
                    nRMSE(torch.angle(X)[mask == 1], torch.angle(gt)[mask== 1], use_torch=True).item())
                metrics['vein_nrmse'].append(
                    nRMSE(torch.angle(X)[mask == 2], torch.angle(gt)[mask == 2], use_torch=True).item())
                metrics['psnr'].append(PSNR(torch.abs(X) * regfactor, torch.abs(gt), use_torch=True).item())
                metrics['ssim'].append(SSIM(torch.abs(X.unsqueeze(0).unsqueeze(0)) * regfactor, torch.abs(gt.unsqueeze(0).unsqueeze(0)), use_torch=True))
                metrics['nrmse'].append(nRMSE(torch.abs(X) * regfactor, torch.abs(gt), use_torch=True).item())
    elif method == 'FISTA':
        tp = 1
        Xp = A.mtimes(K, 1)
        Y = Xp.clone()
        for i in loop:
            t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
            axb = A.mtimes(Y, 0) - K
            Y = Y - 1 / L * A.mtimes(axb, 1)
            X, loss2 = ST_HAAR(Y, reg_list / L, device)
            Y = X + (tp - 1) / t * (X - Xp)
            Xp = X
            tp = t
            if save_loss:
                loss1 = torch.sum(torch.abs(A.mtimes(X, 0) - K) ** 2).item()
                metrics['loss1'].append(loss1)
                metrics['loss2'].append(loss2)
                metrics['loss'].append(loss1 * 0.5 + loss2)
            if gt is not None:
                metrics['art_nrmse'].append(
                    nRMSE(torch.angle(X)[mask == 1], torch.angle(gt)[mask== 1], use_torch=True).item())
                metrics['vein_nrmse'].append(
                    nRMSE(torch.angle(X)[mask == 2], torch.angle(gt)[mask == 2], use_torch=True).item())
                metrics['psnr'].append(PSNR(torch.abs(X) * regfactor, torch.abs(gt), use_torch=True).item())
                metrics['ssim'].append(SSIM(torch.abs(X.unsqueeze(0).unsqueeze(0)) * regfactor, torch.abs(gt.unsqueeze(0).unsqueeze(0)), use_torch=True))
                metrics['nrmse'].append(nRMSE(torch.abs(X) * regfactor, torch.abs(gt), use_torch=True).item())
    elif method == 'POGM':
        tp = 1
        gp = 1
        Xp = A.mtimes(K, 1)
        X, Y, Z, Yp, Zp = Xp.clone(), Xp.clone(), Xp.clone(), Xp.clone(), Xp.clone()
        for i in loop:
            t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
            g = 1 / L * (2 * tp + t - 1) / t
            axb = A.mtimes(X, 0) - K
            Y = X - 1 / L * A.mtimes(axb, 1)
            Z = Y + (tp - 1) / t * (Y - Yp) + tp / t * (Y - Xp) + (tp - 1) / (L * gp * t) * (Zp - Xp)
            X, loss2 = ST_HAAR(Z,  reg_list * g , device)
            Xp = X
            Yp = Y
            Zp = Z
            tp = t
            gp = g
            if save_loss:
                loss1 = torch.sum(torch.abs(A.mtimes(X, 0) - K) ** 2).item()
                metrics['loss1'].append(loss1)
                metrics['loss2'].append(loss2)
                metrics['loss'].append(loss1 * 0.5 + loss2)
            if gt is not None:
                metrics['art_nrmse'].append(
                    nRMSE(torch.angle(X)[mask == 1], torch.angle(gt)[mask== 1], use_torch=True).item())
                metrics['vein_nrmse'].append(
                    nRMSE(torch.angle(X)[mask == 2], torch.angle(gt)[mask == 2], use_torch=True).item())
                metrics['psnr'].append(PSNR(torch.abs(X) * regfactor, torch.abs(gt), use_torch=True).item())
                metrics['ssim'].append(SSIM(torch.abs(X.unsqueeze(0).unsqueeze(0)) * regfactor, torch.abs(gt.unsqueeze(0).unsqueeze(0)), use_torch=True))
                metrics['nrmse'].append(nRMSE(torch.abs(X) * regfactor, torch.abs(gt), use_torch=True).item())
    return X * regfactor, metrics