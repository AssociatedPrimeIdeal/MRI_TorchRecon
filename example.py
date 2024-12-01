from utils import *
from wrapper import ReconLLR, ReconLplusS, ReconHAAR
import matplotlib
import matplotlib.pyplot
import imageio
torch.set_num_threads(os.cpu_count())
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def test():
    data = np.load(data_path, allow_pickle=True).item()
    K = data['kspc'][0, 0]
    csm = data['sens']
    mask = data['mask'][0, 0]
    usmask_8, usmask_12, usmask_16, usmask_20, usmask_24 = data['usmask_8'][0, 0], data['usmask_12'][0, 0], data['usmask_16'][0, 0], data[
        'usmask_20'][0, 0], data['usmask_24'][0, 0]
    device = 'cuda:0'
    it = 150
    itmethod = 'POGM'
    Nt, Nc, FE, PE, SPE = K.shape
    print("Kspc shape:", K.shape)
    sos = np.sqrt(np.sum(np.abs(csm) ** 2, axis=0, keepdims=True)) + 1e-11
    csm /= sos
    csm = torch.as_tensor(np.ascontiguousarray(csm)).to(torch.complex64).to(device)
    img_gt = torch.sum(k2i_torch(torch.as_tensor(np.ascontiguousarray(K)).to(torch.complex64).to(device),
                                 ax=[-3, -2, -1]) * torch.conj(csm), 1).to(torch.complex64).to(device)
    mask = torch.as_tensor(np.ascontiguousarray(mask)).to(torch.float32).to(device)

    usmask_use = usmask_24
    us_rate = 1 / np.mean(usmask_use)
    print('US rate: ', us_rate)

    SNR = 30
    stdev = 1 / SNR
    noise = stdev * (np.random.randn(*K.shape) + 1j * np.random.randn(*K.shape))
    Knoise = K + noise
    img_noise = torch.sum(
        k2i_torch(torch.as_tensor(np.ascontiguousarray(Knoise)).to(torch.complex64).to(device),
                  ax=[-3, -2, -1]) * torch.conj(csm), 1).to(torch.complex64).to(device)
    Knoise *= usmask_use

    for i, recon_method in enumerate(['LLR', 'L+S', 'HAAR']):
        if recon_method == 'LLR':
            reg = 0.002
            blk = 8
            L = 2
            llr = True
            param_string = param_string = 'US Rate:{us_rate:.2f}_SNR:{SNR:d}_Iter Num:{it:d}_Reg:{reg:0.4f}_Block Size:{blk:d}_Lip Constant:{L:d}_Iter Method:'.format(
                us_rate=us_rate, SNR=SNR, it=it, reg=reg, blk=blk, L=L) + itmethod
        elif recon_method == 'L+S':
            regL = 0.004
            regS = 0.064
            blk = 8
            L = 2
            llr = True
            param_string = 'US Rate:{us_rate:.2f}_SNR:{SNR:d}_Iter Num:{it:d}_RegL:{regL:0.4f}_RegS:{regS:0.4f}_Block Size:{blk:d}_Lip Constant:{L:d}_Iter Method:'.format(
                us_rate=us_rate, SNR=SNR, it=it, regL=regL, regS=regS, blk=blk, L=L) + itmethod
        elif recon_method == 'HAAR':
            regt = 0.0028
            regs = 0.0002
            reg_list = np.array([regt, regt, regt, regt, regt, regt, regt, regt, regs, regs, regs, regs, regs, regs, regs, regs])
            L = 2
            param_string = 'US Rate:{us_rate:.2f}_SNR:{SNR:d}_Iter Num:{it:d}_RegT:{regt:0.4f}_RegS:{regs:0.4f}_Lip Constant:{L:d}_Iter Method:'.format(
                us_rate=us_rate, SNR=SNR, it=it, regt=regt, regs=regs, L=L) + itmethod
        print("Param: ", param_string)
        st = time.time()
        K = torch.as_tensor(np.ascontiguousarray(Knoise)).to(torch.complex64).to(device)
        us_mask = (torch.abs(K[:, 0:1, FE // 2:FE // 2 + 1]) > 0).to(torch.float32).to(device)
        rcomb = torch.sum(k2i_torch(K, ax=[-3, -2, -1]) * torch.conj(csm), 1)
        regFactor = torch.max(torch.abs(rcomb))
        K /= regFactor
        A = Eop(csm, us_mask)
        if recon_method == 'LLR':
            X, metrics = ReconLLR(A, K, itmethod, llr, it, reg, blk, L, regfactor=regFactor.item(),device=device,gt=img_gt,save_loss=True, mask=mask)
        elif recon_method == 'HAAR':
            X, metrics = ReconHAAR(A, K, itmethod, it, reg_list, L, regfactor=regFactor.item(),device=device,gt=img_gt,save_loss=True, mask=mask)
        elif recon_method == 'L+S':
            X, metrics = ReconLplusS(A, K, itmethod, llr, it, regL, regS, blk, L, regfactor=regFactor.item(),device=device,gt=img_gt,save_loss=True, mask=mask)
        if 'cuda' in device:
            X = X.cpu().numpy()
        duration = time.time() - st
        print("TIME COMSUMING:{:.2f}s".format(duration))
        save_data = {"X":X, 'metrics':metrics}
        np.save(save_path + param_string + '.npy', save_data)

def draw():
    llrdata = np.load(save_path + 'US Rate:24.00_SNR:30_Iter Num:150_Reg:0.0020_Block Size:8_Lip Constant:2_Iter Method:POGM.npy', allow_pickle=True).item()
    lsdata = np.load(save_path + 'US Rate:24.00_SNR:30_Iter Num:150_RegL:0.0040_RegS:0.0640_Block Size:8_Lip Constant:2_Iter Method:POGM.npy', allow_pickle=True).item()
    haardata = np.load(save_path + 'US Rate:24.00_SNR:30_Iter Num:150_RegT:0.0028_RegS:0.0002_Lip Constant:2_Iter Method:POGM.npy', allow_pickle=True).item()
    data = np.load(data_path, allow_pickle=True).item()
    K = data['kspc'][0, 0]
    csm = data['sens']
    mask = data['mask'][0, 0]
    usmask_8, usmask_12, usmask_16, usmask_20, usmask_24 = data['usmask_8'][0, 0], data['usmask_12'][0, 0], data['usmask_16'][0, 0], data[
        'usmask_20'][0, 0], data['usmask_24'][0, 0]
    device = 'cuda:0'
    it = 150
    itmethod = 'POGM'
    Nt, Nc, FE, PE, SPE = K.shape
    print("Kspc shape:", K.shape)
    sos = np.sqrt(np.sum(np.abs(csm) ** 2, axis=0, keepdims=True)) + 1e-11
    csm /= sos
    csm = torch.as_tensor(np.ascontiguousarray(csm)).to(torch.complex64).to(device)
    img_gt = torch.sum(k2i_torch(torch.as_tensor(np.ascontiguousarray(K)).to(torch.complex64).to(device),
                                    ax=[-3, -2, -1]) * torch.conj(csm), 1).cpu().numpy()

    usmask_use = usmask_24
    us_rate = 1 / np.mean(usmask_use)
    print('US rate: ', us_rate)

    SNR = 30
    stdev = 1 / SNR
    noise = stdev * (np.random.randn(*K.shape) + 1j * np.random.randn(*K.shape))
    Knoise = K + noise
    img_noise = torch.sum(
        k2i_torch(torch.as_tensor(np.ascontiguousarray(Knoise)).to(torch.complex64).to(device),
                    ax=[-3, -2, -1]) * torch.conj(csm), 1).cpu().numpy()

    imgnorm = matplotlib.colors.Normalize(0, 1)
    imgdiffnrom = matplotlib.colors.Normalize(0, 0.3)
    phasenorm = matplotlib.colors.Normalize(0, np.pi)
    phasediffnorm = matplotlib.colors.Normalize(0, np.pi/3)
    row = 4
    col = 5
    show_T = 10
    show_SPE = 30
    Nt, FE, PE, SPE = llrdata['X'].shape
    venc = 150
    a_mask = mask==1
    v_mask = mask==2
    flow_a_gt = np.angle(img_gt) * a_mask
    flow_v_gt = np.angle(img_gt)  * v_mask
    flow_a_no = np.angle(img_noise) * a_mask
    flow_v_no = np.angle(img_noise)  * v_mask

    method_list = ['LLR', 'L+S', 'HAAR']
    color_list = ['blue', 'green', 'purple']

    writer = imageio.get_writer(save_path + 'output.gif', mode='I', duration=1, loop=0)
    for show_T in range(Nt):
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(row, col)
        gt_show = np.abs(img_gt[show_T, :, :, show_SPE])
        noise_show = np.abs(img_noise[show_T, :, :, show_SPE])
        gtph_show = np.angle(img_gt[show_T, :, :, show_SPE])
        noiseph_show = np.angle(img_noise[show_T, :, :, show_SPE])
        ax = fig.add_subplot(gs[0, 0])
        plt.imshow(gt_show,cmap='gray',origin='lower',norm=imgnorm)
        plt.title('Ground Truth', fontweight='bold')
        ax.text(-0.1, 0.5, "Magnitude", va='center', ha='right', transform=ax.transAxes, fontweight='bold', rotation=90, fontsize=12)
        plt.axis('off')    
        plt.colorbar()

        ax = fig.add_subplot(gs[0, 1])
        plt.imshow(noise_show,cmap='gray',origin='lower',norm=imgnorm)
        plt.title('Noisy Image', fontweight='bold')
        plt.axis('off')
        plt.colorbar()
        ax = fig.add_subplot(gs[1, 0])
        plt.imshow(gtph_show,cmap='gray',origin='lower',norm=phasenorm)
        ax.text(-0.1, 0.5, "Phase", va='center', ha='right', transform=ax.transAxes, fontweight='bold', rotation=90, fontsize=12)
        plt.axis('off')    
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_ticks([0, np.pi/2, np.pi])
        cbar.set_ticklabels([0 , f"$\pi$ / {2}" ,f"$\pi$" ])
        ax = fig.add_subplot(gs[1, 1])
        plt.imshow(noiseph_show,cmap='gray',origin='lower',norm=phasenorm) 
        plt.axis('off')
        plt.colorbar()
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_ticks([0, np.pi/2, np.pi])
        cbar.set_ticklabels([0 , f"$\pi$ / {2}" ,f"$\pi$" ])
        ax2 = fig.add_subplot(gs[2, :2])
        ax2.plot(np.sum(flow_a_gt, axis=(-1,-2,-3)) / np.sum(a_mask[0]), label='Ground Truth', color='red')
        ax2.set_title('Artery Phase', fontweight='bold')
        ax3 = fig.add_subplot(gs[3, :2])
        ax3.plot(np.sum(flow_v_gt, axis=(-1,-2,-3)) / np.sum(v_mask[0]), label='Ground Truth', color='red')
        ax3.set_title('Vein Phase', fontweight='bold')
        ax3.legend()

        for i, X in enumerate([llrdata['X'], lsdata['X'], haardata['X']]):
            x_show = np.abs(X[show_T, :, :, show_SPE])
            xph_show = np.angle(X[show_T, :, :, show_SPE])
            ax = fig.add_subplot(gs[0, 2 + i])
            plt.imshow(x_show,cmap='gray',origin='lower',norm=imgnorm) 
            plt.title(method_list[i], fontweight='bold')
            plt.axis('off')
            ax = fig.add_subplot(gs[1, 2 + i])
            plt.imshow(xph_show,cmap='gray',origin='lower',norm=phasenorm) 
            plt.axis('off')
            ax = fig.add_subplot(gs[2, 2 + i])
            plt.imshow(np.abs(x_show-gt_show),cmap='gray',origin='lower',norm=imgdiffnrom)
            plt.colorbar()
            if i == 0:
                ax.text(-0.1, 0.5, "Mag Difference", va='center', ha='right', transform=ax.transAxes, fontweight='bold', rotation=90, fontsize=12)
            plt.axis('off') 
            ax = fig.add_subplot(gs[3, 2 + i])
            plt.imshow(np.abs(xph_show-gtph_show),cmap='gray',origin='lower',norm=phasediffnorm) 
            plt.colorbar()
            if i == 0:
                ax.text(-0.1, 0.5, "Phase Difference", va='center', ha='right', transform=ax.transAxes, fontweight='bold', rotation=90, fontsize=12)
            plt.axis('off')

            flow_a_x = np.angle(X) * a_mask
            flow_v_x = np.angle(X) * v_mask
            ax2.plot(np.sum(flow_a_x, axis=(-1,-2,-3)) / np.sum(a_mask[0]), label=method_list[i], color=color_list[i])
            ax2.legend()
            ax3.plot(np.sum(flow_v_x, axis=(-1,-2,-3)) / np.sum(v_mask[0]), label=method_list[i], color=color_list[i])
            ax3.legend()
        plt.tight_layout()
        plt.savefig(save_path + str(show_T) + '.png')
        plt.close()
        writer.append_data(imageio.imread(save_path + str(show_T) + '.png'))
        # os.remove(save_path + str(show_T) + '.png')

if __name__ == '__main__':
    data_path = '/nas-data/xcat2/res2/data_xcat_res2d5.npy'
    save_path = '/nas-data/xcat2/temp/'
    # test()
    draw()