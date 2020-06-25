import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from data_loading import TTSDataset, BatchSampler, collate_batch
from argparse import ArgumentParser
from Text2Mel import Text2Mel
from SSRN import SSRN
from audio_processing import N_MELS, N_MAGS
import os
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime


def guided_attention_loss(attention):
    """
    computes the attention loss, Latt(A) = E_nt[A_nt W_nt], where W_nt = 1 − exp{−(n/N − t/T)² /2g²}. We set g = 0.2.
    :param attention: the attention tensor (Batch, Num_chars, Time_dim_size)
    :return: the value of the attention loss, averaged over the batch dimention
    """
    batch, num_chars, time_dim_size = attention.shape
    weight = torch.from_numpy(1 - np.exp(-(np.arange(num_chars).reshape((1, -1, 1)) / num_chars
                                           - np.arange(time_dim_size).reshape((1, 1, -1)) / time_dim_size)**2
                                         / (2 * .2**2))).to(attention.device)
    return torch.mean(attention * weight)


def noam_scheduler(init_lr, nbr_batches, warmup_steps=4000.0):
    """Noam lr scheduler"""
    return init_lr * warmup_steps**0.5 * min(nbr_batches * warmup_steps**-1.5, nbr_batches**-0.5)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="training script for Text2Mel or SSRN, saves checkpoints every 150k samples"
                    "+ at the end of training if a max_num_samples_to_train_on is specified")
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--net', type=str, choices=["Text2Mel", "SSRN"], required=True)
    parser.add_argument('--train_data_folder', type=str, required=True)
    parser.add_argument('--max_num_samples_to_train_on', type=int)
    parser.add_argument('--base_model', type=str,
                        help="either a pytorch file to resume training from, a folder containing tf checkpoints to do "
                             "the same or nothing to start from scratch")
    parser.add_argument('--use_apex_fast_mixed_precision', action="store_true",
                        help="uses nvidia apex to perform mixed precision training (O2 flag)")
    parser.add_argument('--use_noam_scheduler', action="store_true", help="uses Noam LR scheduler as in tf "
                                                                          "implementation")
    # https://medium.com/the-artificial-impostor/use-nvidia-apex-for-easy-mixed-precision-training-in-pytorch-46841c6eed8c
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--num_t2m_hidden_units', type=int, default=256)
    parser.add_argument('--num_ssrn_hidden_units', type=int, default=512)
    parser.add_argument("--dropout_rate", type=float, default=.05)
    parser.add_argument('--vocab', type=str,
                        help="authorized text token, the first must always stand for padding and the second for end of "
                             "sentence, if no uppercase letter is present in vocab, the input text will be lowercased",
                        default=u'''␀␃ !"',-.:;?AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZzàâæçèéêëîïôùûœ–’''')
    parser.add_argument('--max_num_chars', type=int, default=250,
                        help="training data with more chars than that will be removed, so that 2 long training samples "
                             "do not prevent from using a good batch size")
    parser.add_argument('--max_num_mag_time_frames', type=int, default=800,
                        help="training data whose mag transform has more than these num of time frames will be removed,"
                             " so that 2 long training samples do not prevent from using a good batch size")
    parser.add_argument('--num_workers', type=int, default=8, help="number of data loading workers")
    parser.add_argument('--samples_before_ckpt', type=int, default=150_000,
                        help="number of samples that the model will treat before being ckpted, does not include "
                             "samples treated previously if we are loading a model")
    args = parser.parse_args()
    print("\n")
    print(args)

    checkpoint_dir = os.path.join('checkpoints', args.name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = Text2Mel(len(args.vocab), args.embed_size, args.num_t2m_hidden_units, N_MELS, args.dropout_rate) \
        if args.net == "Text2Mel" else SSRN(N_MELS, N_MAGS, args.num_ssrn_hidden_units, args.dropout_rate)

    dataset = TTSDataset(args.train_data_folder, args.net == "Text2Mel", args.vocab, args.max_num_chars,
                         args.max_num_mag_time_frames)
    batch_sampler = BatchSampler(dataset, args.batch_size)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers,
                             collate_fn=lambda x: collate_batch(x, args.net == "Text2Mel"))

    optimizer = optim.Adam([v for v in model.parameters() if v.requires_grad], lr=args.lr, betas=(.5, .9), eps=1e-6)

    # reload checkpoint parameters
    epoch = 0
    num_samples_treated = 0
    num_batches_treated = 0
    if args.base_model is not None:
        if os.path.isfile(args.base_model):
            checkpoint = torch.load(args.base_model)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch'] + 1  # we start a new epoch
            num_samples_treated = checkpoint['num_samples_treated']
            num_batches_treated = checkpoint['num_batches_treated']
        else:  # tf model
            from load_tf_models import load_ssrn_from_tf, load_t2m_from_tf  # imported here so that installing tf is
            # not mandatory
            load_t2m_from_tf(model, args.base_model) if args.net == "Text2Mel" else \
                load_ssrn_from_tf(model, args.base_model)

    max_num_samples_to_train_on = num_samples_treated + args.max_num_samples_to_train_on \
        if args.max_num_samples_to_train_on is not None else 1e10  # 1e10 in case we
    # want to loop "indefinitely"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    if args.use_apex_fast_mixed_precision:
        from apex import amp
        APEX_AVAILABLE = True

        model, optimizer = amp.initialize(model, optimizer, opt_level="O2",
                                          keep_batchnorm_fp32=True, loss_scale="dynamic")
        if args.base_model is not None and os.path.isfile(args.base_model):  # load amp saved state
            amp.load_state_dict(checkpoint['amp'])
    else:
        APEX_AVAILABLE = False

    if args.use_noam_scheduler:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda x: noam_scheduler(args.lr, x), last_epoch=num_batches_treated)
    else:
        scheduler = None

    while True:  # epoch
        avg_loss, num_phrases_considered = 0, 0

        # switch to train mode
        model.train()

        begin = time.time()

        pbar = tqdm(data_loader)
        for batch in pbar:
            batch = [item.to(device) for item in batch]
            batch_size = batch[0].shape[0]

            # compute output
            if args.net == "Text2Mel":
                text, in_mel, target_mel = batch
                pred_mel, logits, attention = model(text, in_mel)
                loss = F.l1_loss(pred_mel, target_mel) + \
                       F.binary_cross_entropy_with_logits(logits, target_mel) + guided_attention_loss(attention)
            else:
                mel, mag = batch
                pred_mags, logits = model(mel)
                loss = F.l1_loss(pred_mags, mag) + F.binary_cross_entropy_with_logits(logits, mag)
            loss_val = loss.item()

            new_num_phrases_considered = num_phrases_considered + batch_size
            avg_loss = (avg_loss * num_phrases_considered + loss_val * batch_size) / new_num_phrases_considered
            num_phrases_considered = new_num_phrases_considered

            # compute gradient and do SGD step
            optimizer.zero_grad()

            if APEX_AVAILABLE:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_value_(amp.master_params(optimizer), 1)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 1)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # increments global step and save data if needed be
            new_num_samples_treated = num_samples_treated + batch[0].shape[0]
            num_batches_treated += 1

            if new_num_samples_treated > max_num_samples_to_train_on:
                state_dict = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'num_samples_treated': new_num_samples_treated,
                    'amp': amp.state_dict() if APEX_AVAILABLE else None,
                    'num_batches_treated': num_batches_treated
                }
                torch.save(state_dict, os.path.join(checkpoint_dir, f'{args.net}-{new_num_samples_treated}.ckpt'))
                print("training ended")
                exit()

            if (num_samples_treated // args.samples_before_ckpt) != \
                    (new_num_samples_treated // args.samples_before_ckpt):
                state_dict = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'num_samples_treated': new_num_samples_treated,
                    'amp': amp.state_dict() if APEX_AVAILABLE else None,
                    'num_batches_treated': num_batches_treated
                }
                torch.save(state_dict, os.path.join(checkpoint_dir, f'{args.net}-{new_num_samples_treated}.ckpt'))

            pbar.set_description(f"{new_num_samples_treated :.2e} samples seen - Loss {loss_val:.4f} (avg: {avg_loss:.4f})")
            num_samples_treated = new_num_samples_treated

        print(f'{datetime.now()} - Epoch: {epoch} - Elapsed seconds {time.time() - begin:.1f} \t Loss {avg_loss:.3f}')
        epoch += 1
