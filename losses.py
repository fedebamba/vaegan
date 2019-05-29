import torchimport torch.nn.functional as Fdef vae_loss_function(x, z, mean, logvar):    bce = F.binary_cross_entropy(x, z.view(-1, 784), reduction='sum') #    kld = -.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())    # kld = -.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())    return (bce + kld) # torch.log(bce + kld)def boundary_seeking_loss(y_pred, y_true):    return 0.5 * torch.mean((torch.log(y_pred) - torch.log(1 - y_pred)) ** 2)# =====================================================================================================================def enc_dec_dis_losses(o_layer, r_layer, true_labels, fake_labels,  logvar, mean, lambda_mse):    mse = torch.sum(.5 * ((r_layer - o_layer) ** 2), 1)  # encoder loss = kl+mse... decoder loss = mse+    kld = -.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())    bce_dis_original_value = -torch.log(true_labels + 1e-4) # F.binary_cross_entropy(true_labels, 1) reduction is sum    bce_dis_sampled_value = -torch.log(1-fake_labels + 1e-4) # F.binary_cross_entropy(fake_labels, 0)    loss_encoder = torch.sum(kld) + torch.sum(mse)    loss_discriminator = torch.sum(bce_dis_original_value) + torch.sum(bce_dis_sampled_value)    # decoder loss --> KL + (lambda*mse_lth + (1-lambda)*GAN) (lambda just for dec)    loss_decoder = torch.sum(lambda_mse * mse) - (1.0 - lambda_mse) * loss_discriminator    return (loss_encoder, loss_decoder, loss_discriminator), (torch.sum(bce_dis_original_value), torch.sum(bce_dis_sampled_value))