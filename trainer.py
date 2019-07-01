import torch
import time

import losses

class VAE_Trainer:
    def __init__(self, model, dataloader, loss, optimizer, device, testloader):
        self.model = model
        self.dataLoader = dataloader
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.testLoader = testloader

    def single_pass(self):
        self.model.eval()

        array = torch.Tensor().to("cuda")
        targets = torch.Tensor().long().to("cuda")

        with torch.no_grad():
            for batch_index, (data, target) in enumerate(self.testLoader):
                data = data.to(self.device)
                latent_repr = self.model.encode(data)

                array = torch.cat((array, latent_repr[0] ), 0)
                targets = torch.cat((targets, target.to("cuda")), 0)
                print(array.size(), end="\r")
        print("")
        return array, targets

    def train(self, epoch):
        self.model.train()
        train_loss = 0

        start_time = time.time()
        for batch_index, (data, target) in enumerate(self.dataLoader):
            data = data.to(self.device)
            self.optimizer.zero_grad()

            result, mean, logvar = self.model(data)
            loss = self.loss(result, data, mean, logvar)
            loss.backward()
            train_loss += loss.item()

            self.optimizer.step()
            print("Train epoch {0} ({1:.1f}%) >> loss: {2:.2f} \r".format(epoch, 100*(batch_index/(len(self.dataLoader.dataset)/self.dataLoader.batch_size)), loss/self.dataLoader.batch_size), end="\r")
        print("Train epoch {0} (--.-%) >> avg.loss: {1:.2f} {2:^20} ".format(epoch, train_loss / len(self.dataLoader.dataset), "elapsed time: {0:.2f} s.".format(time.time() - start_time)))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for batch_index, (data, target) in enumerate(self.testLoader):
                data = data.to(self.device)
                result, mean, logvar = self.model(data)
                test_loss += self.loss(result, data, mean, logvar).item()

        test_loss /= len(self.testLoader.dataset)
        print("Test: accuracy : {:.3f}".format(test_loss))


class GAN_trainer:
    def __init__(self, d_model, g_model, dataloader, d_optimizer, g_optimizer, device):
        self.d_model = d_model
        self.g_model = g_model
        self.dataLoader = dataloader
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.device = device

        self.g_loss = None
        self.d_loss = torch.nn.BCELoss()

    def train(self, epoch):
        train_loss = [0,0]
        start_time = time.time()
        for batch_index, (data, target) in enumerate(self.dataLoader):
            data = data.to(self.device)
            valid = torch.Tensor(len(data)).fill_(1).to(self.device)
            fake = torch.Tensor(len(data)).fill_(0).to(self.device)

            # update G
            self.g_model.zero_grad()
            self.g_optimizer.zero_grad()

            noise = torch.randn(32, 64).to("cuda") # batch-size, latent-space
            fake_data = self.g_model.decode_image(noise)

            g_loss = self.d_model(fake_data).sum()  # D has updated weights
            g_loss.backward()
            self.g_optimizer.step()

            # update D
            self.d_optimizer.zero_grad()
            self.d_model.zero_grad()

            true_loss = self.d_loss(self.d_model(data), valid).sum()
            fake_loss = self.d_loss(self.d_model(fake_data.detach()), fake).sum()
            train_loss = [train_loss[0] + true_loss, train_loss[1] + fake_loss]
            d_loss = fake_loss + true_loss
            d_loss.backward()
            self.d_optimizer.step()

            print("Train epoch {0} ({1:.1f}%) >> loss: {2:.2f} ({3:.2f} /{4:.2f} )\r".format(epoch, 100*(batch_index/(len(self.dataLoader.dataset)/self.dataLoader.batch_size)), d_loss, fake_loss, true_loss), end="\r")
        print("Train epoch {0} (--.-%) >> avg.loss: {1:.2f} ({3:.2f} /{4:.2f} ){2:^20} ".format(epoch,(train_loss[1] - train_loss[0]) / len(self.dataLoader.dataset), "elapsed time: {0:.2f} s.".format(time.time() - start_time), train_loss[1], train_loss[0]))
    def test(self, epoch):
        pass

#todo
class VAE_GAN_trainer:
    def __init__(self, d_model, g_model, dataloader, d_optimizer, g_optimizer, device, g_loss, testLoader=None):
        self.d_model = d_model
        self.g_model = g_model
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss = torch.nn.BCELoss()
        self.g_loss = g_loss

        self.dataLoader = dataloader
        self.testLoader = testLoader
        self.device = device

        self.margin = .45
        self.eq = .5


    def single_pass(self):
        self.g_model.eval()

        array = torch.Tensor().to("cuda")
        targets = torch.Tensor().long().to("cuda")

        with torch.no_grad():
            for batch_index, (data, target) in enumerate(self.testLoader):
                data = data.to(self.device)
                latent_repr = self.g_model.encode(data)

                array = torch.cat((array, latent_repr[0] ), 0)
                targets = torch.cat((targets, target.to("cuda")), 0)
                print(array.size(), end="\r")
        print("")
        return array, targets


    def train_the_G(self, batch_data, target, gan_weight=500, vae_weight=1):
        self.g_model.zero_grad()
        self.g_optimizer.zero_grad()

        # vae loss
        result, mean, logvar = self.g_model(batch_data)
        vae_loss = self.g_loss(result, batch_data, mean, logvar)

        # gan loss
        noise = torch.randn(32, 20).to("cuda")  # batch-size, latent-space
        fake_data = self.g_model.decode_image(noise)
        gan_loss = losses.boundary_seeking_loss(self.d_model(fake_data.detach()),target)

        vae_loss *= vae_weight
        gan_loss *= gan_weight

        g_loss = vae_loss + gan_loss
        g_loss.backward()
        self.g_optimizer.step()
        return (vae_loss, gan_loss), fake_data

    def train_the_D(self, true_batch_data, fake_batch_data, target_true, target_fake):
        self.d_optimizer.zero_grad()
        self.d_model.zero_grad()

        true_loss = self.d_loss(self.d_model(true_batch_data), target_true).sum()
        fake_loss = self.d_loss(self.d_model(fake_batch_data.detach()), target_fake).sum()
        d_loss = fake_loss + true_loss
        d_loss.backward()
        self.d_optimizer.step()
        return fake_loss, true_loss

    def train(self, epoch):
        train_loss = [0, 0, 0, 0]
        start_time = time.time()
        for batch_index, (data, target) in enumerate(self.dataLoader):
            print(data.size())
            # train the G
            data = data.to(self.device)
            mean, logvar = self.g_model.encode(data)
            z = self.g_model.gaussian_sampling(mean, logvar)
            x2 = self.g_model.decode_image(z)

            # still train the G
            _, fake_hidden = self.d_model(x2, hidden=True)
            _, true_hidden = self.d_model(data, hidden=True)

            # train the D
            p = (torch.randn(self.dataLoader.batch_size, 20)).to(self.device)
            xp = self.g_model.decode_image(p)
            true_labels = self.d_model(data)
            fake_labels = self.d_model(xp)

            (loss_encoder, loss_decoder, loss_discriminator), (o,f) = losses.enc_dec_dis_losses(o_layer=true_hidden, r_layer=fake_hidden, true_labels=true_labels, fake_labels=fake_labels, logvar=logvar, mean=mean, lambda_mse=.0005)

            train_dis = True
            train_dec = True
            # if o < self.margin + self.eq or o < self.margin + self.eq:
            # if o < self.margin - self.eq or f < self.margin - self.eq:
            # if o < ( self.eq - self.margin)*self.dataLoader.batch_size or f < ( self.eq - self.margin)*self.dataLoader.batch_size:
            if o < (self.eq - self.margin) or f < (self.eq - self.margin):
            # if loss_decoder > loss_discriminator * .1:
                train_dis = False
            # if f > self.margin - self.eq or f > self.margin - self.eq:
            # if o > self.margin + self.eq or f > self.margin + self.eq:
            #if o > (self.margin + self.eq)*self.dataLoader.batch_size or f > (self.margin + self.eq)*self.dataLoader.batch_size:
            # if loss_decoder * .1 < loss_discriminator:
            if o > (self.margin + self.eq)or f > (self.margin + self.eq):
                train_dec = False
            if not train_dec and not train_dis:
                train_dec = True
                train_dis = True

            self.g_model.zero_grad()
            self.d_model.zero_grad()

            loss_encoder.backward(retain_graph=True)

            if train_dec:
                loss_decoder.backward(retain_graph=True)
                self.d_model.zero_grad()
            self.g_optimizer.step() # todo: split it
            # self.g_model.zero_grad()
            if train_dis:
                loss_discriminator.backward()
                self.d_optimizer.step()

            train_loss = [train_loss[0] + o.item(), train_loss[1] + f.item(), train_loss[2] + loss_encoder.item(), train_loss[3] + loss_decoder.item() ]

            # torch.cuda.empty_cache()

            print("Train epoch {0} ({1:.1f}%) >> losses: D: {2:.2f} (o:{3:.2f} + f:{4:.2f} )  enc: {5:.2f}  dec {6:.2f}     {7}  {8}".format(
                epoch,
                100 * (batch_index / (len(self.dataLoader.dataset) / self.dataLoader.batch_size)),
                (o + f)/ self.dataLoader.batch_size ,
                o/ self.dataLoader.batch_size,
                f/ self.dataLoader.batch_size,
                loss_encoder/ self.dataLoader.batch_size,
                loss_decoder/ self.dataLoader.batch_size,
                "" if train_dec else "e",
                "" if train_dis else "i"
                ),
                end="\r")
        print("Train epoch {0} (100.0%) >> avg.loss: {1:.2f} (o:{3:.2f} + f:{4:.2f} )  enc: {5:.2f}  dec {6:.2f} {2:^55} ".format(
            epoch,
            (train_loss[1] + train_loss[0]) / len(self.dataLoader.dataset),
            "elapsed time: {0:.2f} s.".format(time.time() - start_time),
            train_loss[0]/ len(self.dataLoader.dataset),
            train_loss[1]/ len(self.dataLoader.dataset),
            train_loss[2] / len(self.dataLoader.dataset),
            train_loss[3] / len(self.dataLoader.dataset)))

    def _train(self, epoch):
        train_loss = [0, 0, 0, 0]
        start_time = time.time()
        for batch_index, (data, target) in enumerate(self.dataLoader):
            data = data.to(self.device)
            valid = torch.Tensor(len(data)).fill_(1).to(self.device)
            fake = torch.Tensor(len(data)).fill_(0).to(self.device)

            (vae_loss, gan_loss), fake_data = self.train_the_G(data, fake)
            g_loss = vae_loss + gan_loss

            true_loss, fake_loss = self.train_the_D(data, fake_data, valid, fake)
            train_loss = [train_loss[0] + true_loss, train_loss[1] + fake_loss, train_loss[2] + gan_loss, train_loss[3] + vae_loss]

            print("Train epoch {0} ({1:.1f}%) >> losses: D: {2:.2f} ({3:.2f} /{4:.2f} )    G: {5:.2f}".format(epoch, 100 * (batch_index / (len(self.dataLoader.dataset) / self.dataLoader.batch_size)), fake_loss + true_loss, fake_loss,true_loss,g_loss), end="\r")
        print("Train epoch {0} (--.-%) >> avg.loss: {1:.2f} ({3:.2f} /{4:.2f} )    G: {5:.2f} ({6:.2f}+{7:.2f})  {2:^55} ".format(epoch, (train_loss[1] + train_loss[0]), "elapsed time: {0:.2f} s.".format(time.time() - start_time), train_loss[1], train_loss[0], (train_loss[2]+train_loss[3])/len(self.dataLoader.dataset), train_loss[2]/len(self.dataLoader.dataset), train_loss[3]/len(self.dataLoader.dataset)))

        for batch_index, (data, target) in enumerate(self.dataLoader):
            data = data.to(self.device)




    def test(self, epoch):
        pass