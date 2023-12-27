import torch
import random
import numpy as np

from configs import FLAGS


class TransformerSolver:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

        self.start_epoch = 1
        self.model = None           # torch.nn.Module
        self.optimizer = None       # torch.optim.Optimizer  
        self.scheduler = None       # torch.optim.lr_scheduler._LRScheduler
        self.train_loader = None
        self.test_loader = None
        if FLAGS.device < 0:
           self.device = torch.device('cpu')
        else:
           self.device = torch.device(f'cuda:{FLAGS.device}')
        # .....

    def config_model(self):
        pass 

    def load_vqvae(self):
        # load trained VQ-VAE model to extract quantited embeddings of images
        pass

    def get_dataset(self, flag):
        pass
    
    def config_dataloader(self, disable_train=False):
        pass

    def config_optimizer(self):
        pass

    def config_scheduler(self):
        pass
    
    def config_log(self):
        pass

    def train(self):
        self.manual_seed()
        self.config_model()
        self.config_dataloader()
        self.config_optimizer()
        self.config_scheduler()
        self.config_log()
        self.load_vqvae()

        # set model as train mode
        # read data
        # model forward process
        # compute loss
        # compute gradient
        # optimize parameters
        # ......

    def test(self):
        self.manual_seed()
        self.config_model()
        self.config_dataloader(True)
        self.config_optimizer()
        self.config_scheduler()
        self.config_log()
        self.load_vqvae()
        # set model as eval mode
        # read data
        # model forward process
        # compute loss
        # ......

    def manual_seed(self):
        rand_seed = self.FLAGS.rand_seed
        if rand_seed > 0:
            random.seed(rand_seed)
            np.random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def run(self):
        eval('self.%s()' % self.FLAGS.run)

    @classmethod
    def main(cls):
        completion = cls(FLAGS)
        completion.run()

if __name__ == '__main__':
    TransformerSolver.main()
