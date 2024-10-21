
import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, args):
        log_dir = args.log_dir + '-' + str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.writer = SummaryWriter(log_dir)

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def add_image(self, tag, image, step):
        self.writer.add_image(tag, image, step)

    def add_graph(self, model, input_to_model):
        self.writer.add_graph(model, input_to_model)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.log_dir = '../../logs/test'


    logger = Logger(Args())
    x = range(100)
    for i in x:
        logger.add_scalar('y=2x', i * 2, i)
