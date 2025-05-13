from util.argparser import get_args
from util.env import prepare_env
from util.Data import Dataset
from models.vae.VAE_wrapper import VAE_wrapper

def main():
    args = get_args()
    prepare_env(args)
    data = Dataset(args)
    if args.model== 'vae':
        model = VAE_wrapper(args,data,tensorboard=True)
    else:
        raise Exception('{} model not implemented'.format(args.model))
    if model.load():
        model.train()
    model.score()


if __name__=="__main__":
    main()