from runner.runner import Runner


class GAILRunner(Runner):
    def __init__(self, args, env, logger):
        super().__init__(args, env, logger)
        if not args.imitation_learning:
            raise RuntimeError("You are not using imitation learning, please check your arguments.")

    def imitation_learning(self):
        self.load_expert_data()
        self.run()
