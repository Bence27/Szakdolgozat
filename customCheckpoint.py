from rl.callbacks import FileLogger, ModelIntervalCheckpoint

class CustomModelIntervalCheckpoint(ModelIntervalCheckpoint):
    def __init__(self, filepath, interval, verbose=0):
        super(CustomModelIntervalCheckpoint, self).__init__(filepath, interval, verbose)
        self.a = 0

    def on_step_end(self, step, logs={}):
        self.a += 1  
        if self.a != 0 and self.a % self.interval == 0:
            # Create a unique checkpoint filename using the current step number
            filename = f"{self.filepath.split('.h5')[0]}_{self.a}.h5"
            if self.verbose > 0:
                print(f"\nStep {self.a}: saving model to {filename}")
            self.model.save_weights(filename, overwrite=True)