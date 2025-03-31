from monai.transforms import RandomizableTransform
from monai.config.type_definitions import KeysCollection


class SelectRandomSliced(RandomizableTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__()
        self.keys = keys

    def randomize(self):
        super().randomize(None)

    def __call__(self, data):
        self.randomize()
        k = self.R.randint(0, data[self.keys[0]].shape[0])

        if not self._do_transform:
            return data

        for key in self.keys:
            data[key] = data[key][k, :, :].unsqueeze(0)
        return data
