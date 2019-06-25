from vaex.serialize import to_dict, from_dict
import vaex.utils


class Pipeline(list):

    def save(self, f):
        states = [to_dict(k) for k in self]
        vaex.utils.write_json_or_yaml(f, states)

    def load(self, f):
        states = vaex.utils.read_json_or_yaml(f)
        objects = [from_dict(k) for k in states]
        del self[:]  # TODOPY2: replace by .clear
        self.extend(objects)

    def predict(self, dataframe):
        assert len(self) > 0, "cannot predict when pipeline is empty"
        # apply transformations for all but the last
        transforms = self[:-1]
        for t in transforms:
            dataframe = t.transform(dataframe)
        # last does predictions
        return self[-1].predict(dataframe)

    def transform(self, dataframe):
        assert len(self) > 0, "cannot predict when pipeline is empty"
        # apply transformations for all but the last
        for t in self:
            dataframe = t.transform(dataframe)
        return dataframe
