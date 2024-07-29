from fairseq.data.dictionary import Dictionary

class GWLANDictionary(Dictionary):
    def mask(self):
        return self.mask_index