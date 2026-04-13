"""
Character tokenizer and converter for text recognition
Based on deep-text-recognition-benchmark
"""
import torch


class CTCLabelConverter:
    """Convert between text-label and text-index for CTC"""

    def __init__(self, character):
        dict_character = list(character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1
        self.character = ['[blank]'] + dict_character

    def encode(self, text, batch_max_length=25):
        """
        Args:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default
        Returns:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text, torch.IntTensor(length))

    def decode(self, text_index, length):
        """
        Args:
            text_index: [batch_size x batch_max_length]
            length: [batch_size]
        Returns:
            texts: text labels of each image
        """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]
            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)
            texts.append(text)
        return texts


class AttnLabelConverter:
    """Convert between text-label and text-index for Attention"""

    def __init__(self, character):
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """
        Args:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default
        Returns:
            text: text index for Attention. [batch_size x (max_length+2)] for [GO] and [s] tokens
            length: length of each text. [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            # batch_text[:, 0] = [GO] token
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)
        return (batch_text, torch.IntTensor(length))

    def decode(self, text_index, length):
        """
        Args:
            text_index: [batch_size x batch_max_length]
            length: [batch_size]
        Returns:
            texts: text labels of each image
        """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Tokenizer:
    """Simple tokenizer that wraps AttnLabelConverter"""

    def __init__(self, charset):
        self.charset = charset
        self.converter = AttnLabelConverter(charset)

    def encode(self, labels, device=None):
        """Encode text labels to indices"""
        batch_text, length = self.converter.encode(labels)
        if device:
            batch_text = batch_text.to(device)
            length = length.to(device)
        return batch_text, length

    def decode(self, token_dists, raw=False):
        """
        Decode token distributions to text following deep-text-recognition-benchmark approach

        Args:
            token_dists: [batch_size x seq_length x num_classes] probability distributions
            raw: if True, return raw probabilities (not used, for API compatibility)
        Returns:
            texts: list of decoded text strings
            confidences: list of confidence scores
        """
        batch_size = token_dists.shape[0]
        seq_length = token_dists.shape[1]

        # Select max probability (greedy decoding)
        preds_max_prob, preds_index = token_dists.max(
            dim=2)  # [batch_size x seq_length]

        texts = []
        confidences = []

        # First decode using the converter (this includes special tokens)
        length_for_pred = torch.IntTensor([seq_length] * batch_size)
        preds_str = self.converter.decode(preds_index, length_for_pred)

        for i in range(batch_size):
            pred = preds_str[i]

            # Find and remove everything after [s] token (end of sentence)
            pred_EOS = pred.find('[s]')
            if pred_EOS != -1:
                pred = pred[:pred_EOS]

            # Remove [GO] token if present at the start
            if pred.startswith('[GO]'):
                pred = pred[4:]  # Remove '[GO]' which is 4 characters

            # Calculate confidence score as product of max probabilities until EOS
            # Use pred_EOS if found, otherwise use full length
            end_idx = pred_EOS if pred_EOS != -1 else seq_length
            try:
                if end_idx > 0:
                    confidence_score = preds_max_prob[i][:end_idx].cumprod(
                        dim=0)[-1].item()
                else:
                    confidence_score = 0.0
            except:
                confidence_score = 0.0

            texts.append(pred)
            confidences.append(confidence_score)

        return texts, confidences
