import io
import numpy as np
import tensorflow as tf
from hparams import hparams
from models.dual_source_tacotron import DualSourceTacotron
from util import audio, textinput_fr


class DualSourceSynthesizer:
  def load(self, checkpoint_path, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    inputs1 = tf.placeholder(tf.int32, [1, None], 'inputs1')
    input_lengths1 = tf.placeholder(tf.int32, [1], 'input_lengths1')
    inputs2 = tf.placeholder(tf.int32, [1, None], 'inputs2')
    input_lengths2 = tf.placeholder(tf.int32, [1], 'input_lengths2')
    with tf.variable_scope('model') as scope:
      self.model = DualSourceTacotron(hparams)
      self.model.initialize(inputs1, input_lengths1, inputs2, input_lengths2)

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)


  def synthesize(self, text1, text2):
    seq1 = textinput_fr.to_sequence(text1,
      force_lowercase=hparams.force_lowercase,
      expand_abbreviations=hparams.expand_abbreviations)
    seq2 = textinput_fr.to_sequence(text2,
      force_lowercase=hparams.force_lowercase,
      expand_abbreviations=False)
    feed_dict = {
      self.model.inputs1: [np.asarray(seq1, dtype=np.int32)],
      self.model.input_lengths1: np.asarray([len(seq1)], dtype=np.int32),
      self.model.inputs2: [np.asarray(seq2, dtype=np.int32)],
      self.model.input_lengths2: np.asarray([len(seq2)], dtype=np.int32)
    }
    spec, alignments1, alignments2 = self.session.run([self.model.linear_outputs[0], self.model.alignments1[0], self.model.alignments2[0]], feed_dict=feed_dict)
    out = io.BytesIO()
    audio.save_wav(audio.inv_spectrogram(spec.T), out)
    return out.getvalue(), alignments1, alignments2
 