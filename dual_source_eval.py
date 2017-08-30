import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from dual_source_synthesizer import DualSourceSynthesizer
from util import plot

sentences1 = [
  # From July 8, 2017 New York Times:
  'Scientists at the CERN laboratory say they have discovered a new particle.',
  'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
  'President Trump met with other leaders at the Group of 20 conference.',
  'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
  # From Google's Tacotron example page:
  'Generative adversarial network or variational auto-encoder.',
  'The buses aren\'t the problem, they actually provide a solution.',
  'Does the quick brown fox jump over the lazy dog?',
  'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
]

sentences2 = [
  # SIWIS part4
  "Comme il plaira à Monsieur",
  "Conseil était mon domestique.",
  "Et cependant, quel brave et digne garçon !",
  "Monsieur m’appelle ? dit-il en entrant.",
  "Comme il conviendra à monsieur, répondit paisiblement Conseil.",
  "Oui, monsieur, répondit l’ingénieur.",
  # SIWIS part5
  "criai-je d’une voix impatiente.",
  "répétai-je, tout en commençant d’une main fébrile mes préparatifs de départ.",
  "Certainement, j’étais sûr de ce garçon si dévoué.",
]


def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  synth = DualSourceSynthesizer()
  synth.load(args.checkpoint)
  base_path = get_output_base_path(args.checkpoint)
  for i, text in enumerate(sentences1):
    n_encoder = 1
    path = '%s-%d-%d.wav' % (base_path, n_encoder, i)
    alignment_path = '%s-%d-%d.png' % (base_path, n_encoder, i)
    print('Synthesizing: %s' % path)
    audio, alignment1, alignment2 = synth.synthesize(text, " " * len(text))
    with open(path, 'wb') as f:
      f.write(audio)
    plot.plot_alignment_and_text(alignment1, '%s-en%d-%d.png' % (base_path, i, 1), text)
    plot.plot_alignment_and_text(alignment2, '%s-en%d-%d.png' % (base_path, i, 2), " " * len(text))

  for i, text in enumerate(sentences2):
    n_encoder = 2
    path = '%s-%d-%d.wav' % (base_path, n_encoder, i)
    alignment_path = '%s-%d-%d.png' % (base_path, n_encoder, i)
    print('Synthesizing: %s' % path)
    audio, alignment1, alignment2 = synth.synthesize(" " * len(text), text)
    with open(path, 'wb') as f:
      f.write(audio)
    plot.plot_alignment_and_text(alignment1, '%s-fr%d-%d.png' % (base_path, i, 1), " " * len(text))
    plot.plot_alignment_and_text(alignment2, '%s-fr%d-%d.png' % (base_path, i, 2), text)
    

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.max_iters = 100
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == "__main__":
  main()
