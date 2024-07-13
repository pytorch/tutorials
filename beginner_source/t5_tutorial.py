"""
T5-Base Model for Summarization, Sentiment Classification, and Translation
==========================================================================

**Authors**: `Pendo Abbo <pabbo@fb.com>`__, `Joe Cummings <jrcummings@fb.com>`__

"""

######################################################################
# Overview
# --------
#
# This tutorial demonstrates how to use a pretrained T5 Model for summarization, sentiment classification, and
# translation tasks. We will demonstrate how to use the torchtext library to:
#
# 1. Build a text preprocessing pipeline for a T5 model
# 2. Instantiate a pretrained T5 model with base configuration
# 3. Read in the CNNDM, IMDB, and Multi30k datasets and preprocess their texts in preparation for the model
# 4. Perform text summarization, sentiment classification, and translation
#
# .. note::
#    This tutorial requires PyTorch 2.0.0 or later.
#
#######################################################################
# Data Transformation
# -------------------
#
# The T5 model does not work with raw text. Instead, it requires the text to be transformed into numerical form
# in order to perform training and inference. The following transformations are required for the T5 model:
#
# 1. Tokenize text
# 2. Convert tokens into (integer) IDs
# 3. Truncate the sequences to a specified maximum length
# 4. Add end-of-sequence (EOS) and padding token IDs
#
# T5 uses a ``SentencePiece`` model for text tokenization. Below, we use a pretrained ``SentencePiece`` model to build
# the text preprocessing pipeline using torchtext's T5Transform. Note that the transform supports both
# batched and non-batched text input (for example, one can either pass a single sentence or a list of sentences), however the T5 model expects the input to be batched.
#

from torchtext.models import T5Transform

padding_idx = 0
eos_idx = 1
max_seq_len = 512
t5_sp_model_path = "https://download.pytorch.org/models/text/t5_tokenizer_base.model"

transform = T5Transform(
    sp_model_path=t5_sp_model_path,
    max_seq_len=max_seq_len,
    eos_idx=eos_idx,
    padding_idx=padding_idx,
)

#######################################################################
# Alternatively, we can also use the transform shipped with the pretrained models that does all of the above out-of-the-box
#
# .. code-block::
#
#   from torchtext.models import T5_BASE_GENERATION
#   transform = T5_BASE_GENERATION.transform()
#


######################################################################
# Model Preparation
# -----------------
#
# torchtext provides SOTA pretrained models that can be used directly for NLP tasks or fine-tuned on downstream tasks. Below
# we use the pretrained T5 model with standard base configuration to perform text summarization, sentiment classification, and
# translation. For additional details on available pretrained models, see `the torchtext documentation <https://pytorch.org/text/main/models.html>`__
#
#
from torchtext.models import T5_BASE_GENERATION


t5_base = T5_BASE_GENERATION
transform = t5_base.transform()
model = t5_base.get_model()
model.eval()


#######################################################################
# Using ``GenerationUtils``
# -------------------------
#
# We can use torchtext's ``GenerationUtils`` to produce an output sequence based on the input sequence provided. This calls on the
# model's encoder and decoder, and iteratively expands the decoded sequences until the end-of-sequence token is generated
# for all sequences in the batch. The ``generate`` method shown below uses greedy search to generate the sequences. Beam search and
# other decoding strategies are also supported.
#
#
from torchtext.prototype.generate import GenerationUtils

sequence_generator = GenerationUtils(model)


#######################################################################
# Datasets
# --------
# torchtext provides several standard NLP datasets. For a complete list, refer to the documentation
# at https://pytorch.org/text/stable/datasets.html. These datasets are built using composable torchdata
# datapipes and hence support standard flow-control and mapping/transformation using user defined
# functions and transforms.
#
# Below we demonstrate how to preprocess the CNNDM dataset to include the prefix necessary for the
# model to identify the task it is performing. The CNNDM dataset has a train, validation, and test
# split. Below we demo on the test split.
#
# The T5 model uses the prefix "summarize" for text summarization. For more information on task
# prefixes, please visit Appendix D of the `T5 Paper <https://arxiv.org/pdf/1910.10683.pdf>`__
#
# .. note::
#       Using datapipes is still currently subject to a few caveats. If you wish
#       to extend this example to include shuffling, multi-processing, or
#       distributed learning, please see :ref:`this note <datapipes_warnings>`
#       for further instructions.

from functools import partial

from torch.utils.data import DataLoader
from torchtext.datasets import CNNDM

cnndm_batch_size = 5
cnndm_datapipe = CNNDM(split="test")
task = "summarize"


def apply_prefix(task, x):
    return f"{task}: " + x[0], x[1]


cnndm_datapipe = cnndm_datapipe.map(partial(apply_prefix, task))
cnndm_datapipe = cnndm_datapipe.batch(cnndm_batch_size)
cnndm_datapipe = cnndm_datapipe.rows2columnar(["article", "abstract"])
cnndm_dataloader = DataLoader(cnndm_datapipe, shuffle=True, batch_size=None)

#######################################################################
# Alternately, we can also use batched API, for example, apply the prefix on the whole batch:
#
# .. code-block::
#
#   def batch_prefix(task, x):
#    return {
#        "article": [f'{task}: ' + y for y in x["article"]],
#        "abstract": x["abstract"]
#    }
#
#   cnndm_batch_size = 5
#   cnndm_datapipe = CNNDM(split="test")
#   task = 'summarize'
#
#   cnndm_datapipe = cnndm_datapipe.batch(cnndm_batch_size).rows2columnar(["article", "abstract"])
#   cnndm_datapipe = cnndm_datapipe.map(partial(batch_prefix, task))
#   cnndm_dataloader = DataLoader(cnndm_datapipe, batch_size=None)
#

#######################################################################
# We can also load the IMDB dataset, which will be used to demonstrate sentiment classification using the T5 model.
# This dataset has a train and test split. Below we demo on the test split.
#
# The T5 model was trained on the SST2 dataset (also available in torchtext) for sentiment classification using the
# prefix ``sst2 sentence``. Therefore, we will use this prefix to perform sentiment classification on the IMDB dataset.
#

from torchtext.datasets import IMDB

imdb_batch_size = 3
imdb_datapipe = IMDB(split="test")
task = "sst2 sentence"
labels = {"1": "negative", "2": "positive"}


def process_labels(labels, x):
    return x[1], labels[str(x[0])]


imdb_datapipe = imdb_datapipe.map(partial(process_labels, labels))
imdb_datapipe = imdb_datapipe.map(partial(apply_prefix, task))
imdb_datapipe = imdb_datapipe.batch(imdb_batch_size)
imdb_datapipe = imdb_datapipe.rows2columnar(["text", "label"])
imdb_dataloader = DataLoader(imdb_datapipe, batch_size=None)

#######################################################################
# Finally, we can also load the Multi30k dataset to demonstrate English to German translation using the T5 model.
# This dataset has a train, validation, and test split. Below we demo on the test split.
#
# The T5 model uses the prefix "translate English to German" for this task.

from torchtext.datasets import Multi30k

multi_batch_size = 5
language_pair = ("en", "de")
multi_datapipe = Multi30k(split="test", language_pair=language_pair)
task = "translate English to German"

multi_datapipe = multi_datapipe.map(partial(apply_prefix, task))
multi_datapipe = multi_datapipe.batch(multi_batch_size)
multi_datapipe = multi_datapipe.rows2columnar(["english", "german"])
multi_dataloader = DataLoader(multi_datapipe, batch_size=None)

#######################################################################
# Generate Summaries
# ------------------
#
# We can put all of the components together to generate summaries on the first batch of articles in the CNNDM test set
# using a beam size of 1.
#

batch = next(iter(cnndm_dataloader))
input_text = batch["article"]
target = batch["abstract"]
beam_size = 1

model_input = transform(input_text)
model_output = sequence_generator.generate(model_input, eos_idx=eos_idx, num_beams=beam_size)
output_text = transform.decode(model_output.tolist())

for i in range(cnndm_batch_size):
    print(f"Example {i+1}:\n")
    print(f"prediction: {output_text[i]}\n")
    print(f"target: {target[i]}\n\n")


#######################################################################
# Summarization Output
# --------------------
# 
# Summarization output might vary since we shuffle the dataloader.
#
# .. code-block::
#
#    Example 1:
#
#    prediction: the 24-year-old has been tattooed for over a decade . he has landed in australia
#    to start work on a new campaign . he says he is 'taking it in your stride' to be honest .
#
#    target: London-based model Stephen James Hendry famed for his full body tattoo . The supermodel
#    is in Sydney for a new modelling campaign . Australian fans understood to have already located
#    him at his hotel . The 24-year-old heartthrob is recently single .
#
#
#    Example 2:
#
#    prediction: a stray pooch has used up at least three of her own after being hit by a
#    car and buried in a field . the dog managed to stagger to a nearby farm, dirt-covered
#    and emaciated, where she was found . she suffered a dislocated jaw, leg injuries and a
#    caved-in sinus cavity -- and still requires surgery to help her breathe .
#
#    target: Theia, a bully breed mix, was apparently hit by a car, whacked with a hammer
#    and buried in a field . "She's a true miracle dog and she deserves a good life," says
#    Sara Mellado, who is looking for a home for Theia .
#
#
#    Example 3:
#
#    prediction: mohammad Javad Zarif arrived in Iran on a sunny friday morning . he has gone
#    a long way to bring Iran in from the cold and allow it to rejoin the international
#    community . but there are some facts about him that are less well-known .
#
#    target: Mohammad Javad Zarif has spent more time with John Kerry than any other
#    foreign minister . He once participated in a takeover of the Iranian Consulate in San
#    Francisco . The Iranian foreign minister tweets in English .
#
#
#    Example 4:
#
#    prediction: five americans were monitored for three weeks after being exposed to Ebola in
#    west africa . one of the five had a heart-related issue and has been discharged but hasn't
#    left the area . they are clinicians for Partners in Health, a Boston-based aid group .
#
#    target: 17 Americans were exposed to the Ebola virus while in Sierra Leone in March .
#    Another person was diagnosed with the disease and taken to hospital in Maryland .
#    National Institutes of Health says the patient is in fair condition after weeks of
#    treatment .
#
#
#    Example 5:
#
#    prediction: the student was identified during an investigation by campus police and
#    the office of student affairs . he admitted to placing the noose on the tree early
#    Wednesday morning . the incident is one of several recent racist events to affect
#    college students .
#
#    target: Student is no longer on Duke University campus and will face disciplinary
#    review . School officials identified student during investigation and the person
#    admitted to hanging the noose, Duke says . The noose, made of rope, was discovered on
#    campus about 2 a.m.
#


#######################################################################
# Generate Sentiment Classifications
# ----------------------------------
#
# Similarly, we can use the model to generate sentiment classifications on the first batch of reviews from the IMDB test set
# using a beam size of 1.
#

batch = next(iter(imdb_dataloader))
input_text = batch["text"]
target = batch["label"]
beam_size = 1

model_input = transform(input_text)
model_output = sequence_generator.generate(model_input, eos_idx=eos_idx, num_beams=beam_size)
output_text = transform.decode(model_output.tolist())

for i in range(imdb_batch_size):
    print(f"Example {i+1}:\n")
    print(f"input_text: {input_text[i]}\n")
    print(f"prediction: {output_text[i]}\n")
    print(f"target: {target[i]}\n\n")


#######################################################################
# Sentiment Output
# ----------------
#
# .. code-block:: bash
#
#    Example 1:
#
#    input_text: sst2 sentence: I love sci-fi and am willing to put up with a lot. Sci-fi
#    movies/TV are usually underfunded, under-appreciated and misunderstood. I tried to like
#    this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original).
#    Silly prosthetics, cheap cardboard sets, stilted dialogues, CG that doesn't match the
#    background, and painfully one-dimensional characters cannot be overcome with a 'sci-fi'
#    setting. (I'm sure there are those of you out there who think Babylon 5 is good sci-fi TV.
#    It's not. It's clichéd and uninspiring.) While US viewers might like emotion and character
#    development, sci-fi is a genre that does not take itself seriously (cf. Star Trek). It may
#    treat important issues, yet not as a serious philosophy. It's really difficult to care about
#    the characters here as they are not simply foolish, just missing a spark of life. Their
#    actions and reactions are wooden and predictable, often painful to watch. The makers of Earth
#    KNOW it's rubbish as they have to always say "Gene Roddenberry's Earth..." otherwise people
#    would not continue watching. Roddenberry's ashes must be turning in their orbit as this dull,
#    cheap, poorly edited (watching it without advert breaks really brings this home) trudging
#    Trabant of a show lumbers into space. Spoiler. So, kill off a main character. And then bring
#    him back as another actor. Jeeez. Dallas all over again.
#
#    prediction: negative
#
#    target: negative
#
#
#    Example 2:
#
#    input_text: sst2 sentence: Worth the entertainment value of a rental, especially if you like
#    action movies. This one features the usual car chases, fights with the great Van Damme kick
#    style, shooting battles with the 40 shell load shotgun, and even terrorist style bombs. All
#    of this is entertaining and competently handled but there is nothing that really blows you
#    away if you've seen your share before.<br /><br />The plot is made interesting by the
#    inclusion of a rabbit, which is clever but hardly profound. Many of the characters are
#    heavily stereotyped -- the angry veterans, the terrified illegal aliens, the crooked cops,
#    the indifferent feds, the bitchy tough lady station head, the crooked politician, the fat
#    federale who looks like he was typecast as the Mexican in a Hollywood movie from the 1940s.
#    All passably acted but again nothing special.<br /><br />I thought the main villains were
#    pretty well done and fairly well acted. By the end of the movie you certainly knew who the
#    good guys were and weren't. There was an emotional lift as the really bad ones got their just
#    deserts. Very simplistic, but then you weren't expecting Hamlet, right? The only thing I found
#    really annoying was the constant cuts to VDs daughter during the last fight scene.<br /><br />
#    Not bad. Not good. Passable 4.
#
#    prediction: positive
#
#    target: negative
#
#
#    Example 3:
#
#    input_text: sst2 sentence: its a totally average film with a few semi-alright action sequences
#    that make the plot seem a little better and remind the viewer of the classic van dam films.
#    parts of the plot don't make sense and seem to be added in to use up time. the end plot is that
#    of a very basic type that doesn't leave the viewer guessing and any twists are obvious from the
#    beginning. the end scene with the flask backs don't make sense as they are added in and seem to
#    have little relevance to the history of van dam's character. not really worth watching again,
#    bit disappointed in the end production, even though it is apparent it was shot on a low budget
#    certain shots and sections in the film are of poor directed quality.
#
#    prediction: negative
#
#    target: negative
#


#######################################################################
# Generate Translations
# ---------------------
#
# Finally, we can also use the model to generate English to German translations on the first batch of examples from the Multi30k
# test set.
#

batch = next(iter(multi_dataloader))
input_text = batch["english"]
target = batch["german"]

model_input = transform(input_text)
model_output = sequence_generator.generate(model_input, eos_idx=eos_idx, num_beams=beam_size)
output_text = transform.decode(model_output.tolist())

for i in range(multi_batch_size):
    print(f"Example {i+1}:\n")
    print(f"input_text: {input_text[i]}\n")
    print(f"prediction: {output_text[i]}\n")
    print(f"target: {target[i]}\n\n")


#######################################################################
# Translation Output
# ------------------
#
# .. code-block:: bash
#
#    Example 1:
#
#    input_text: translate English to German: A man in an orange hat starring at something.
#
#    prediction: Ein Mann in einem orangen Hut, der an etwas schaut.
#
#    target: Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt.
#
#
#    Example 2:
#
#    input_text: translate English to German: A Boston Terrier is running on lush green grass in front of a white fence.
#
#    prediction: Ein Boston Terrier läuft auf üppigem grünem Gras vor einem weißen Zaun.
#
#    target: Ein Boston Terrier läuft über saftig-grünes Gras vor einem weißen Zaun.
#
#
#    Example 3:
#
#    input_text: translate English to German: A girl in karate uniform breaking a stick with a front kick.
#
#    prediction: Ein Mädchen in Karate-Uniform bricht einen Stöck mit einem Frontkick.
#
#    target: Ein Mädchen in einem Karateanzug bricht ein Brett mit einem Tritt.
#
#
#    Example 4:
#
#    input_text: translate English to German: Five people wearing winter jackets and helmets stand in the snow, with snowmobiles in the background.
#
#    prediction: Fünf Menschen mit Winterjacken und Helmen stehen im Schnee, mit Schneemobilen im Hintergrund.
#
#    target: Fünf Leute in Winterjacken und mit Helmen stehen im Schnee mit Schneemobilen im Hintergrund.
#
#
#    Example 5:
#
#    input_text: translate English to German: People are fixing the roof of a house.
#
#    prediction: Die Leute fixieren das Dach eines Hauses.
#
#    target: Leute Reparieren das Dach eines Hauses.
#
