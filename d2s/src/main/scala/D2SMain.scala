import d2s.config.D2SModelConfiguration
import dataset.Corpus
import dataset.DrawingDataset.Split
import syntheticFloorplansDataset.SyntheticFloorplans

/** Entry points for the D2S experiments, on a corpus (e.g., `sketch-xl` or `synthetic-floorplans`)
  * at a model configuration (e.g., `s-deep`): `sbt "d2s/runMain d2sTrain sketch-xl s-deep simplest-first 0"`.
  *
  * Each kind of record has a model of its own: the sketch corpora (see [[Corpus]]) are read by
  * [[d2s.sketch]], the synthetic floor plans by [[d2s.floorplan]]. The corpus names which, and the
  * order of the nodes is one of that model's: [[d2s.sketch.NodeOrder.named]] or
  * [[d2s.floorplan.NodeOrder.named]]. See [[D2SModelConfiguration]] for the sizes.
  *
  * [[d2sTrain]] trains a run in an order of the nodes, from a seed, so that a setting can be run again.
  * [[d2sEval]] scores the newest run of that corpus and size (greedy decoding).
  * [[d2sTest]] scores its last checkpoint on the test split.
  * [[d2sPlot]] shows what it transcribes (visualization).
  * [[d2sDraw]] writes what every checkpoint transcribed as a picture.
  * [[d2sEmbeddings]] writes what the newest floor plan run makes of a sample of the validation plans.
  */

@main
def d2sTrain(corpus: String, size: String, order: String, seed: Int): Unit =
  if corpus == SyntheticFloorplans.name then
    val setup = d2s.floorplan.config.D2SSetup(D2SModelConfiguration.named(size))
    d2s.floorplan.train.trainSetTranscriber(setup.copy(nodeOrder = d2s.floorplan.NodeOrder.named(order), seed = seed))
  else
    val setup = d2s.sketch.config.D2SSetup(Corpus.named(corpus), D2SModelConfiguration.named(size))
    d2s.sketch.train.trainSetTranscriber(setup.copy(nodeOrder = d2s.sketch.NodeOrder.named(order), seed = seed))

@main
def d2sEval(corpus: String, size: String): Unit = score(corpus, size, Split.Validation)

@main
def d2sTest(corpus: String, size: String): Unit = score(corpus, size, Split.Test)

@main
def d2sPlot(corpus: String, size: String): Unit =
  if corpus == SyntheticFloorplans.name then d2s.floorplan.eval.plotSetTranscriber(d2s.floorplan.config.D2SSetup(D2SModelConfiguration.named(size)))
  else d2s.sketch.eval.plotSetTranscriber(d2s.sketch.config.D2SSetup(Corpus.named(corpus), D2SModelConfiguration.named(size)))

/** Writes what every checkpoint of the newest run transcribed, one picture each. */
@main
def d2sDraw(corpus: String, size: String): Unit =
  if corpus == SyntheticFloorplans.name then d2s.floorplan.eval.drawSetTranscriber(d2s.floorplan.config.D2SSetup(D2SModelConfiguration.named(size)), size)
  else d2s.sketch.eval.drawSetTranscriber(d2s.sketch.config.D2SSetup(Corpus.named(corpus), D2SModelConfiguration.named(size)), size)

/** Writes the embeddings, at either end of the model, of a sample of the validation floor plans. */
@main
def d2sEmbeddings(size: String): Unit = d2s.floorplan.eval.writeEmbeddings(d2s.floorplan.config.D2SSetup(D2SModelConfiguration.named(size)), size)

private def score(corpus: String, size: String, split: Split): Unit =
  if corpus == SyntheticFloorplans.name then
    d2s.floorplan.eval.scoreSetTranscriber(d2s.floorplan.config.D2SSetup(D2SModelConfiguration.named(size)), size, split)
  else
    d2s.sketch.eval.scoreSetTranscriber(d2s.sketch.config.D2SSetup(Corpus.named(corpus), D2SModelConfiguration.named(size)), size, split)
