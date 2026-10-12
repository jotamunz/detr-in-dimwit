import d2s.*
import d2s.config.*
import d2s.model.D2S
import d2s.train.D2STrainState
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.Runs
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import dimwit.jax.Jax
import dimwit.tensor.Tensor4
import documentEncoder.DocumentEncoder
import documentEncoder.ImageToPatchEmbedder

import java.nio.file.Files
import java.nio.file.Path

import scala.language.implicitConversions

/** Writes what the encoder of the newest checkpoint of the newest run makes of every validation
  * drawing, one embedding a drawing: `sbt "d2s/runMain d2sEncodings vitruvion-primitives s-deep"`.
  *
  * A drawing's embedding is the mean of its encoded tiles that hold ink. Most tiles of a drawing
  * are blank, and averaging them in would make every drawing look alike.
  *
  * The embeddings land in `OUTPUT_DIR` as one `.npz`, in the order of the split: `embeddings`, one
  * row a drawing, and `inkedTiles`, how many tiles each row is the mean of.
  */
@main
def d2sEncodings(corpus: String, size: String): Unit =
  dimwit.initialize()
  val setup = D2SSetup(Corpus.named(corpus), D2SModelConfiguration.named(size))

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  val step = checkpoints.iterations.lastOption.getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}"))
  println(s"reading step $step of ${checkpoints.rootPath}")
  val params = checkpoints.load[D2STrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params

  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(Split.Validation)
  val jitEncoded = jit(encoded)

  /** A batch handing over fewer drawings is filled up with one it already holds, so that every
    * batch is the same shape and compiles once, and is read back short again.
    */
  val (embeddings, inkedTiles) = data.samples
    .map(_.image)
    .grouped(EncodedTogether)
    .map: documents =>
      val (embeddings, inkedTiles) = jitEncoded(params, stack(documents.padTo(EncodedTogether, documents.last), Axis[Drawing]))
      val handedOver = Axis[Drawing].at(0 until documents.size)
      (embeddings.slice(handedOver), inkedTiles.slice(handedOver))
    .toSeq
    .unzip

  def numpy[T <: Tuple: Labels, V](tensor: Tensor[T, V]) = TensorTree[Tensor[T, V]].toNumpyTree(tensor)
  val path = Path.of(Runs.outputDir, s"d2s-encodings-${setup.corpus.name}-$size-${Split.Validation.fileName}-$step.npz")
  Files.createDirectories(path.getParent)
  Jax.np.savez(
    path.toString,
    embeddings = numpy(concatenate(embeddings, Axis[Drawing])),
    inkedTiles = numpy(concatenate(inkedTiles, Axis[Drawing]))
  )
  println(s"${data.numSamples} drawings written to $path")

/** Every drawing's embedding, and how many tiles it is the mean of. */
private def encoded(
    params: D2S.Params[Float32],
    documents: Tensor4[Drawing, Width, Height, Channel, Float32]
): (Tensor2[Drawing, Embedding, Float32], Tensor1[Drawing, Float32]) =
  val encode = DocumentEncoder(params.encoder)
  documents.vmap(Axis[Drawing]): document =>
    val inked = inkedTiles(document)
    (inked.dot(Axis[Patch])(encode(document)) /! inked.sum, inked.sum)

/** One for every tile of the drawing that holds ink, in the order the encoder lays its patches out.
  * A blank tile is pure white.
  */
private def inkedTiles(document: Tensor3[Width, Height, Channel, Float32]): Tensor1[Patch, Float32] =
  trait PatchX derives Label
  trait PatchY derives Label
  val side = ImageToPatchEmbedder.patchSize
  val tiles = document
    .unflatten(Axis[Width], Shape(Axis[PatchX] -> document.shape(Axis[Width]) / side, Axis[Width] -> side))
    .unflatten(Axis[Height], Shape(Axis[PatchY] -> document.shape(Axis[Height]) / side, Axis[Height] -> side))
  val blank = tiles.min((Axis[Width], Axis[Height], Axis[Channel])) elementEquals_! 1f
  (!blank).asFloat(VType[Float32]).flatten((Axis[PatchX], Axis[PatchY])).relabel(Axis[PatchX |*| PatchY] -> Axis[Patch])

/** How many drawings are encoded together: one traced computation. */
private val EncodedTogether = 32

/** Axis of the drawings encoded together. */
private trait Drawing derives Label
