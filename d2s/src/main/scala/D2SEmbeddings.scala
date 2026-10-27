import d2s.*
import d2s.config.*
import d2s.eval.answeredAt
import d2s.model.D2S
import d2s.train.D2STrainState
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.NodeClass
import dataset.RecordNodes
import dataset.Runs
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import dimwit.jax.Jax
import dimwit.tensor.Tensor4
import documentEncoder.ImageToPatchEmbedder

import java.nio.file.Files
import java.nio.file.Path

import scala.language.implicitConversions

/** Writes what the newest checkpoint of the newest run makes of a sample of the validation
  * drawings, at either end of the model: `sbt "d2s/runMain d2sEmbeddings vitruvion-primitives s-deep"`.
  *
  * The encoder's are the drawing's encoded tiles, as its last block leaves them, and which of them
  * hold ink. The decoder's are the prediction embeddings the nodes it writes down for the drawing
  * were read from, as the head reads them, and which of its slots hold a node. The drawing is
  * transcribed as [[d2s.eval.Transcriber]] transcribes it, reading back only what the model has
  * taken itself.
  *
  * They land in `OUTPUT_DIR` as one `.npz`, a row a drawing: `drawing`, which drawing of the split
  * the row is; `patches` and `inked`, its tiles; `items` and `written`, its node slots.
  */
@main
def d2sEmbeddings(corpus: String, size: String): Unit =
  dimwit.initialize()
  val setup = D2SSetup(Corpus.named(corpus), D2SModelConfiguration.named(size))

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  val step = checkpoints.iterations.lastOption.getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}"))
  println(s"reading step $step of ${checkpoints.rootPath}")
  val params = checkpoints.load[D2STrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params

  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(Split.Validation)
  val sampled = Random.permutation(Axis[Drawing] -> data.numSamples)(Random.Key(SampleSeed)).slice(Axis[Drawing].at(0 until Sampled)).toArray.sorted
  val isSampled = sampled.toSet
  val jitEmbedded = jit(embedded(Axis[Node] -> setup.nodeSlots))

  /** A batch handing over fewer drawings is filled up with one it already holds, so that every
    * batch is the same shape and compiles once, and is read back short again.
    */
  val batches = data.samples
    .zipWithIndex
    .collect { case (sample, at) if isSampled(at) => sample.image }
    .grouped(EmbeddedTogether)
    .map: documents =>
      val (patches, inked, items, written) = jitEmbedded(params, stack(documents.padTo(EmbeddedTogether, documents.last), Axis[Drawing]))
      val handedOver = Axis[Drawing].at(0 until documents.size)
      (patches.slice(handedOver), inked.slice(handedOver), items.slice(handedOver), written.slice(handedOver))
    .toSeq

  def numpy[T <: Tuple: Labels, V](tensor: Tensor[T, V]) = TensorTree[Tensor[T, V]].toNumpyTree(tensor)
  val path = Path.of(Runs.outputDir, s"d2s-embeddings-${setup.corpus.name}-$size-${Split.Validation.fileName}-$step.npz")
  Files.createDirectories(path.getParent)
  Jax.np.savez(
    path.toString,
    drawing = numpy(Tensor1(Axis[Drawing], VType[Int32]).fromArray(sampled)),
    patches = numpy(concatenate(batches.map(_._1), Axis[Drawing])),
    inked = numpy(concatenate(batches.map(_._2), Axis[Drawing])),
    items = numpy(concatenate(batches.map(_._3), Axis[Drawing])),
    written = numpy(concatenate(batches.map(_._4), Axis[Drawing]))
  )
  println(s"${sampled.length} drawings written to $path")

/** Every drawing's encoded tiles and which of them hold ink, and the prediction embeddings of the
  * nodes the decoder writes down for it and which of its slots hold one.
  */
private def embedded(nodes: AxisExtent[Node])(
    params: D2S.Params[Float32],
    documents: Tensor4[Drawing, Width, Height, Channel, Float32]
): (Tensor3[Drawing, Patch, Embedding, Float32], Tensor2[Drawing, Patch, Float32], Tensor3[Drawing, Node, Embedding, Float32], Tensor2[Drawing, Node, Float32]) =
  val model = D2S(params)
  val slots = Tensor1(nodes.axis).fromRange(0 until nodes.size)
  documents.vmap(Axis[Drawing]): document =>
    val encoded = model.encodeDocument(document)

    val documentCache = model.read(encoded)
    val (_, (writtenClasses, predictions)) = scan(Axis[Node])((model.nothingTaken(nodes), Tensor0(true)), slots):
      case ((taken, writing), slot) =>
        val (node, stillWriting, prediction) = answeredAt(model, documentCache, taken, slot, writing)
        def one[W](value: Tensor0[W]) = value.prependAxis(Axis[Node])
        val (cls, isConstruction, sx, sy, ex, ey, mx, my) = node
        val taking = RecordNodes(one(cls), one(isConstruction), one(sx), one(sy), one(ex), one(ey), one(mx), one(my))
        val (_, nowTaken) = model.take(documentCache, taken, slot, taking)
        ((nowTaken, stillWriting), (cls, prediction))
    val written = (!(writtenClasses elementEquals_! NodeClass.NoNode.id)).asFloat(VType[Float32])

    (encoded, inkedTiles(document), predictions, written)

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

/** How many drawings of the split are embedded: drawn at random, the same ones every time. */
private val Sampled = 5000
private val SampleSeed = 0

/** How many drawings are embedded together: one traced computation. */
private val EmbeddedTogether = 32

/** Axis of the drawings embedded together. */
private trait Drawing derives Label
