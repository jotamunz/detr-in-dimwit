package d2s.floorplan.eval

import d2s.*
import d2s.floorplan.config.*
import d2s.floorplan.model.D2S
import d2s.floorplan.train.D2STrainState
import dataset.DrawingDataset.Split
import dataset.Runs
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import dimwit.jax.Jax
import dimwit.tensor.Tensor4
import documentEncoder.ImageToPatchEmbedder
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanNodes
import syntheticFloorplansDataset.SyntheticFloorplans

import java.nio.file.Files
import java.nio.file.Path

import scala.language.implicitConversions

/** Writes what the newest checkpoint of the newest run makes of a sample of the validation plans,
  * at either end of the model: `sbt "d2s/runMain d2sEmbeddings s-deep"`.
  *
  * The encoder's are the plan's encoded tiles, as its last block leaves them, and which of them
  * hold ink. The decoder's are the prediction embeddings the nodes it writes down for the plan were
  * read from, as the head reads them, and which of its slots hold a node. The plan is transcribed
  * as [[Transcriber]] transcribes it, reading back only what the model has taken itself.
  *
  * They land in `OUTPUT_DIR` as one `.npz`, a row a plan: `drawing`, which plan of the split the row
  * is; `pictures`, the plan as the model read it, in grey levels, a row of pixels after another;
  * `patches` and `inked`, its tiles; `items` and `written`, its node slots.
  */
def writeEmbeddings(setup: D2SSetup, size: String): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  val step = checkpoints.iterations.lastOption.getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}"))
  println(s"reading step $step of ${checkpoints.rootPath}")
  val params = checkpoints.load[D2STrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params

  val data = open(setup, Split.Validation)
  val sampled = Random.permutation(Axis[Plan] -> data.numSamples)(Random.Key(SampleSeed)).slice(Axis[Plan].at(0 until math.min(Sampled, data.numSamples))).toArray.sorted
  val isSampled = sampled.toSet
  val jitEmbedded = jit(embedded(Axis[Node] -> setup.nodeSlots))

  /** A batch handing over fewer plans is filled up with one it already holds, so that every batch is
    * the same shape and compiles once, and is read back short again.
    */
  val batches = data.samples
    .zipWithIndex
    .collect { case (sample, at) if isSampled(at) => sample.image }
    .grouped(EmbeddedTogether)
    .map: documents =>
      val (pictures, patches, inked, items, written) = jitEmbedded(params, stack(documents.padTo(EmbeddedTogether, documents.last), Axis[Plan]))
      val handedOver = Axis[Plan].at(0 until documents.size)
      (pictures.slice(handedOver), patches.slice(handedOver), inked.slice(handedOver), items.slice(handedOver), written.slice(handedOver))
    .toSeq

  def numpy[T <: Tuple: Labels, V](tensor: Tensor[T, V]) = TensorTree[Tensor[T, V]].toNumpyTree(tensor)
  val path = Path.of(Runs.outputDir, s"d2s-embeddings-${SyntheticFloorplans.name}-$size-${Split.Validation.fileName}-$step.npz")
  Files.createDirectories(path.getParent)
  Jax.np.savez(
    path.toString,
    drawing = numpy(Tensor1(Axis[Plan], VType[Int32]).fromArray(sampled)),
    pictures = numpy(concatenate(batches.map(_._1), Axis[Plan])),
    patches = numpy(concatenate(batches.map(_._2), Axis[Plan])),
    inked = numpy(concatenate(batches.map(_._3), Axis[Plan])),
    items = numpy(concatenate(batches.map(_._4), Axis[Plan])),
    written = numpy(concatenate(batches.map(_._5), Axis[Plan]))
  )
  println(s"${sampled.length} plans written to $path")

/** Every plan as the model read it and its encoded tiles and which of them hold ink, and the
  * prediction embeddings of the nodes the decoder writes down for it and which of its slots hold one.
  */
private def embedded(nodes: AxisExtent[Node])(
    params: D2S.Params[Float32],
    documents: Tensor4[Plan, Width, Height, Channel, Float32]
): (Tensor3[Plan, Height, Width, UInt8], Tensor3[Plan, Patch, Embedding, Float32], Tensor2[Plan, Patch, Float32], Tensor3[Plan, Node, Embedding, Float32], Tensor2[Plan, Node, Float32]) =
  val model = D2S(params)
  val slots = Tensor1(nodes.axis).fromRange(0 until nodes.size)
  documents.vmap(Axis[Plan]): document =>
    val encoded = model.encodeDocument(document)

    val documentCache = model.read(encoded)
    val (_, (writtenClasses, predictions)) = scan(Axis[Node])((model.nothingTaken(nodes), Tensor0(true)), slots):
      case ((taken, writing), slot) =>
        val (node, stillWriting, prediction) = answeredAt(model, documentCache, taken, slot, writing)
        def one[W](value: Tensor0[W]) = value.prependAxis(Axis[Node])
        val (cls, sx, sy, ex, ey) = node
        val (_, nowTaken) = model.take(documentCache, taken, slot, FloorplanNodes(one(cls), one(sx), one(sy), one(ex), one(ey)))
        ((nowTaken, stillWriting), (cls, prediction))
    val written = (!(writtenClasses elementEquals_! FloorplanClass.NoNode.id)).asFloat(VType[Float32])

    // Rounded rather than cut, so that a grey level read as a fraction comes back as itself.
    val picture = (document.squeeze(Axis[Channel]) *! 255f +! 0.5f).asInt(VType[UInt8]).swap(Axis[Width], Axis[Height])
    (picture, encoded, inkedTiles(document), predictions, written)

/** One for every tile of the plan that holds ink, in the order the encoder lays its patches out. A
  * blank tile is pure white.
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

/** How many plans of the split are embedded: drawn at random, the same ones every time. */
private val Sampled = 5000
private val SampleSeed = 0

/** How many plans are embedded together: one traced computation. */
private val EmbeddedTogether = 32

/** Axis of the plans embedded together. */
private trait Plan derives Label
