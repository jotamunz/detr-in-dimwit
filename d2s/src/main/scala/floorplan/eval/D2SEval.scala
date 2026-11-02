package d2s.floorplan.eval

import d2s.*
import d2s.model.*
import d2s.floorplan.model.*
import d2s.floorplan.train.*
import d2s.floorplan.config.*
import d2s.eval.chosen
import NodeHead.NodeLogits
import dataset.DrawingDataset.Split
import dataset.Metrics
import dataset.greyLevels
import dataset.RecordScoring
import dataset.Runs
import dataset.Tolerances
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.attention.KVCache
import dimwit.*
import dimwit.tensor.Tensor4
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanClasses
import syntheticFloorplansDataset.FloorplanDrawing
import syntheticFloorplansDataset.FloorplanGraph
import syntheticFloorplansDataset.FloorplanNodes
import syntheticFloorplansDataset.FloorplanRecordBatch
import syntheticFloorplansDataset.FloorplanScoring
import syntheticFloorplansDataset.FloorplanTranscripts
import syntheticFloorplansDataset.SyntheticFloorplans

import plotwit.*
import viz.PlotTargets.websocket

import scala.language.implicitConversions

/** Scores the newest run of a setup on a whole split, and writes the metrics as one CSV, and what
  * the last checkpoint wrote down for every plan as [[FloorplanTranscripts]]. On the validation split
  * every checkpoint is scored, the last first; the test split is looked at once, by the last alone.
  */
def scoreSetTranscriber(setup: D2SSetup, size: String, split: Split): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val nodes = Axis[Node] -> setup.nodeSlots
  val data = open(setup, split)
  println(s"pool of ${setup.queryPool} queries, ${data.numSamples} drawings\n")

  val transcriber = Transcriber(nodes, TranscribedTogether)

  /** Every plan of the split transcribed by a model, beside the nodes of the record it was drawn
    * from — a set holds no relationships, so the record it is held against holds none either.
    */
  def transcribed(params: D2S.Params[Float32]): Seq[(FloorplanGraph, FloorplanGraph)] =
    data.samples
      .grouped(TranscribedTogether)
      .flatMap(batch => batch.map(sample => FloorplanGraph.of(sample.target).copy(edges = Seq.empty)).zip(transcriber(params, batch.map(_.image))))
      .toSeq

  val csv = Metrics.Csv("d2s", SyntheticFloorplans.name, size, split, Runs.parameters(checkpoints.loadLatest[D2STrainState].get.params), Runs.trainingSeconds(checkpoints.rootPath))
  println(s"writing to ${csv.path}")

  val scored = split match
    case Split.Test => Seq(checkpoints.iterations.last)
    case _          => checkpoints.iterations.reverse
  scored.foreach: step =>
    println(s"scoring checkpoint $step")
    val drawings = transcribed(checkpoints.load[D2STrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params)
    val measured = Tolerances.map(tolerance => Metrics.Row(step, None, tolerance, drawings.map((target, written) => FloorplanScoring.score(target, written, tolerance / setup.canvas))))
    csv.append(measured)
    if step == checkpoints.iterations.last then
      measured.foreach(row => RecordScoring.reportAt(row.tolerance, row.scored))
      println(s"transcripts written to ${FloorplanTranscripts.write("d2s", size, split, step, setup.canvas, drawings)}")

/** Plots what a trained model transcribes: every plan as it was drawn, the nodes of the record it
  * was drawn from, and the nodes the model wrote down, each drawn over the plan. Note that touching
  * the training split downloads its first part on first use.
  */
def plotSetTranscriber(setup: D2SSetup): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val params = checkpoints.loadLatest[D2STrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params
  val transcriber = Transcriber(Axis[Node] -> setup.nodeSlots)
  val rows = Seq(Split.Validation, Split.Train).flatMap: split =>
    open(setup, split)
      .samples
      .take(3)
      .zipWithIndex
      .map: (sample, index) =>
        val document = greyLevels(sample.image)
        val target = FloorplanGraph.of(sample.target).copy(edges = Seq.empty)
        val transcribed = transcriber(params, sample.image)
        def drawn(plan: FloorplanGraph) = FloorplanDrawing(plan, document, Axis[Channel])
        Seq(
          plots.imagePlot(document, _.title := s"${split.fileName} $index"),
          plots.imagePlot(drawn(target), _.title := s"${split.fileName} $index — record: ${target.counted}"),
          plots.imagePlot(drawn(transcribed), _.title := s"${split.fileName} $index — transcribed: ${transcribed.counted}")
        )
      .toSeq

  display(grid(rows))

/** How many drawings are transcribed together. A batch is one traced computation, so this is what
  * the split costs in calls rather than in work.
  */
private val TranscribedTogether = 32

/** Axis of the drawings transcribed together. */
private trait Drawing derives Label

/** Transcribes documents into the set of nodes they hold, one node at a time.
  *
  * Nothing but the document goes in, and every step reads back only what the model itself has
  * taken, so no target record can reach what is written down.
  *
  * A slot's queries each answer with a different remaining node, and the pool is read as a whole:
  * at every slot the answer the model gives the highest log probability is written down — the
  * score of its class plus the score of each coordinate the class places — and never looked back
  * on.
  *
  * A batch of drawings is transcribed in lockstep and as a single traced computation — the
  * encoder, every decoding step and the scorer together — so a batch costs one compiled call
  * rather than one dispatch per operation per step per drawing. `drawings` is how wide that batch
  * is; a call handing over fewer is filled up with a drawing it already holds and read back short
  * again, so that every batch is the same shape. The parameters are an argument, so the one
  * compiled computation serves every batch of every checkpoint.
  *
  * A drawing takes its nodes one slot after the other, in one `scan` over the slots, and a step
  * reads what the slots before it have taken from the decoder's caches rather than reading them
  * again. A drawing that has answered [[FloorplanClass.NoNode]] takes nothing more, and the slots it
  * would have filled hold nothing — which is what a position a record does not reach holds
  * anyway. So where a transcription stops is a value rather than a branch, and the drawings of a
  * batch are spread over by `vmap` with nothing read back to the host until all are written down.
  */
class Transcriber(nodes: AxisExtent[Node], drawings: Int = 1):

  private val jitTranscribe = jit(transcribe)

  private def transcribe(params: D2S.Params[Float32], documents: Tensor4[Drawing, Width, Height, Channel, Float32]): FloorplanRecordBatch[Drawing, Node, Relationship] =
    val model = D2S(params)
    val slots = Tensor1(nodes.axis).fromRange(0 until nodes.size)
    val (nodeClass, startX, startY, endX, endY) = documents.vmap(Axis[Drawing]): document =>
      val documentCache = model.read(model.encodeDocument(document))
      val (_, written) = scan(Axis[Node])((model.nothingTaken(nodes), Tensor0(true)), slots):
        case ((taken, writing), slot) =>
          val (node, stillWriting, _) = answeredAt(model, documentCache, taken, slot, writing)
          def one[W](value: Tensor0[W]) = value.prependAxis(Axis[Node])
          val (cls, sx, sy, ex, ey) = node
          val taking = FloorplanNodes(one(cls), one(sx), one(sy), one(ex), one(ey))
          val (_, nowTaken) = model.take(documentCache, taken, slot, taking)
          ((nowTaken, stillWriting), node)
      written
    val noRelationships = Tensor(Shape2(documents.shape.extent(Axis[Drawing]), Axis[Relationship] -> 0), VType[Int32]).fill(0)
    FloorplanRecordBatch(nodeClass, startX, startY, endX, endY, noRelationships, noRelationships, noRelationships)

  def apply(params: D2S.Params[Float32], document: Tensor3[Width, Height, Channel, Float32]): FloorplanGraph =
    apply(params, Seq(document)).head

  def apply(params: D2S.Params[Float32], documents: Seq[Tensor3[Width, Height, Channel, Float32]]): Seq[FloorplanGraph] =
    require(documents.nonEmpty, "there is nothing to transcribe")
    require(documents.size <= drawings, s"${documents.size} drawings do not fit in a batch of $drawings")
    val filled = documents.padTo(drawings, documents.last)
    FloorplanGraph.of(jitTranscribe(params, stack(filled, Axis[Drawing]))).take(documents.size)

private[floorplan] def open(setup: D2SSetup, split: Split) =
  SyntheticFloorplans.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(split, setup.canvas)

/** The likeliest of what the pool answers at `slot` — the node a drawing that is still `writing`
  * takes there, nothing for one that is not — whether it is still writing after it, and the
  * prediction embedding that answer was read from.
  */
def answeredAt(model: D2S[Float32], documentCache: List[KVCache[Patch, Float32]], taken: List[KVCache[Node, Float32]], slot: Tensor0[Int32], writing: Tensor0[Bool]) =
  val predictions = model.predictionsAt(documentCache, taken, slot).relabel(Axis[PoolQuery] -> Axis[Node])
  val (said, score) = answered(model.nodeHead, model.nodeHead(predictions))
  val likeliest = Axis[Node].at(score.argmax(Axis[Node]))
  val stillWriting = writing and !said.nodeClass.slice(likeliest).elementEquals(FloorplanClass.NoNode.id)
  val node = (
    where(stillWriting, said.nodeClass.slice(likeliest), FloorplanClass.NoNode.id),
    where(stillWriting, said.startX.slice(likeliest), 0),
    where(stillWriting, said.startY.slice(likeliest), 0),
    where(stillWriting, said.endX.slice(likeliest), 0),
    where(stillWriting, said.endY.slice(likeliest), 0)
  )
  (node, stillWriting, predictions.slice(likeliest))

/** The node every position along `Node` answers with, and the log probability of that answer: the
  * score of its class plus the score of each coordinate of its two ends.
  */
def answered(scorer: NodeHead[Float32], logits: NodeLogits[Float32]): (FloorplanNodes[Node], Tensor1[Node, Float32]) =
  val decided = scorer.decide(logits)
  val score = chosen(logits.nodeClass) + chosen(logits.startX) + chosen(logits.startY) + chosen(logits.endX) + chosen(logits.endY)
  (decided, score)
