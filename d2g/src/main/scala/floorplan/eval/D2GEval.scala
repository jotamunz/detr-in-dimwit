package d2g.floorplan.eval

import d2g.*
import d2g.model.*
import d2g.floorplan.model.*
import d2g.floorplan.train.*
import d2g.floorplan.config.*
import d2s.eval.chosen
import d2s.model.PoolQuery
import d2s.floorplan.eval.answeredAt
import dataset.DrawingDataset.Split
import dataset.Metrics
import dataset.greyLevels
import dataset.RecordScoring
import dataset.Runs
import dataset.Tolerances
import deepwit.attention.KVCache
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import dimwit.tensor.Tensor4
import plotwit.*
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanDrawing
import syntheticFloorplansDataset.FloorplanEdges
import syntheticFloorplansDataset.FloorplanGraph
import syntheticFloorplansDataset.FloorplanNodes
import syntheticFloorplansDataset.FloorplanRecordBatch
import syntheticFloorplansDataset.FloorplanRelationship
import syntheticFloorplansDataset.FloorplanScoring
import syntheticFloorplansDataset.FloorplanTranscripts
import syntheticFloorplansDataset.SyntheticFloorplans
import viz.PlotTargets.websocket

import scala.language.implicitConversions

/** Scores the newest run of a setup on a whole split, and writes the metrics as one CSV, and what
  * the last checkpoint wrote down for every plan as [[FloorplanTranscripts]]. On the validation split
  * every checkpoint is scored, the last first; the test split is looked at once, by the last alone.
  */
def scoreTranscriber(setup: D2GSetup, size: String, split: Split): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val (nodes, edges) = (Axis[Node] -> setup.nodeSlots, Axis[Edge] -> setup.edgeSlots)
  val data = open(setup, split)
  println(s"pool of ${setup.queryPool} queries, ${data.numSamples} plans\n")

  val transcriber = Transcriber(nodes, edges, TranscribedTogether)

  /** Every plan of the split transcribed by a model, beside the record it was drawn from. */
  def transcribed(params: D2G.Params[Float32]): Seq[(FloorplanGraph, FloorplanGraph)] =
    data.samples
      .grouped(TranscribedTogether)
      .flatMap(batch => batch.map(sample => FloorplanGraph.of(sample.target)).zip(transcriber(params, batch.map(_.image))))
      .toSeq

  val csv = Metrics.Csv("d2g", SyntheticFloorplans.name, size, split, Runs.parameters(checkpoints.loadLatest[SavedParams].get.params), Runs.trainingSeconds(checkpoints.rootPath))
  println(s"writing to ${csv.path}")

  val scored = split match
    case Split.Test => Seq(checkpoints.iterations.last)
    case _          => checkpoints.iterations.reverse
  scored.foreach: step =>
    println(s"scoring checkpoint $step")
    val plans = transcribed(checkpoints.load[SavedParams](step).getOrElse(sys.error(s"no checkpoint $step")).params)
    val measured = Tolerances.map(tolerance => Metrics.Row(step, None, tolerance, plans.map((target, written) => FloorplanScoring.score(target, written, tolerance / setup.canvas))))
    csv.append(measured)
    if step == checkpoints.iterations.last then
      measured.foreach(row => RecordScoring.reportAt(row.tolerance, row.scored))
      println(s"transcripts written to ${FloorplanTranscripts.write("d2g", size, split, step, setup.canvas, plans)}")

/** Plots what a trained model transcribes: every plan as it was drawn, the record it was drawn
  * from, and the record the model wrote down, each drawn over the plan. Note that touching the
  * training split downloads its first part on first use.
  */
def plotTranscriber(setup: D2GSetup): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val params = checkpoints.loadLatest[SavedParams].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params
  val transcriber = Transcriber(Axis[Node] -> setup.nodeSlots, Axis[Edge] -> setup.edgeSlots)
  val rows = Seq(Split.Validation, Split.Train).flatMap: split =>
    open(setup, split)
      .samples
      .take(3)
      .zipWithIndex
      .map: (sample, index) =>
        val document = greyLevels(sample.image)
        val target = FloorplanGraph.of(sample.target)
        val transcribed = transcriber(params, sample.image)
        def drawn(plan: FloorplanGraph) = FloorplanDrawing(plan, document, Axis[Channel])
        Seq(
          plots.imagePlot(document, _.title := s"${split.fileName} $index"),
          plots.imagePlot(drawn(target), _.title := s"${split.fileName} $index — record: ${target.counted}"),
          plots.imagePlot(drawn(transcribed), _.title := s"${split.fileName} $index — transcribed: ${transcribed.counted}")
        )
      .toSeq

  display(grid(rows))

/** How many plans are transcribed together. A batch is one traced computation, so this is what
  * the split costs in calls rather than in work.
  */
private val TranscribedTogether = 32

/** Axis of the plans transcribed together. */
trait Drawing derives Label

/** What a batch of plans is transcribed into: the record each is written down as, and what it was
  * written from.
  *
  * @param patches The encoded document, as both decoders read it.
  * @param nodePredictions At every node slot, the prediction embedding of the query whose answer
  *                        the slot holds. Only the slots that hold a node stand for one: the slot 
  *                        after them holds the prediction that ended the record, and the rest are padding.
  * @param relationshipPredictions The same at every relationship slot.
  */
case class Transcription(
    record: FloorplanRecordBatch[Drawing, Node, Edge],
    patches: Tensor3[Drawing, Patch, Embedding, Float32],
    nodePredictions: Tensor3[Drawing, Node, Embedding, Float32],
    relationshipPredictions: Tensor3[Drawing, Edge, Embedding, Float32]
)

/** Transcribes floor plans: the nodes of each one at a time, then the relationships between them.
  *
  * Nothing but the document goes in, and every step reads back only what the model itself has
  * taken, so no target record can reach what is written down.
  *
  * The nodes are written as [[d2s.floorplan.eval.Transcriber]] writes them, and the relationships
  * the same way, reading the nodes as the node decoder carried them: at every slot the answer of
  * the pool with the highest log probability is written down and never looked back on.
  *
  * A batch of plans is transcribed in lockstep and as a single traced computation, so a batch costs
  * one compiled call. `drawings` is how wide that batch is; a call handing over fewer is filled up
  * with a plan it already holds and read back short again, so that every batch is the same shape.
  * A plan's nodes stop at [[FloorplanClass.NoNode]], and its relationships at
  * [[FloorplanRelationship.NoRelationship]]; the slots after hold nothing.
  */
class Transcriber(nodes: AxisExtent[Node], edges: AxisExtent[Edge], drawings: Int = 1):

  private val jitTranscribe = jit(transcribe)

  private def transcribe(params: D2G.Params[Float32], documents: Tensor4[Drawing, Width, Height, Channel, Float32]): Transcription =
    val model = D2G(params)
    val nodeSlots = Tensor1(nodes.axis).fromRange(0 until nodes.size)
    val edgeSlots = Tensor1(edges.axis).fromRange(0 until edges.size)
    val (nodeClass, startX, startY, endX, endY, relationship, subject, obj, patches, nodePredictions, relationshipPredictions) = documents.vmap(Axis[Drawing]): document =>
      val encoded = model.encodeDocument(document)
      val documentCache = model.set.read(encoded)
      val (_, (writtenNodes, carried, nodePredictions)) = scan(Axis[Node])((model.set.nothingTaken(nodes), Tensor0(true)), nodeSlots):
        case ((taken, writing), slot) =>
          val (node, stillWriting, prediction) = answeredAt(model.set, documentCache, taken, slot, writing)
          def one[W](value: Tensor0[W]) = value.prependAxis(Axis[Node])
          val (cls, sx, sy, ex, ey) = node
          val (carried, nowTaken) = model.set.take(documentCache, taken, slot, FloorplanNodes(one(cls), one(sx), one(sy), one(ex), one(ey)))
          ((nowTaken, stillWriting), (node, carried, prediction))
      val (nodeClass, startX, startY, endX, endY) = writtenNodes
      val written = NodeSource(carried, !(nodeClass elementEquals_! FloorplanClass.NoNode.id))
      val sources = model.readForEdges(encoded, written)
      val (_, ((relationship, subject, obj), relationshipPredictions)) = scan(Axis[Edge])((model.noEdgeTaken(edges), written.presentMask.any), edgeSlots):
        case ((taken, writing), slot) =>
          val (edge, stillWriting, prediction) = answeredEdgeAt(model, written, sources, taken, slot, writing)
          def one[W](value: Tensor0[W]) = value.prependAxis(Axis[Edge])
          val (related, sub, ob) = edge
          ((model.takeEdge(written, sources, taken, slot, FloorplanEdges(one(related), one(sub), one(ob))), stillWriting), (edge, prediction))
      (nodeClass, startX, startY, endX, endY, relationship, subject, obj, encoded, nodePredictions, relationshipPredictions)
    Transcription(FloorplanRecordBatch(nodeClass, startX, startY, endX, endY, relationship, subject, obj), patches, nodePredictions, relationshipPredictions)

  /** The likeliest of what the pool answers at relationship `slot` — the relationship a plan that
    * is still `writing` takes there, nothing for one that is not — whether it is still writing after
    * it, and the prediction embedding of the query that answered it.
    */
  private def answeredEdgeAt(model: D2G[Float32], nodes: NodeSource[Embedding, Float32], sources: List[EdgeSources[Float32]], taken: List[KVCache[Edge, Float32]], slot: Tensor0[Int32], writing: Tensor0[Bool]) =
    val (predictions, candidates) = model.answerEdgeAt(nodes, sources, taken, slot)
    val said = model.edgeHead.decide(candidates)
    val score = chosen(candidates.relationship) + chosen(candidates.subject) + chosen(candidates.obj)
    val likeliest = Axis[Edge].at(score.argmax(Axis[Edge]))
    val stillWriting = writing and !said.relationship.slice(likeliest).elementEquals(FloorplanRelationship.NoRelationship.id)
    val edge = (
      where(stillWriting, said.relationship.slice(likeliest), FloorplanRelationship.NoRelationship.id),
      where(stillWriting, said.subject.slice(likeliest), 0),
      where(stillWriting, said.obj.slice(likeliest), 0)
    )
    (edge, stillWriting, predictions.slice(Axis[PoolQuery].at(score.argmax(Axis[Edge]))))

  def apply(params: D2G.Params[Float32], document: Tensor3[Width, Height, Channel, Float32]): FloorplanGraph =
    apply(params, Seq(document)).head

  def apply(params: D2G.Params[Float32], documents: Seq[Tensor3[Width, Height, Channel, Float32]]): Seq[FloorplanGraph] =
    require(documents.nonEmpty, "there is nothing to transcribe")
    require(documents.size <= drawings, s"${documents.size} plans do not fit in a batch of $drawings")
    val filled = documents.padTo(drawings, documents.last)
    FloorplanGraph.of(jitTranscribe(params, stack(filled, Axis[Drawing])).record).take(documents.size)

  /** A whole batch of plans transcribed, with what they were written from. */
  def transcription(params: D2G.Params[Float32], documents: Tensor4[Drawing, Width, Height, Channel, Float32]): Transcription =
    jitTranscribe(params, documents)

private[floorplan] def open(setup: D2GSetup, split: Split) =
  SyntheticFloorplans.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(split, setup.canvas)
