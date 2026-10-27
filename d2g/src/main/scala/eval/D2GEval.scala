package d2g.eval

import d2g.*
import d2g.model.*
import d2g.train.*
import d2g.config.*
import d2s.model.*
import d2s.eval.answeredAt
import d2s.eval.chosen
import d2s.eval.counted
import d2s.eval.describe
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.EdgeClass
import dataset.Metrics
import dataset.NodeClass
import dataset.greyLevels
import dataset.RecordBatch
import dataset.RecordEdges
import dataset.RecordNodes
import dataset.RecordDrawing
import dataset.RecordGraph
import dataset.RecordScoring
import dataset.Runs
import dataset.Tolerances
import dataset.Transcripts
import dataset.at
import dataset.report
import deepwit.attention.KVCache
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import dimwit.tensor.Tensor4
import plotwit.*
import viz.PlotTargets.websocket

import scala.language.implicitConversions

/** Scores every checkpoint of the newest run of a setup on the whole validation split, the last
  * checkpoint first, and writes the metrics as one CSV, and what the last checkpoint wrote down
  * for every drawing as [[Transcripts]].
  */
def scoreTranscriber(setup: D2GSetup, size: String): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val (nodes, edges) = (Axis[Node] -> setup.nodeSlots, Axis[Edge] -> setup.edgeSlots)
  val data = open(setup, Split.Validation)
  println(s"pool of ${setup.queryPool} queries, ${data.numSamples} drawings\n")

  val transcriber = Transcriber(nodes, edges, TranscribedTogether)

  /** Every drawing of the split transcribed by a model, beside the record it was rendered from. */
  def transcribed(params: D2G.Params[Float32]): Seq[(RecordGraph, RecordGraph)] =
    data.samples
      .grouped(TranscribedTogether)
      .flatMap(batch => batch.map(sample => RecordGraph.of(sample.target)).zip(transcriber(params, batch.map(_.image))))
      .toSeq

  val csv = Metrics.Csv("d2g", setup.corpus, size, Split.Validation, Runs.parameters(checkpoints.loadLatest[D2GTrainState].get.params), Runs.trainingSeconds(checkpoints.rootPath))
  println(s"writing to ${csv.path}")

  checkpoints.iterations.reverse.foreach: step =>
    println(s"scoring checkpoint $step")
    val drawings = transcribed(checkpoints.load[D2GTrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params)
    val measured = Tolerances.map(tolerance => Metrics.Row(step, None, tolerance, drawings.map((target, written) => RecordScoring.score(target, written, tolerance / setup.corpus.canvas))))
    csv.append(measured)
    if step == checkpoints.iterations.last then
      measured.foreach(row => RecordScoring.reportAt(row.tolerance, row.scored))
      println(s"transcripts written to ${Transcripts.write("d2g", setup.corpus, size, Split.Validation, step, drawings)}")

/** Plots what a trained model transcribes.
  *
  * A record has no drawing of its own, so it is drawn as the objects it stands for, and its
  * relationships are printed. Note that touching the training split downloads the whole of it on
  * first use.
  */
def plotTranscriber(setup: D2GSetup): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val params = checkpoints.loadLatest[D2GTrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params
  val (nodes, edges) = (Axis[Node] -> setup.nodeSlots, Axis[Edge] -> setup.edgeSlots)
  val transcriber = Transcriber(nodes, edges)
  val rows = Seq(Split.Validation, Split.Train).flatMap: split =>
    val data = open(setup, split)
    data
      .samples
      .take(3)
      .zipWithIndex
      .map: (sample, index) =>
        val document = greyLevels(sample.image)
        val target = RecordGraph.of(sample.target)
        val transcribed = transcriber(params, sample.image)
        println(s"${split.fileName} $index target:      ${describe(target)}")
        println(s"${split.fileName} $index transcribed: ${describe(transcribed)}")
        def drawn(record: RecordGraph) = RecordDrawing(record, document, Axis[Channel])
        Seq(
          plots.imagePlot(document, _.title := s"${split.fileName} $index"),
          plots.imagePlot(drawn(target), _.title := s"${split.fileName} $index — record: ${counted(target)}"),
          plots.imagePlot(drawn(transcribed), _.title := s"${split.fileName} $index — transcribed: ${counted(transcribed)}")
        )
      .toSeq

  display(grid(rows))

/** How many drawings are transcribed together. A batch is one traced computation, so this is what
  * the split costs in calls rather than in work.
  */
private val TranscribedTogether = 32

/** Axis of the drawings transcribed together. */
private trait Drawing derives Label

/** Transcribes documents: the nodes of each one at a time, then the relationships between them.
  *
  * Nothing but the document goes in, and every step reads back only what the model itself has
  * taken, so no target record can reach what is written down.
  *
  * A slot's queries each answer with a different remaining node, and the pool is read as a whole:
  * at every slot the answer the model gives the highest log probability is written down — the
  * score of its class plus the score of each coordinate the class places — and never looked back
  * on. The relationships are written the same way, reading the nodes as the node decoder carried
  * them.
  *
  * A batch of drawings is transcribed in lockstep and as a single traced computation — the
  * encoder, every decoding step and both scorers together — so a batch costs one compiled call
  * rather than one dispatch per operation per step per drawing. `drawings` is how wide that batch
  * is; a call handing over fewer is filled up with a drawing it already holds and read back short
  * again, so that every batch is the same shape. The parameters are an argument, so the one
  * compiled computation serves every batch of every checkpoint.
  *
  * A drawing takes its nodes one slot after the other, in one `scan` over the slots, and then its
  * relationships in a second one; a step reads what the slots before it have taken from the
  * decoders' caches rather than reading them again. A drawing that has answered
  * [[NodeClass.NoNode]] takes nothing more, and the slots it would have filled hold nothing — which
  * is what a position a record does not reach holds anyway. Its relationships stop the same way at
  * [[EdgeClass.NoEdge]].
  */
class Transcriber(nodes: AxisExtent[Node], edges: AxisExtent[Edge], drawings: Int = 1):

  private val jitTranscribe = jit(transcribe)

  private def transcribe(params: D2G.Params[Float32], documents: Tensor4[Drawing, Width, Height, Channel, Float32]): RecordBatch[Drawing, Node, Edge] =
    val model = D2G(params)
    val nodeSlots = Tensor1(nodes.axis).fromRange(0 until nodes.size)
    val edgeSlots = Tensor1(edges.axis).fromRange(0 until edges.size)
    val (nodeClass, construction, startX, startY, endX, endY, midX, midY, edgeClass, subject, obj) = documents.vmap(Axis[Drawing]): document =>
      val encoded = model.encodeDocument(document)
      val documentCache = model.set.read(encoded)
      val (_, (writtenNodes, carried)) = scan(Axis[Node])((model.set.nothingTaken(nodes), Tensor0(true)), nodeSlots):
        case ((taken, writing), slot) =>
          val (node, stillWriting, _) = answeredAt(model.set, documentCache, taken, slot, writing)
          def one[W](value: Tensor0[W]) = value.prependAxis(Axis[Node])
          val (cls, isConstruction, sx, sy, ex, ey, mx, my) = node
          val taking = RecordNodes(one(cls), one(isConstruction), one(sx), one(sy), one(ex), one(ey), one(mx), one(my))
          val (carried, nowTaken) = model.set.take(documentCache, taken, slot, taking)
          ((nowTaken, stillWriting), (node, carried))
      val (nodeClass, construction, startX, startY, endX, endY, midX, midY) = writtenNodes
      val sources = model.readForEdges(encoded, NodeSource(carried, !(nodeClass elementEquals_! NodeClass.NoNode.id)))
      val (_, (edgeClass, subject, obj)) = scan(Axis[Edge])((model.noEdgeTaken(edges), Tensor0(true)), edgeSlots):
        case ((taken, writing), slot) =>
          val (edge, stillWriting) = answeredEdgeAt(model, sources, taken, slot, writing)
          def one[W](value: Tensor0[W]) = value.prependAxis(Axis[Edge])
          val (cls, sub, ob) = edge
          ((model.takeEdge(sources, taken, slot, RecordEdges(one(cls), one(sub), one(ob))), stillWriting), edge)
      (nodeClass, construction, startX, startY, endX, endY, midX, midY, edgeClass, subject, obj)
    RecordBatch(nodeClass, construction, startX, startY, endX, endY, midX, midY, edgeClass, subject, obj)

  /** The likeliest of what the pool answers at relationship `slot` — the relationship a drawing
    * that is still `writing` takes there, nothing for one that is not — and whether it is still
    * writing after it.
    */
  private def answeredEdgeAt(model: D2G[Float32], sources: List[EdgeSources[Float32]], taken: List[KVCache[Edge, Float32]], slot: Tensor0[Int32], writing: Tensor0[Bool]) =
    val candidates = model.answerEdgeAt(sources, taken, slot)
    val said = model.edgeHead.decide(candidates)
    val score = chosen(candidates.edgeClass) + chosen(candidates.subject) + chosen(candidates.obj)
    val likeliest = Axis[Edge].at(score.argmax(Axis[Edge]))
    val stillWriting = writing and !said.edgeClass.slice(likeliest).elementEquals(EdgeClass.NoEdge.id)
    val edge = (
      where(stillWriting, said.edgeClass.slice(likeliest), EdgeClass.NoEdge.id),
      where(stillWriting, said.subject.slice(likeliest), 0),
      where(stillWriting, said.obj.slice(likeliest), 0)
    )
    (edge, stillWriting)

  def apply(params: D2G.Params[Float32], document: Tensor3[Width, Height, Channel, Float32]): RecordGraph =
    apply(params, Seq(document)).head

  def apply(params: D2G.Params[Float32], documents: Seq[Tensor3[Width, Height, Channel, Float32]]): Seq[RecordGraph] =
    require(documents.nonEmpty, "there is nothing to transcribe")
    require(documents.size <= drawings, s"${documents.size} drawings do not fit in a batch of $drawings")
    val filled = documents.padTo(drawings, documents.last)
    RecordGraph.of(jitTranscribe(params, stack(filled, Axis[Drawing]))).take(documents.size)

private def open(setup: D2GSetup, split: Split) =
  DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(split)
