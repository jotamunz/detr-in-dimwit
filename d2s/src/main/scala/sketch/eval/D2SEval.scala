package d2s.sketch.eval

import d2s.*
import d2s.sketch.*
import d2s.model.*
import d2s.sketch.model.*
import d2s.train.*
import d2s.sketch.train.*
import d2s.config.*
import d2s.sketch.config.*
import d2s.eval.chosen
import NodeHead.NodeLogits
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.Metrics
import dataset.NodeClass
import dataset.NodeClasses
import dataset.greyLevels
import dataset.RecordBatch
import dataset.RecordNodes
import dataset.RecordDrawing
import dataset.RecordGraph
import dataset.RecordScoring
import dataset.Runs
import dataset.Tolerances
import dataset.Transcripts
import dataset.at
import dataset.report
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.attention.KVCache
import dimwit.*
import dimwit.tensor.Tensor4
import plotwit.*
import viz.PlotTargets.websocket

import scala.language.implicitConversions

/** Scores the newest run of a setup on a whole split, and writes the metrics as one CSV, and what
  * the last checkpoint wrote down for every drawing as [[Transcripts]]. On the validation split
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

  /** Every drawing of the split transcribed by a model, beside the nodes of the record it was
    * rendered from — a set holds no relationships, so the record it is held against holds none
    * either.
    */
  def transcribed(params: D2S.Params[Float32]): Seq[(RecordGraph, RecordGraph)] =
    data.samples
      .grouped(TranscribedTogether)
      .flatMap(batch => batch.map(sample => RecordGraph.of(sample.target).copy(edges = Seq.empty)).zip(transcriber(params, batch.map(_.image))))
      .toSeq

  val csv = Metrics.Csv("d2s", setup.corpus.name, size, split, Runs.parameters(checkpoints.loadLatest[D2STrainState].get.params), Runs.trainingSeconds(checkpoints.rootPath))
  println(s"writing to ${csv.path}")

  val scored = split match
    case Split.Test => Seq(checkpoints.iterations.last)
    case _          => checkpoints.iterations.reverse
  scored.foreach: step =>
    println(s"scoring checkpoint $step")
    val drawings = transcribed(checkpoints.load[D2STrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params)
    val measured = Tolerances.map(tolerance => Metrics.Row(step, None, tolerance, drawings.map((target, written) => RecordScoring.score(target, written, tolerance / setup.corpus.canvas))))
    csv.append(measured)
    if step == checkpoints.iterations.last then
      measured.foreach(row => RecordScoring.reportAt(row.tolerance, row.scored))
      println(s"transcripts written to ${Transcripts.write("d2s", setup.corpus, size, split, step, drawings)}")

/** Plots what a trained model transcribes.
  *
  * A record has no drawing of its own, so it is drawn as the objects it stands for. Note that
  * touching the training split downloads the whole of it on first use.
  */
def plotSetTranscriber(setup: D2SSetup): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val params = checkpoints.loadLatest[D2STrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params
  val transcriber = Transcriber(Axis[Node] -> setup.nodeSlots)
  val rows = Seq(Split.Validation, Split.Train).flatMap: split =>
    val data = open(setup, split)
    data
      .samples
      .take(3)
      .zipWithIndex
      .map: (sample, index) =>
        val document = greyLevels(sample.image)
        val target = RecordGraph.of(sample.target).copy(edges = Seq.empty)
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
  * again. A drawing that has answered [[NodeClass.NoNode]] takes nothing more, and the slots it
  * would have filled hold nothing — which is what a position a record does not reach holds
  * anyway. So where a transcription stops is a value rather than a branch, and the drawings of a
  * batch are spread over by `vmap` with nothing read back to the host until all are written down.
  */
class Transcriber(nodes: AxisExtent[Node], drawings: Int = 1):

  private val jitTranscribe = jit(transcribe)

  private def transcribe(params: D2S.Params[Float32], documents: Tensor4[Drawing, Width, Height, Channel, Float32]): RecordBatch[Drawing, Node, Relationship] =
    val model = D2S(params)
    val slots = Tensor1(nodes.axis).fromRange(0 until nodes.size)
    val (nodeClass, construction, startX, startY, endX, endY, midX, midY) = documents.vmap(Axis[Drawing]): document =>
      val documentCache = model.read(model.encodeDocument(document))
      val (_, written) = scan(Axis[Node])((model.nothingTaken(nodes), Tensor0(true)), slots):
        case ((taken, writing), slot) =>
          val (node, stillWriting) = answeredAt(model, documentCache, taken, slot, writing)
          def one[W](value: Tensor0[W]) = value.prependAxis(Axis[Node])
          val (cls, isConstruction, sx, sy, ex, ey, mx, my) = node
          val taking = RecordNodes(one(cls), one(isConstruction), one(sx), one(sy), one(ex), one(ey), one(mx), one(my))
          val (_, nowTaken) = model.take(documentCache, taken, slot, taking)
          ((nowTaken, stillWriting), node)
      written
    val noRelationships = Tensor(Shape2(documents.shape.extent(Axis[Drawing]), Axis[Relationship] -> 0), VType[Int32]).fill(0)
    RecordBatch(nodeClass, construction, startX, startY, endX, endY, midX, midY, noRelationships, noRelationships, noRelationships)

  def apply(params: D2S.Params[Float32], document: Tensor3[Width, Height, Channel, Float32]): RecordGraph =
    apply(params, Seq(document)).head

  def apply(params: D2S.Params[Float32], documents: Seq[Tensor3[Width, Height, Channel, Float32]]): Seq[RecordGraph] =
    require(documents.nonEmpty, "there is nothing to transcribe")
    require(documents.size <= drawings, s"${documents.size} drawings do not fit in a batch of $drawings")
    val filled = documents.padTo(drawings, documents.last)
    RecordGraph.of(jitTranscribe(params, stack(filled, Axis[Drawing]))).take(documents.size)

private def open(setup: D2SSetup, split: Split) =
  DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(split)

/** The likeliest of what the pool answers at `slot` — the node a drawing that is still `writing`
  * takes there, nothing for one that is not — and whether it is still writing after it.
  */
def answeredAt(model: D2S[Float32], documentCache: List[KVCache[Patch, Float32]], taken: List[KVCache[Node, Float32]], slot: Tensor0[Int32], writing: Tensor0[Bool]) =
  val (said, score) = answered(model.nodeHead, model.answerAt(documentCache, taken, slot))
  val likeliest = Axis[Node].at(score.argmax(Axis[Node]))
  val stillWriting = writing and !said.nodeClass.slice(likeliest).elementEquals(NodeClass.NoNode.id)
  val node = (
    where(stillWriting, said.nodeClass.slice(likeliest), NodeClass.NoNode.id),
    where(stillWriting, said.construction.slice(likeliest), 0),
    where(stillWriting, said.startX.slice(likeliest), 0),
    where(stillWriting, said.startY.slice(likeliest), 0),
    where(stillWriting, said.endX.slice(likeliest), 0),
    where(stillWriting, said.endY.slice(likeliest), 0),
    where(stillWriting, said.midX.slice(likeliest), 0),
    where(stillWriting, said.midY.slice(likeliest), 0)
  )
  (node, stillWriting)

/** The node every position along `Node` answers with, and the log probability of that answer: the
  * score of its class plus the score of each coordinate the class places.
  */
def answered(scorer: NodeHead[Float32], logits: NodeLogits[Float32]): (RecordNodes[Node], Tensor1[Node, Float32]) =
  val decided = scorer.decide(logits)
  def carries(holds: NodeClass => Boolean) = NodeClass.indicator(VType[Float32])(holds).slice(Axis[NodeClasses].at(decided.nodeClass))
  val score = chosen(logits.nodeClass) + chosen(logits.construction) + chosen(logits.startX) + chosen(logits.startY) +
    (chosen(logits.endX) + chosen(logits.endY)) * carries(_.numPoints > 1) +
    (chosen(logits.midX) + chosen(logits.midY)) * carries(_.numPoints > 2)
  (decided, score)

/** How many nodes of each class a record holds, for the header of a drawing of it. */
def counted(record: RecordGraph): String =
  val held = NodeClass.values.toSeq.flatMap: nodeClass =>
    val count = record.nodes.count(_.nodeClass == nodeClass)
    Option.when(count > 0)(s"$count ${nodeClass.toString.toLowerCase}${if count == 1 then "" else "s"}")
  if held.isEmpty then "nothing" else held.mkString(", ")

def describe(record: RecordGraph): String =
  val nodes = record.nodes.map(node => s"${node.nodeClass}(${node.points.map(point => f"${point.x}%.3f, ${point.y}%.3f").mkString("; ")})")
  val edges = record.edges.map(edge => s"${edge.edgeClass}(${edge.subject}, ${edge.obj})")
  (nodes ++ edges).mkString(", ")
